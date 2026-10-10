//! Prompt rendering, tokenization and the branch layout, following OmniJev @ 14dbec4
//! (`MSO1._prompt`, `MSO1._encode_many`, `MSO1.ask_branch`, `mso/branch.py`).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::image_decode::{DecodeLimits, decode_rgb8};
use omni_qwen3_5_native::image_preprocess::{
    ImageLimits, ProcessedImage, preprocess_rgb8_with_limits,
};
use omni_qwen3_5_native::inputs::image_positions;
use tokenizers::Tokenizer;

use crate::contract::{self, Image, Question, Request};

pub const IMAGE_TOKEN: u32 = 248_056;
pub const OPTION_OPEN: u32 = 248_077;
pub const OPTION_CLOSE: u32 = 248_078;
/// `MSO1`'s default image budget, which overrides the processor config's 401,408.
pub const MAX_PIXELS: usize = 768 * 28 * 28;
pub const MIN_PIXELS: usize = 65_536;
/// A single question's tokens: prompt, image and every option.
pub const MAX_SEQUENCE: usize = 8192;
/// Tokens the reference processes for a request: its prefix once, then every row.
pub const MAX_PROCESSED: usize = 65_536;
/// Tokens the eager worker runs for a request, the prefix again with every row, until
/// the rows reuse the prefix: about 20 s on an RTX 6000 Ada.
pub const MAX_PASS_TOKENS: usize = 262_144;
/// Option-text tokens whose log-probabilities a request reads; each takes a log-softmax
/// over the vocabulary on the CPU.
pub const MAX_TARGETS: usize = 16_384;

const PAD: &str = "<|image_pad|>";

/// The processor's resize and patches for `MSO1`'s budget; the source limits are the
/// contract's.
const IMAGE_LIMITS: ImageLimits = ImageLimits {
    max_source_side: contract::MAX_IMAGE_SIDE,
    max_source_pixels: contract::MAX_IMAGE_SIDE * contract::MAX_IMAGE_SIDE,
    min_pixels: MIN_PIXELS,
    max_pixels: MAX_PIXELS,
    max_patches: MAX_PIXELS / 256,
};

/// Decode the request's image to RGB as Pillow's `convert("RGB")` does, then
/// `preprocess` it.
pub fn pixels(image: &Image) -> Result<ProcessedImage> {
    let decoded = decode_rgb8(
        &image.bytes,
        image.format,
        &DecodeLimits {
            max_side: contract::MAX_IMAGE_SIDE,
            max_pixels: contract::MAX_IMAGE_SIDE * contract::MAX_IMAGE_SIDE,
            max_aspect: 200,
            // a 16-bit RGBA image at the largest size
            max_alloc: (contract::MAX_IMAGE_SIDE * contract::MAX_IMAGE_SIDE * 8) as u64,
        },
    )?;
    preprocess(decoded.width, decoded.height, &decoded.rgb)
}

/// Resize, normalize and pack interleaved RGB8 as the reference's
/// `Qwen2VLImageProcessor` does with `MSO1`'s budget.
pub fn preprocess(width: usize, height: usize, rgb: &[u8]) -> Result<ProcessedImage> {
    preprocess_rgb8_with_limits(width, height, rgb, &IMAGE_LIMITS)
}

/// Image tokens after the 2×2 spatial merge.
pub fn image_tokens(grid: [usize; 3]) -> usize {
    grid[1] / 2 * (grid[2] / 2)
}

/// Python's `str.isspace`, which Jinja's `trim` uses.
fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// The chat-template text of one question with its option blocks appended, with one
/// image placeholder that tokenization expands.
pub fn prompt(question: &Question) -> String {
    // The template trims the user content, which starts with the image, so only the
    // instructions' trailing whitespace goes.
    let content = format!(
        "<|vision_start|>{PAD}<|vision_end|>{}",
        question.instructions.trim_end_matches(python_space)
    );
    let mut text =
        format!("<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n");
    for option in &question.options {
        text += "<|opt|>";
        text += option;
        text += "<|/opt|>";
    }
    text
}

/// One (question, option) row: the rest of the question after the shared prefix,
/// then one option block.
#[derive(Debug, PartialEq)]
pub struct Row {
    pub tokens: Vec<u32>,
    /// The token before the option marker, where the question state `zq` is read.
    pub zq: usize,
    /// The closing marker, where the option state `u` is read; the row's last token.
    pub u: usize,
    /// For each option-text token: the position that predicts it and the token.
    pub targets: Vec<(usize, u32)>,
    /// The first option-text token, predicted from `zq`.
    pub first: Option<u32>,
}

pub struct PreparedQuestion {
    /// The single-question sequence: prompt, image and every option block.
    pub token_ids: Vec<u32>,
    pub rows: Vec<Row>,
}

/// The reference's split of a request into a shared prefix and rows.
pub struct Layout {
    /// The prefix length: the questions' common token prefix, cut at least one token
    /// before any question's first option marker.
    pub prefix: usize,
    /// The rotary position of the first token after the prefix.
    pub next_position: i64,
    /// The reference's input-token count: the prefix once, then every row.
    pub processed_tokens: usize,
}

/// Option-marker positions in a single-question sequence.
pub fn spans(ids: &[u32]) -> (Vec<usize>, Vec<usize>) {
    let at = |token| (0..ids.len()).filter(|&i| ids[i] == token).collect();
    (at(OPTION_OPEN), at(OPTION_CLOSE))
}

/// Rows and readouts of every question, as `ask_branch` and `branch_forward` build them.
pub fn layout(sequences: &[Vec<u32>], grid: [usize; 3]) -> Result<(Layout, Vec<Vec<Row>>)> {
    ensure!(!sequences.is_empty(), "no questions");
    let shortest = sequences.iter().map(Vec::len).min().unwrap_or(0);
    let mut prefix = 0;
    while prefix < shortest && sequences.iter().all(|s| s[prefix] == sequences[0][prefix]) {
        prefix += 1;
    }
    let mut spans_of = Vec::with_capacity(sequences.len());
    for ids in sequences {
        let (opens, closes) = spans(ids);
        ensure!(
            !opens.is_empty()
                && opens.len() == closes.len()
                && opens.iter().zip(&closes).all(|(o, c)| o < c)
                && closes.iter().zip(&opens[1..]).all(|(c, o)| c < o),
            "every option needs one opening and one closing marker, in order"
        );
        // Every row keeps at least the token before its option marker.
        ensure!(opens[0] >= 2, "options must follow the prompt");
        prefix = prefix.min(opens[0] - 1);
        spans_of.push((opens, closes));
    }
    let prefix = prefix.max(1);
    let mut rows = Vec::with_capacity(sequences.len());
    let mut processed = prefix;
    for (ids, (opens, closes)) in sequences.iter().zip(&spans_of) {
        let head = &ids[prefix..opens[0]];
        let question_rows: Vec<Row> = opens
            .iter()
            .zip(closes)
            .map(|(&o, &c)| {
                let tokens: Vec<u32> = head.iter().chain(&ids[o..=c]).copied().collect();
                let open = head.len();
                let last = tokens.len() - 1;
                let inner = open + 1..last;
                Row {
                    zq: open - 1,
                    u: last,
                    targets: inner.clone().map(|t| (t - 1, tokens[t])).collect(),
                    first: inner.clone().next().map(|t| tokens[t]),
                    tokens,
                }
            })
            .collect();
        processed += question_rows.iter().map(|r| r.tokens.len()).sum::<usize>();
        rows.push(question_rows);
    }
    let positions = image_positions(&sequences[0], IMAGE_TOKEN, grid)?;
    let next_position = positions
        .iter()
        .map(|axis| axis[prefix - 1])
        .max()
        .unwrap_or(0)
        + 1;
    Ok((
        Layout {
            prefix,
            next_position,
            processed_tokens: processed,
        },
        rows,
    ))
}

pub struct Processor {
    tokenizer: Tokenizer,
}

/// A request ready for execution; `questions` keeps identity and order for finishing.
pub struct PreparedRequest {
    pub pixels: ProcessedImage,
    pub grid: [usize; 3],
    pub questions: Vec<Question>,
    pub inputs: Vec<PreparedQuestion>,
    pub layout: Layout,
    /// Time spent decoding, preprocessing and tokenizing, which the reference's latency
    /// includes.
    pub preparation_seconds: f64,
}

impl Processor {
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        for (text, id) in [
            ("<|image_pad|>", IMAGE_TOKEN),
            ("<|opt|>", OPTION_OPEN),
            ("<|/opt|>", OPTION_CLOSE),
        ] {
            ensure!(
                tokenizer.token_to_id(text) == Some(id),
                "tokenizer does not map {text} to {id}"
            );
        }
        Ok(Self { tokenizer })
    }

    /// The single-question sequence: the prompt's text tokens around the image's
    /// expanded placeholder.
    pub fn tokenize(&self, question: &Question, grid: [usize; 3]) -> Result<Vec<u32>> {
        let text = prompt(question);
        let (before, after) = text.split_once(PAD).context("prompt has no image")?;
        let encode = |part: &str| -> Result<Vec<u32>> {
            Ok(self
                .tokenizer
                .encode(part, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec())
        };
        let mut ids = encode(before)?;
        ids.extend(std::iter::repeat_n(IMAGE_TOKEN, image_tokens(grid)));
        ids.extend(encode(after)?);
        Ok(ids)
    }

    /// Validate and lay out the whole request before any device work.
    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedRequest> {
        let start = std::time::Instant::now();
        let Request { image, questions } = contract::compile(raw)?;
        let pixels = pixels(&image)?;
        let grid = pixels.image_grid_thw;
        let sequences = questions
            .iter()
            .map(|q| {
                let ids = self.tokenize(q, grid)?;
                ensure!(
                    ids.len() <= MAX_SEQUENCE,
                    "question {}: {} tokens exceeds {MAX_SEQUENCE}",
                    q.id,
                    ids.len()
                );
                Ok(ids)
            })
            .collect::<Result<Vec<_>>>()?;
        let (layout, rows) = layout(&sequences, grid)?;
        ensure!(
            layout.processed_tokens <= MAX_PROCESSED,
            "request needs {} tokens, above {MAX_PROCESSED}",
            layout.processed_tokens
        );
        let pass_tokens: usize = rows
            .iter()
            .flatten()
            .map(|r| layout.prefix + r.tokens.len())
            .sum();
        ensure!(
            pass_tokens <= MAX_PASS_TOKENS,
            "request needs {pass_tokens} tokens in row passes, above {MAX_PASS_TOKENS}"
        );
        // Score's ordinal head reads no log-probabilities.
        let targets: usize = questions
            .iter()
            .zip(&rows)
            .filter(|(q, _)| q.kind != contract::Kind::Score)
            .flat_map(|(_, rows)| rows)
            .map(|r| r.targets.len() + usize::from(r.first.is_some()))
            .sum();
        ensure!(
            targets <= MAX_TARGETS,
            "request reads {targets} option-text tokens, above {MAX_TARGETS}"
        );
        let inputs = sequences
            .into_iter()
            .zip(rows)
            .map(|(token_ids, rows)| PreparedQuestion { token_ids, rows })
            .collect();
        Ok(PreparedRequest {
            pixels,
            grid,
            questions,
            inputs,
            layout,
            preparation_seconds: start.elapsed().as_secs_f64(),
        })
    }
}
