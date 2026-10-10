//! Single-image native prompt preparation and end-to-end execution.

use crate::contract::{LETTERS, Question, SYSTEM_PROMPT};
use anyhow::{Result, ensure};

pub const MAX_TOKENS: usize = 4096;

pub fn chat_image(question: &Question, image_tokens: usize) -> Result<String> {
    ensure!(
        (1..MAX_TOKENS).contains(&image_tokens),
        "invalid image token count"
    );
    ensure!(
        (1..=26).contains(&question.labels.len()),
        "invalid option count"
    );
    let goal = if question.goal.is_empty() {
        String::new()
    } else {
        format!("Goal: {}\n\n", question.goal)
    };
    let options = LETTERS
        .chars()
        .zip(&question.labels)
        .map(|(letter, label)| format!("{letter}. Decision \"{label}\" -> select"))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n<|vision_start|>{}<|vision_end|>{goal}App: Cua Driver\nTask family: closed-candidate decision\n\nThe current screenshot is attached.\n\nOptions:\n{options}\n\nAnswer with a single letter.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|image_pad|>".repeat(image_tokens)
    ))
}

/// A prepared single-image prompt. Every prompt in a request is prepared before
/// running the shared image encoder.
pub struct ImagePrompt {
    pub token_ids: Vec<u32>,
    pub image_token_indices: Vec<usize>,
    pub position_ids: [Vec<i64>; 3],
}

pub fn prepare_prompt(
    tokenizer: &tokenizers::Tokenizer,
    question: &Question,
    grid: [usize; 3],
    image_token: u32,
) -> Result<ImagePrompt> {
    let count = grid[1]
        .checked_mul(grid[2])
        .ok_or_else(|| anyhow::anyhow!("grid overflow"))?
        / 4;
    let encoded = tokenizer
        .encode(chat_image(question, count)?, false)
        .map_err(anyhow::Error::msg)?;
    let token_ids = encoded.get_ids().to_vec();
    ensure!(
        !token_ids.is_empty() && token_ids.len() <= MAX_TOKENS,
        "processed prompt exceeds {MAX_TOKENS} tokens or is empty"
    );
    let position_ids = crate::inputs::image_positions(&token_ids, image_token, grid)?;
    let image_token_indices = token_ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (id == image_token).then_some(i))
        .collect();
    Ok(ImagePrompt {
        token_ids,
        image_token_indices,
        position_ids,
    })
}

/// Validate public RGB callers as well as the JSON mapper before GPU work.
pub(crate) fn validate_questions(questions: &[Question]) -> Result<()> {
    ensure!(
        (1..=8).contains(&questions.len()),
        "expected 1 to 8 questions"
    );
    let mut names = std::collections::HashSet::new();
    for q in questions {
        ensure!(
            (1..=256).contains(&q.name.chars().count()) && names.insert(&q.name),
            "invalid or duplicate question name"
        );
        ensure!(
            (1..=26).contains(&q.keys.len()) && q.keys.len() == q.labels.len(),
            "option key/label counts must agree and be 1 to 26"
        );
        let mut keys = std::collections::HashSet::new();
        ensure!(
            q.keys
                .iter()
                .all(|k| (1..=256).contains(&k.chars().count()) && keys.insert(k)),
            "invalid or duplicate option key"
        );
        ensure!(
            q.goal.chars().count() + q.labels.iter().map(|s| s.chars().count()).sum::<usize>()
                <= 16384,
            "combined question text exceeds 16384 characters"
        );
        for text in std::iter::once(&q.goal).chain(&q.labels) {
            ensure!(
                ![
                    "<|image_pad|>",
                    "<|video_pad|>",
                    "<|vision_start|>",
                    "<|vision_end|>"
                ]
                .iter()
                .any(|t| text.contains(t)),
                "unsupported media control token"
            );
        }
    }
    Ok(())
}
