//! Token preparation: each question's user content in the checkpoint's chat template with thinking off, as
//! vLLM renders it for the helper, plus the candidate letter token ids in option order.

use std::path::Path;

use anyhow::{Result, ensure};
use tokenizers::Tokenizer;

use crate::contract::{self, MAX_OPTIONS, MAX_PROMPT_TOKENS, Question};

/// `chat_template.jinja` for one user message, `add_generation_prompt=True`, `enable_thinking=False`.
pub const CHAT_PREFIX: &str = "<|im_start|>user\n";
pub const CHAT_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub struct Processor {
    tokenizer: Tokenizer,
    /// The token id of each letter `A`..`Z`, `a`..`z`, each a single token.
    letters: Vec<u32>,
    max_tokens: usize,
}

/// One question ready for a single prefill: its prompt ids and the ids whose scores are read at the next position.
#[derive(Debug)]
pub struct Prepared {
    pub question: Question,
    pub input_ids: Vec<u32>,
    pub candidate_ids: Vec<u32>,
}

impl Processor {
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let letters = (0..MAX_OPTIONS)
            .map(|i| {
                let ids = tokenizer
                    .encode(contract::letter(i).to_string(), false)
                    .map_err(anyhow::Error::msg)?
                    .get_ids()
                    .to_vec();
                ensure!(
                    ids.len() == 1,
                    "letter {} is not a single token",
                    contract::letter(i)
                );
                Ok(ids[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let mut unique = letters.clone();
        unique.sort_unstable();
        unique.dedup();
        ensure!(
            unique.len() == letters.len(),
            "candidate token ids are not unique"
        );
        Ok(Self {
            tokenizer,
            letters,
            max_tokens: MAX_PROMPT_TOKENS,
        })
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Every question's ids, after validating every prompt length; never truncates.
    pub fn prepare(&self, raw: &[u8]) -> Result<Vec<Prepared>> {
        contract::compile(raw)?
            .into_iter()
            .map(|question| {
                let text = format!("{CHAT_PREFIX}{}{CHAT_SUFFIX}", question.content);
                let input_ids = self
                    .tokenizer
                    .encode(text, false)
                    .map_err(anyhow::Error::msg)?
                    .get_ids()
                    .to_vec();
                ensure!(
                    input_ids.len() <= self.max_tokens,
                    "questions.{} prompt is {} tokens; the limit is {}",
                    question.id,
                    input_ids.len(),
                    self.max_tokens
                );
                let candidate_ids = self.letters[..question.keys.len()].to_vec();
                Ok(Prepared {
                    question,
                    input_ids,
                    candidate_ids,
                })
            })
            .collect()
    }
}
