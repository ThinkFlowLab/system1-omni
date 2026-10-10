//! Bounded packing of independent complete rows within a single admitted request.
use anyhow::{Context, Result, ensure};
use std::ops::Range;

#[derive(Clone, Copy, Debug)]
pub struct BatchLimits {
    max_rows: usize,
    max_tokens: usize,
}
impl Default for BatchLimits {
    fn default() -> Self {
        Self {
            max_rows: 1,
            max_tokens: 4096,
        }
    }
}
impl BatchLimits {
    pub fn new(max_rows: usize, max_tokens: usize) -> Result<Self> {
        ensure!(
            (1..=4).contains(&max_rows),
            "batch rows must be within 1..=4"
        );
        ensure!(
            (1..=4096).contains(&max_tokens),
            "batch tokens must be within 1..=4096"
        );
        Ok(Self {
            max_rows,
            max_tokens,
        })
    }
    pub fn from_values(rows: Option<&str>, tokens: Option<&str>) -> Result<Self> {
        let rows = rows
            .map_or(Ok(1), str::parse)
            .context("DECIDER_BATCH_MAX_ROWS must be an integer")?;
        let tokens = tokens
            .map_or(Ok(4096), str::parse)
            .context("DECIDER_BATCH_MAX_TOKENS must be an integer")?;
        Self::new(rows, tokens)
    }
    pub fn from_env() -> Result<Self> {
        let get = |name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(e),
        };
        let rows = get("DECIDER_BATCH_MAX_ROWS")?;
        let tokens = get("DECIDER_BATCH_MAX_TOKENS")?;
        Self::from_values(rows.as_deref(), tokens.as_deref())
    }
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    /// Retain original row order; a row above the packing token budget runs alone.
    /// Total request/row admission remains the processor/executor's separate limit.
    pub fn ranges(&self, lengths: &[usize]) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        let mut start = 0;
        let mut tokens = 0usize;
        for (index, &length) in lengths.iter().enumerate() {
            if index > start
                && (index - start == self.max_rows
                    || tokens
                        .checked_add(length)
                        .is_none_or(|total| total > self.max_tokens))
            {
                ranges.push(start..index);
                start = index;
                tokens = 0;
            }
            tokens = tokens.saturating_add(length);
        }
        if start < lengths.len() {
            ranges.push(start..lengths.len());
        }
        ranges
    }
}
