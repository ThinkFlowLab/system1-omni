//! Bounded packing of independent candidate prompts within one admitted request.

use std::ops::Range;

const MAX_SEQUENCES: usize = 16;
const MAX_TOKENS: usize = 4096;

/// Keep candidate order, with long prompts executing alone at their original size.
pub(crate) fn ranges(inputs: &[&[u32]]) -> Vec<Range<usize>> {
    let mut batches = Vec::new();
    let mut start = 0;
    let mut tokens = 0;
    for (index, ids) in inputs.iter().enumerate() {
        if index > start && (index - start == MAX_SEQUENCES || tokens + ids.len() > MAX_TOKENS) {
            batches.push(start..index);
            start = index;
            tokens = 0;
        }
        tokens += ids.len();
    }
    if start < inputs.len() {
        batches.push(start..inputs.len());
    }
    batches
}

#[cfg(test)]
#[path = "../../../../../tests/open_jev/batching.rs"]
mod tests;
