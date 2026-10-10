//! Request-local exact token prefix planning; no state is cached across requests.
use crate::RowInput;
use omni_qwen3_5_native::model::{PromptGroup, SharedPrompts};
pub struct PrefixPlan<'a> {
    pub prompts: SharedPrompts<'a>,
    pub saved_tokens: usize,
}
fn common(rows: &[RowInput], skip: usize) -> usize {
    let first = &rows[0].ids;
    let limit = rows
        .iter()
        .map(|row| row.ids.len().saturating_sub(1))
        .min()
        .unwrap();
    (skip..limit)
        .take_while(|&i| rows.iter().all(|row| row.ids[i] == first[i]))
        .count()
}
impl<'a> PrefixPlan<'a> {
    /// Conservative opt-in policy; savings are aligned token work, not a speed guarantee.
    pub fn worth_auto(&self, rows: &[RowInput]) -> bool {
        let total = rows.iter().map(|row| row.ids.len()).sum::<usize>();
        self.saved_tokens >= 4096 && self.saved_tokens >= total.div_ceil(3)
    }

    pub fn new(rows: &'a [RowInput]) -> Option<Self> {
        if rows.len() < 2 || rows.iter().any(|row| row.ids.is_empty()) {
            return None;
        }
        let request = common(rows, 0);
        let p = request / 64 * 64;
        let mut saved_tokens = (rows.len() - 1) * p;
        let mut groups = Vec::new();
        let mut start = 0;
        while start < rows.len() {
            let mut end = start + 1;
            while end < rows.len() && rows[end].question_id == rows[start].question_id {
                end += 1;
            }
            let group = &rows[start..end];
            let prefix = if group.len() > 1 {
                common(group, request)
            } else {
                0
            };
            let q = (request + prefix) / 64 * 64;
            saved_tokens += (group.len() - 1) * (q - p);
            groups.push(PromptGroup {
                prefix: &group[0].ids[request..request + prefix],
                branches: group
                    .iter()
                    .map(|row| &row.ids[request + prefix..])
                    .collect(),
            });
            start = end;
        }
        (saved_tokens > 0).then(|| Self {
            prompts: SharedPrompts {
                prefix: &rows[0].ids[..request],
                groups,
            },
            saved_tokens,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct PrefixStats {
    pub shared_requests: u64,
    pub auto_independent_requests: u64,
    pub saved_tokens: u64,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefixMode {
    #[default]
    Off,
    Fixed,
    Shared,
    Auto,
}
impl PrefixMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fixed => "fixed",
            Self::Shared => "shared",
            Self::Auto => "auto",
        }
    }
}
