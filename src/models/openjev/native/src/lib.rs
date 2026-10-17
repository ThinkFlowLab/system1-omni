//! openjev/openjev's letter-readout contract: request compilation into one prompt per question, the candidate
//! letter token ids, and the calibrated typed answers computed from the candidates' scores. CPU only; the
//! executor (the selected `lm_head` rows on the shared Qwen backbone) is a separate step.
pub mod contract;
pub mod processing;
pub mod pyrepr;
