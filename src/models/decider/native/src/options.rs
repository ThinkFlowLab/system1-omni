//! Model-owned execution switches, independent from other workers' environment.
use anyhow::{Result, bail};
pub fn graph_value(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("DECIDER_GRAPH must be 0 or 1"),
    }
}
pub fn graph_env() -> Result<bool> {
    let value = match std::env::var("DECIDER_GRAPH") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    graph_value(value.as_deref())
}
