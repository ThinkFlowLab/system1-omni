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

pub fn prefix_values(
    prefix: Option<&str>,
    fixed: Option<&str>,
    graph: bool,
) -> Result<crate::prefix::PrefixMode> {
    use crate::prefix::PrefixMode;
    let switch = |name, value| match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("{name} must be 0 or 1"),
    };
    let prefix = match prefix {
        None | Some("0") => PrefixMode::Off,
        Some("1") => PrefixMode::Shared,
        Some("auto") => PrefixMode::Auto,
        Some(_) => bail!("DECIDER_PREFIX must be 0, 1 or auto"),
    };
    let fixed = switch("DECIDER_FIXED", fixed)?;
    anyhow::ensure!(
        !(prefix != PrefixMode::Off && fixed),
        "select only one of DECIDER_PREFIX and DECIDER_FIXED"
    );
    anyhow::ensure!(
        !(graph && (prefix != PrefixMode::Off || fixed)),
        "shared/fixed execution requires DECIDER_GRAPH=0"
    );
    Ok(if prefix != PrefixMode::Off {
        prefix
    } else if fixed {
        PrefixMode::Fixed
    } else {
        PrefixMode::Off
    })
}
pub fn prefix_env(graph: bool) -> Result<crate::prefix::PrefixMode> {
    let get = |name| match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error),
    };
    let prefix = get("DECIDER_PREFIX")?;
    let fixed = get("DECIDER_FIXED")?;
    prefix_values(prefix.as_deref(), fixed.as_deref(), graph)
}
