//! Opt-in request and processed-token budgets for one loaded Decider worker.
use anyhow::{Context, Result, ensure};
use omni_runtime::{admission::BoundedAdmission, control::ExecutionContext, engine::Report};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Default)]
/// All limits are opt-in. Unset environment variables retain unbounded serving.
/// `DECIDER_MAX_PENDING_REQUESTS` includes upload, preparation, queueing and execution.
/// `DECIDER_MAX_PENDING_TOKENS` counts complete prepared rows, not billable usage.
/// `DECIDER_REQUEST_TIMEOUT_MS` covers upload through response reconstruction.
pub struct RuntimeConfig {
    pub max_requests: Option<usize>,
    pub max_tokens: Option<usize>,
    pub timeout: Option<Duration>,
}
impl RuntimeConfig {
    pub fn from_env() -> Result<Self> {
        let read = |name: &str| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(anyhow::anyhow!("{name}: {error}")),
        };
        Self::from_values(
            read("DECIDER_MAX_PENDING_REQUESTS")?.as_deref(),
            read("DECIDER_MAX_PENDING_TOKENS")?.as_deref(),
            read("DECIDER_REQUEST_TIMEOUT_MS")?.as_deref(),
        )
    }
    pub fn from_values(
        requests: Option<&str>,
        tokens: Option<&str>,
        timeout: Option<&str>,
    ) -> Result<Self> {
        let parse = |name: &str, value: Option<&str>| -> Result<Option<usize>> {
            value
                .map(|value| {
                    let value: usize = value.parse().with_context(|| format!("invalid {name}"))?;
                    ensure!(value > 0, "{name} must be positive");
                    Ok(value)
                })
                .transpose()
        };
        let config = Self {
            max_requests: parse("DECIDER_MAX_PENDING_REQUESTS", requests)?,
            max_tokens: parse("DECIDER_MAX_PENDING_TOKENS", tokens)?,
            timeout: parse("DECIDER_REQUEST_TIMEOUT_MS", timeout)?
                .map(|n| Duration::from_millis(n as u64)),
        };
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.max_requests != Some(0) && self.max_tokens != Some(0),
            "runtime budgets must be positive"
        );
        if let Some(timeout) = self.timeout {
            ensure!(
                !timeout.is_zero() && Instant::now().checked_add(timeout).is_some(),
                "request timeout must be positive and fit the monotonic clock"
            );
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Overloaded;
impl std::fmt::Display for Overloaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("inference capacity exhausted")
    }
}
impl std::error::Error for Overloaded {}

pub struct Runtime {
    config: RuntimeConfig,
    requests: Option<BoundedAdmission>,
    tokens: Option<BoundedAdmission>,
}
impl Runtime {
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            requests: config.max_requests.map(|n| BoundedAdmission::new(n, None)),
            tokens: config
                .max_tokens
                .map(|n| BoundedAdmission::new(usize::MAX, Some(n))),
        })
    }
    pub fn config(&self) -> RuntimeConfig {
        self.config
    }
    /// Reserve before upload and preparation. The context follows all blocking work.
    pub fn admit(&self, started: Instant) -> Result<ExecutionContext> {
        let context = ExecutionContext::new(self.config.timeout.map(|timeout| started + timeout));
        match &self.requests {
            Some(admission) => Ok(context.retain(admission.try_admit(1).ok_or(Overloaded)?)),
            None => Ok(context),
        }
    }
    /// Cost is the sum of complete rows, including repeated state and Score expansion.
    pub fn reserve_tokens(
        &self,
        context: ExecutionContext,
        tokens: usize,
    ) -> Result<ExecutionContext> {
        context.check()?;
        match &self.tokens {
            Some(admission) => Ok(context.retain(admission.try_admit(tokens).ok_or(Overloaded)?)),
            None => Ok(context),
        }
    }
    /// Request-depth metrics exist only when a request-count bound is configured.
    /// Token-only limits have no finite request capacity to report.
    pub fn report(&self) -> Option<Report> {
        self.requests.as_ref().map(|admission| {
            let mut report = admission.report();
            if let Some(tokens) = &self.tokens {
                report.rejected = report.rejected.saturating_add(tokens.report().rejected);
            }
            report
        })
    }
}
