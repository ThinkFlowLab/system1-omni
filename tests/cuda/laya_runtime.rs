//! CPU tests of production Rust bindings against a deterministic native ABI.
#![cfg(unix)]
#[path = "laya/capture.rs"]
mod capture;
#[path = "laya/fixture.rs"]
mod fixture;
#[path = "laya/grouped.rs"]
mod grouped;
#[path = "laya/resolved.rs"]
mod resolved;
