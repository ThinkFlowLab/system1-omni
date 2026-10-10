//! CPU tests of production Rust bindings against a deterministic native ABI.
#![cfg(unix)]
#[path = "laya/fixture.rs"]
mod fixture;
#[path = "laya/resolved.rs"]
mod resolved;
