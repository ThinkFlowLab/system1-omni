//! Build a distinct library per test so its global counters never leak between tests.
use std::{
    path::{Path, PathBuf},
    process::Command,
};
use tempfile::{TempDir, tempdir};

pub struct Fixture {
    _directory: TempDir,
    path: PathBuf,
}
impl Fixture {
    pub fn new(defines: &[&str]) -> Self {
        let directory = tempdir().expect("fixture directory");
        let path = directory.path().join(if cfg!(target_os = "macos") {
            "liblaya_fixture.dylib"
        } else {
            "liblaya_fixture.so"
        });
        let source = directory.path().join("fixture.c");
        std::fs::write(&source, include_str!("fixture.c")).expect("write fixture source");
        let mut compiler = Command::new("cc");
        compiler.arg("-std=c11");
        if cfg!(target_os = "macos") {
            compiler.arg("-dynamiclib");
        } else {
            compiler.args(["-shared", "-fPIC"]);
        }
        for define in defines {
            compiler.arg(format!("-D{define}"));
        }
        let output = compiler
            .arg(&source)
            .arg("-o")
            .arg(&path)
            .output()
            .expect("a C compiler is required for the CPU ABI fixture");
        assert!(
            output.status.success(),
            "fixture compilation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Self {
            _directory: directory,
            path,
        }
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
}
