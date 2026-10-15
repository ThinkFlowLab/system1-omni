use super::*;

#[test]
fn vision_cache_limits_from_env() {
    // Each probe gets a private environment; parallel tests never mutate global env.
    if std::env::var_os("CUA_S1_TEST_CACHE_LIMITS_PROBE").is_some() {
        let limits = cache_limits_from_env();
        eprintln!("CACHE_LIMITS {} {}", limits.entries, limits.bytes);
        return;
    }
    let test = format!(
        "{}::vision_cache_limits_from_env",
        module_path!().split_once("::").unwrap().1
    );
    let probe = |entries: Option<&str>,
                 bytes: Option<&str>,
                 expected: (usize, usize),
                 warning: Option<&str>| {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", &test, "--nocapture"])
            .env("CUA_S1_TEST_CACHE_LIMITS_PROBE", "1")
            .env_remove("CUA_S1_VISION_CACHE_ENTRIES")
            .env_remove("CUA_S1_VISION_CACHE_BYTES");
        if let Some(value) = entries {
            command.env("CUA_S1_VISION_CACHE_ENTRIES", value);
        }
        if let Some(value) = bytes {
            command.env("CUA_S1_VISION_CACHE_BYTES", value);
        }
        let output = command.output().unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{stderr}");
        assert!(
            stderr.contains(&format!("CACHE_LIMITS {} {}", expected.0, expected.1)),
            "{stderr}"
        );
        assert_eq!(
            stderr.matches("Invalid CUA_S1_VISION_CACHE_").count(),
            usize::from(warning.is_some()),
            "{stderr}"
        );
        if let Some(name) = warning {
            let default = if name.ends_with("ENTRIES") {
                1
            } else {
                256 << 20
            };
            assert!(stderr.contains(&format!("Invalid {name}=")), "{stderr}");
            assert!(
                stderr.contains(&format!("using default {default}")),
                "{stderr}"
            );
        }
    };
    probe(None, None, (1, 256 << 20), None);
    probe(Some("0"), Some("0"), (0, 0), None);
    probe(Some("4"), Some("1024"), (4, 1024), None);
    probe(Some("17"), None, (16, 256 << 20), None);
    for invalid in ["4 ", "-1", "four"] {
        probe(
            Some(invalid),
            None,
            (1, 256 << 20),
            Some("CUA_S1_VISION_CACHE_ENTRIES"),
        );
        probe(
            None,
            Some(invalid),
            (1, 256 << 20),
            Some("CUA_S1_VISION_CACHE_BYTES"),
        );
    }
}
