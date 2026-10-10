//! Tests for the embeddings client, wired in through `#[path]` from
//! `src/models/clm/src/embedding.rs`.
//!
//! `HttpEncoder::body` is private, so CONTRIBUTING's rule for private items
//! applies: the `#[cfg(test)]` and the module path stay in `src/`, the test body
//! lives here.
use super::*;

/// `embedder.py` sends its `max_tokens` as `truncate_prompt_tokens`; a request without
/// it is truncated differently on the server, or rejected instead of truncated.
#[test]
fn the_request_carries_the_truncation_limit() {
    let encoder =
        HttpEncoder::new("http://127.0.0.1:1/v1/embeddings", "m", Duration::ZERO).unwrap();
    let texts = vec!["a".to_string()];
    assert_eq!(encoder.body(&texts)["truncate_prompt_tokens"], 2048);

    let unlimited = encoder.with_max_tokens(None);
    assert!(
        unlimited
            .body(&texts)
            .get("truncate_prompt_tokens")
            .is_none()
    );

    let custom = HttpEncoder::new("http://127.0.0.1:1/v1/embeddings", "m", Duration::ZERO)
        .unwrap()
        .with_max_tokens(Some(64));
    assert_eq!(custom.body(&texts)["truncate_prompt_tokens"], 64);
    assert_eq!(custom.body(&texts)["encoding_format"], "base64");
}
