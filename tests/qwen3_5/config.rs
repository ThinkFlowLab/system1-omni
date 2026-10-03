use omni_qwen3_5_native::model::Config;
use std::path::Path;

#[test]
fn open_jev_27b_backbone_has_supported_dimensions() {
    // Qwen/Qwen3.8-27B @ 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
    let cfg = Config::load(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/qwen3_5/data"
    )))
    .unwrap();
    assert_eq!((cfg.hidden, cfg.intermediate), (5120, 17408));
    assert_eq!((cfg.heads, cfg.kv_heads, cfg.head_dim), (24, 4, 256));
    assert_eq!((cfg.lin_k_heads, cfg.lin_v_heads), (16, 48));
    assert_eq!(cfg.full_attention.len(), 64);
    assert_eq!(cfg.full_attention.iter().filter(|&&x| x).count(), 16);
}
