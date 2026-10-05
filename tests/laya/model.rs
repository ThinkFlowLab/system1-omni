use super::storage_dtype;

#[test]
fn resident_precision_matches_cuda_consumers() {
    for (name, expected) in [
        ("encoder.embeddings.tok_embeddings.weight", "f16"),
        ("encoder.embeddings.norm.weight", "f32"),
        ("encoder.layers.1.attn_norm.weight", "f32"),
        ("encoder.layers.0.attn.Wqkv.weight", "bf16"),
        ("head.layers.0.norm1.bias", "f32"),
        ("head.layers.0.self_attn.in_proj_bias", "f32"),
        ("head.layers.0.self_attn.in_proj_weight", "bf16"),
        ("scorer.0.bias", "f32"),
        ("scorer.1.bias", "bf16"),
        ("act_head.2.weight", "bf16"),
        ("temperature", "f32"),
    ] {
        assert_eq!(storage_dtype(name), expected, "{name}");
    }
    let inventory = crate::weights::checkpoint_tensors();
    assert_eq!(inventory.len(), 206);
    assert_eq!(
        inventory
            .iter()
            .filter(|t| storage_dtype(&t.name) == "f16")
            .count(),
        1
    );
}
