//! Pins for the token expansion + 3-axis positions at the image boundary,
//! hand-computed from HF modeling_qwen3_5 get_rope_index semantics
//! (meshgrid t-outer/h-mid/w-fastest + start; advance = max(grid_h, grid_w)/merge).

use std::sync::Arc;

use omni_jev_vl_native::images::{ImageAsset, expand, meshgrid_positions};

const PAD: u32 = 248056;

fn asset(grid: [i64; 3]) -> Arc<ImageAsset> {
    let [t, h, w] = grid;
    let n = (t * h * w / 4) as usize;
    Arc::new(ImageAsset {
        grid_thw: grid,
        embeddings: vec![half::bf16::ONE; n * 5120],
    })
}

#[test]
fn single_image_positions_follow_hf() {
    // text 2 tokens, image grid [1,4,6] -> n=6 (lg_h=2, lg_w=3), text 1 token.
    // advance = max(4, 6) / 2 = 3 -> last text at position 2 + 3 = 5.
    let text_ids: Vec<u32> = vec![300, 301, PAD, 302];
    let e = expand(&text_ids, PAD, &[asset([1, 4, 6])]).unwrap();
    assert_eq!(e.ids, [300, 301, PAD, PAD, PAD, PAD, PAD, PAD, 302]);
    assert_eq!(e.positions[0], vec![0, 1, 2, 2, 2, 2, 2, 2, 5]);
    assert_eq!(e.positions[1], vec![0, 1, 2, 2, 2, 3, 3, 3, 5]);
    assert_eq!(e.positions[2], vec![0, 1, 2, 3, 4, 2, 3, 4, 5]);
    // R2d cache-anchor geometry.
    assert_eq!(
        e.blocks
            .iter()
            .map(|b| (b.start, b.end, b.base, b.advance, b.asset))
            .collect::<Vec<_>>(),
        vec![(2, 8, 2, 3, 0)]
    );
}

#[test]
fn two_images_continue_after_max_hw() {
    // hand-computed: see module doc.
    let text_ids: Vec<u32> = vec![300, PAD, 301, 302, PAD, 303];
    let e = expand(&text_ids, PAD, &[asset([1, 2, 4]), asset([1, 4, 2])]).unwrap();
    assert_eq!(e.ids, [300, PAD, PAD, 301, 302, PAD, PAD, 303]);
    assert_eq!(e.positions[0], vec![0, 1, 1, 3, 4, 5, 5, 7]);
    assert_eq!(e.positions[1], vec![0, 1, 1, 3, 4, 5, 6, 7]);
    assert_eq!(e.positions[2], vec![0, 1, 2, 3, 4, 5, 5, 7]);
    assert_eq!(
        e.blocks
            .iter()
            .map(|b| (b.start, b.end, b.base, b.advance, b.asset))
            .collect::<Vec<_>>(),
        vec![(1, 3, 1, 2, 0), (5, 7, 5, 2, 1)]
    );
}

#[test]
fn meshgrid_helper_matches_expand_rows() {
    // The helper must produce exactly expand's rows for every suffix of one block.
    let text_ids: Vec<u32> = vec![300, 301, PAD, 302];
    let e = expand(&text_ids, PAD, &[asset([1, 4, 6])]).unwrap();
    let b = &e.blocks[0];
    for from in 0..6usize {
        let got = meshgrid_positions([1, 4, 6], b.base, from, 6 - from);
        for (want, have) in e.positions.iter().zip(&got) {
            assert_eq!(&want[b.start + from..b.end], &have[..], "slice from {from}");
        }
    }
}

#[test]
fn placeholder_count_must_match_images() {
    let e = expand(&[PAD], PAD, &[]);
    assert!(e.is_err());
}

#[test]
fn image_assets_reject_wrong_sources_and_invalid_grids() {
    let dir = tempfile::tempdir().unwrap();
    for (grid, url, model) in [
        ([1, 2, 2], "other-url", "model"),
        ([1, 2, 2], "url", "other-model"),
        ([1, 3, 2], "url", "model"),
        ([2, 2, 2], "url", "model"),
        ([1, i64::MAX - 1, 4], "url", "model"),
    ] {
        std::fs::write(
            dir.path().join("grid.json"),
            serde_json::to_vec(&serde_json::json!({
                "grid_thw": grid, "n_tokens": 1,
                "url_sha256": url, "model_index_sha256": model,
            }))
            .unwrap(),
        )
        .unwrap();
        let error = ImageAsset::load(dir.path(), "url", "model").err().unwrap();
        // Reject metadata before opening or allocating the embedding tensor.
        assert!(!format!("{error:#}").contains("imgcache asset emb"));
    }
}
