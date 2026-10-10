use super::*;
use safetensors::{Dtype, tensor::TensorView};
use std::collections::HashMap;

fn tensor(dtype: Dtype, shape: Vec<usize>, bytes: &[u8]) -> Vec<u8> {
    let view = TensorView::new(dtype, shape, bytes).unwrap();
    safetensors::serialize(
        HashMap::from([("model.language_model.embed_tokens.weight", view)]),
        None,
    )
    .unwrap()
}

#[test]
fn selects_requested_tied_rows_and_zero_pads_the_head() {
    let bytes: Vec<u8> = (0..12)
        .flat_map(|i| half::bf16::from_f32(i as f32).to_le_bytes())
        .collect();
    let encoded = tensor(Dtype::BF16, vec![3, 4], &bytes);
    let st = SafeTensors::deserialize(&encoded).unwrap();
    let selected = selected_rows(&st, &[2, 0], 4, 3).unwrap();
    assert_eq!(&selected[..8], &bytes[16..24]);
    assert_eq!(&selected[8..16], &bytes[..8]);
    assert_eq!(selected.len(), 8 * 4 * 2);
    assert!(selected[16..].iter().all(|&b| b == 0));
}

#[test]
fn rejects_invalid_head_layouts_ids_and_nonfinite_weights() {
    let bytes = vec![0; 24];
    for (dtype, shape, data) in [
        (Dtype::BF16, vec![3, 4], bytes.clone()),
        (Dtype::F32, vec![3, 2], bytes.clone()),
        (Dtype::BF16, vec![2, 6], bytes.clone()),
    ] {
        let encoded = tensor(dtype, shape, &data);
        let st = SafeTensors::deserialize(&encoded).unwrap();
        assert!(selected_rows(&st, &[3], 4, 3).is_err());
    }
    let mut bytes = bytes;
    bytes[..2].copy_from_slice(&half::bf16::NAN.to_le_bytes());
    let encoded = tensor(Dtype::BF16, vec![3, 4], &bytes);
    let st = SafeTensors::deserialize(&encoded).unwrap();
    assert!(selected_rows(&st, &[0], 4, 3).is_err());
    assert!(selected_rows(&st, &[], 4, 3).is_err());
    assert!(selected_rows(&st, &[1, 1], 4, 3).is_err());
}

#[test]
fn checks_actual_file_content_and_length() {
    let path = std::env::temp_dir().join(format!("decider-hash-{}", std::process::id()));
    std::fs::write(&path, b"abc").unwrap();
    let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    verify_file(&path, 3, digest).unwrap();
    assert!(verify_file(&path, 4, digest).is_err());
    assert!(verify_file(&path, 3, &"0".repeat(64)).is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn refuses_an_index_that_would_override_the_verified_single_file() {
    let dir = std::env::temp_dir().join(format!("decider-index-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.safetensors.index.json"), b"{}").unwrap();
    let labels: Vec<u32> = (0..255).collect();
    let error = Checkpoint::load(&dir, &labels).err().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(error.to_string().contains("sharded"), "{error}");
}
