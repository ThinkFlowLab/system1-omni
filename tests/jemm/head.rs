use omni_jemm_native::executor::head_bytes;
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};
#[test]
fn pads_selected_bf16_lm_head_rows_to_256_without_substitution() {
    let raw = (0..64)
        .flat_map(|i| half::bf16::from_f32(i as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let tensor = TensorView::new(Dtype::BF16, vec![32, 2], &raw).unwrap();
    let data = serialize([("weight", tensor)], None).unwrap();
    let padded = head_bytes(&data, 2).unwrap();
    assert_eq!(padded.len(), 256 * 2 * 2);
    assert_eq!(&padded[..raw.len()], raw.as_slice());
    assert!(padded[raw.len()..].iter().all(|&x| x == 0));
    let tensor = TensorView::new(Dtype::BF16, vec![64], &raw).unwrap();
    assert!(head_bytes(&serialize([("weight", tensor)], None).unwrap(), 2).is_err());
}
