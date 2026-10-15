use super::LabelHead;
use half::bf16;
use omni_qwen3_5_native::cuda;
#[test]
#[ignore = "requires JEMM_CUDA_LIB and an available CUDA device"]
fn bf16_label_head_gemm_matches_exact_constant_rows() {
    let library = std::env::var_os("JEMM_CUDA_LIB").expect("set JEMM_CUDA_LIB");
    cuda::load(std::path::Path::new(&library)).unwrap();
    cuda::set_device(0).unwrap();
    let hidden = 5120;
    let mut bytes = vec![0u8; 256 * hidden * 2];
    for row in 0..32 {
        for col in 0..hidden {
            let offset = (row * hidden + col) * 2;
            bytes[offset..offset + 2]
                .copy_from_slice(&bf16::from_f32(row as f32 / 32.).to_le_bytes());
        }
    }
    let mut head = LabelHead::load(&bytes, hidden).unwrap();
    let actual = head.score(&vec![1.; hidden], 32).unwrap();
    for (row, score) in actual.iter().enumerate() {
        assert_eq!(
            *score,
            bf16::from_f32(row as f32 * 160.).to_f32(),
            "label row {row}"
        );
    }
}
