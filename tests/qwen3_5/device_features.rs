use omni_qwen3_5_native::cuda::DeviceFeatureLayout;

#[test]
fn device_layout_requires_exact_nonempty_bf16_matrix_and_same_device() {
    assert!(
        DeviceFeatureLayout::new(3, 2560, 15360, 0)
            .unwrap()
            .validate(3, 2560, 0)
            .is_ok()
    );
    for (rows, columns, bytes, device) in [
        (0, 2560, 0, 0),
        (3, 0, 0, 0),
        (3, 2560, 15359, 0),
        (3, 2560, 15362, 0),
        (3, 2560, 15360, -1),
        (usize::MAX, 2, 0, 0),
        (usize::MAX / 2 + 1, 1, 0, 0),
    ] {
        assert!(DeviceFeatureLayout::new(rows, columns, bytes, device).is_err());
    }
    let layout = DeviceFeatureLayout::new(3, 2560, 15360, 1).unwrap();
    assert!(layout.validate(3, 2560, 0).is_err());
    assert!(layout.validate(2, 2560, 1).is_err());
    assert!(layout.validate(3, 1024, 1).is_err());
}
