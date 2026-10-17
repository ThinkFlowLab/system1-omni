use omni_qwen3_5_native::cuda::DeviceBuffer;

#[test]
fn zero_byte_buffer_accepts_zero_offset_without_cuda() {
    let buffer = DeviceBuffer::new(0).unwrap();
    assert!(buffer.at(0).is_null());
}

#[test]
#[should_panic(expected = "offset <= self.bytes")]
fn zero_byte_buffer_rejects_out_of_bounds_offset_without_cuda() {
    let buffer = DeviceBuffer::new(0).unwrap();
    buffer.at(1);
}
