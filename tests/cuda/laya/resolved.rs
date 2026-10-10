use super::fixture::Fixture;
use omni_cuda::Cuda;

#[test]
fn resolved_handle_retains_context_after_owner_drop() {
    let fixture = Fixture::new(&[]);
    let path = fixture.path();
    let observer = unsafe { libloading::Library::new(path) }.unwrap();
    let frees = unsafe {
        *observer
            .get::<unsafe extern "C" fn() -> i32>(b"test_stream_frees\0")
            .unwrap()
    };
    let cuda = unsafe { Cuda::load(path) }.unwrap();
    let buffer = cuda.upload(&[0; 64]).unwrap();
    let kernel = cuda.resolve("fill").unwrap();
    assert!(cuda.resolve("missing").is_err());
    drop(cuda);
    unsafe { kernel.launch(&[buffer.ptr()], 1, 16) }.unwrap();
    assert_eq!(buffer.read(64).unwrap(), [73; 64]);
    assert!(unsafe { kernel.launch(&[buffer.ptr()], 0, 16) }.is_err());
    drop(buffer);
    assert_eq!(
        unsafe { frees() },
        0,
        "Kernel is the sole remaining Context owner"
    );
    drop(kernel);
    assert_eq!(unsafe { frees() }, 1);
}
