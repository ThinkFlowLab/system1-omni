use super::fixture::Fixture;
use omni_cuda::Cuda;

#[test]
fn capture_unwind_recovery_and_prohibited_operations() {
    let fixture = Fixture::new(&[]);
    let path = fixture.path();
    let cuda = unsafe { Cuda::load(path) }.unwrap();
    let buffer = cuda.alloc(64).unwrap();
    let kernel = cuda.resolve("fill").unwrap();
    let allocations = unsafe {
        cuda.symbol::<unsafe extern "C" fn() -> i32>(b"test_allocations\0")
            .unwrap()
    };
    let nested = unsafe {
        cuda.capture(|| {
            let before = allocations();
            assert!(cuda.alloc(64).is_err());
            assert_eq!(allocations(), before);
            assert!(cuda.sync().is_err());
            assert!(buffer.write(&[0; 64]).is_err());
            assert!(buffer.read(64).is_err());
            assert!(cuda.capture(|| Ok(())).is_err());
            kernel.launch(&[buffer.ptr()], 1, 16)
        })
    }
    .unwrap();
    nested.replay().unwrap();
    cuda.sync().unwrap();
    assert_eq!(buffer.read(64).unwrap(), [73; 64]);
    let error = unsafe { cuda.capture(|| Err(anyhow::anyhow!("injected closure error"))) };
    assert!(error.is_err());
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        cuda.capture(|| panic!("injected panic"))
    }));
    assert!(panic.is_err());
    let recovered = unsafe { cuda.capture(|| kernel.launch(&[buffer.ptr()], 1, 16)) }.unwrap();
    recovered.replay().unwrap();
    cuda.sync().unwrap();
    let mode = unsafe {
        cuda.symbol::<unsafe extern "C" fn(i32)>(b"test_graph_mode\0")
            .unwrap()
    };
    let freed = unsafe {
        cuda.symbol::<unsafe extern "C" fn() -> i32>(b"test_freed\0")
            .unwrap()
    };
    let before = unsafe { freed() };
    unsafe { mode(1) };
    assert!(unsafe { cuda.capture(|| Ok(())) }.is_err());
    assert_eq!(unsafe { freed() }, before + 1);
    unsafe { mode(2) };
    assert!(unsafe { cuda.capture(|| Ok(())) }.is_err());
    unsafe { mode(0) };
    drop(cuda);
    recovered.replay().unwrap();
    drop(recovered);
    drop(nested);
    assert_eq!(buffer.read(64).unwrap(), [73; 64]);
}

#[test]
fn eager_library_without_graph_cleanup_symbols_stays_usable() {
    let fixture = Fixture::new(&["OMIT_GRAPH_FREE"]);
    let path = fixture.path();
    let cuda = unsafe { Cuda::load(path) }.unwrap();
    let begins = unsafe {
        cuda.symbol::<unsafe extern "C" fn() -> i32>(b"test_begins\0")
            .unwrap()
    };
    let before = unsafe { begins() };
    assert!(unsafe { cuda.capture(|| Ok(())) }.is_err());
    assert_eq!(unsafe { begins() }, before);
    let buffer = cuda.upload(&[9; 64]).unwrap();
    assert_eq!(buffer.read(64).unwrap(), [9; 64]);
}

#[test]
fn replay_is_rejected_inside_capture_before_entering_driver() {
    let fixture = Fixture::new(&[]);
    let path = fixture.path();
    let cuda = unsafe { Cuda::load(path) }.unwrap();
    let buffer = cuda.alloc(64).unwrap();
    let kernel = cuda.resolve("fill").unwrap();
    let first = unsafe { cuda.capture(|| kernel.launch(&[buffer.ptr()], 1, 16)) }.unwrap();
    let second = unsafe {
        cuda.capture(|| {
            assert!(first.replay().is_err());
            kernel.launch(&[buffer.ptr()], 1, 16)
        })
    }
    .unwrap();
    first.replay().unwrap();
    cuda.sync().unwrap();
    drop(first);
    second.replay().unwrap();
    cuda.sync().unwrap();
    assert_eq!(buffer.read(64).unwrap(), [73; 64]);
}
