use super::fixture::Fixture;
use omni_cuda::Cuda;

#[test]
fn grouped_upload_prevalidates_and_drains_copy_failures() {
    let fixture = Fixture::new(&[]);
    let path = fixture.path();
    let cuda = unsafe { Cuda::load(path) }.unwrap();
    let other = unsafe { Cuda::load(path) }.unwrap();
    let a = cuda.alloc(64).unwrap();
    let b = cuda.alloc(64).unwrap();
    let foreign = other.alloc(64).unwrap();
    let syncs = unsafe {
        cuda.symbol::<unsafe extern "C" fn() -> i32>(b"test_syncs\0")
            .unwrap()
    };
    let before = unsafe { syncs() };
    cuda.write_many(&[(&a, &[1; 64]), (&b, &[2; 64])]).unwrap();
    assert_eq!(
        unsafe { syncs() },
        before + 1,
        "one drain for the complete group"
    );
    assert_eq!(a.read(64).unwrap(), [1; 64]);
    assert_eq!(b.read(64).unwrap(), [2; 64]);
    assert!(
        cuda.write_many(&[(&a, &[3; 64]), (&foreign, &[4; 64])])
            .is_err()
    );
    assert_eq!(a.read(64).unwrap(), [1; 64]);
    assert!(
        cuda.write_many(&[(&a, &[3; 64][..]), (&b, &[4; 65][..])])
            .is_err()
    );
    assert_eq!(a.read(64).unwrap(), [1; 64]);
    let mode = unsafe {
        cuda.symbol::<unsafe extern "C" fn(i32)>(b"test_copy_mode\0")
            .unwrap()
    };
    unsafe { mode(1) };
    let before = unsafe { syncs() };
    let error = cuda
        .write_many(&[(&a, &[5; 64]), (&b, &[6; 64])])
        .unwrap_err();
    assert!(error.to_string().contains("CUDA"));
    assert_eq!(unsafe { syncs() }, before + 1);
    assert_eq!(
        b.read(64).unwrap(),
        [2; 64],
        "stop submission after the failed member"
    );
    unsafe { mode(3) };
    let error = cuda.write_many(&[(&a, &[5; 64])]).unwrap_err();
    assert!(error.to_string().contains("also failed"));
    unsafe { mode(0) };
    let before = unsafe { syncs() };
    cuda.write_many(&[]).unwrap();
    cuda.write_many(&[(&a, &[])]).unwrap();
    assert_eq!(unsafe { syncs() }, before);
}
