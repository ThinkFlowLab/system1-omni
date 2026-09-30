#![cfg(unix)]

use libloading::Library;
use omni_cuda::Cuda;
use std::{ffi::CStr, path::PathBuf, process::Command};
use tempfile::{TempDir, tempdir};

const MODE_NORMAL: i32 = 0;
const MODE_CREATE_ERROR: i32 = 1;
const MODE_COPY_ERROR: i32 = 2;
const MODE_SYNC_ERROR: i32 = 3;
const MODE_COPY_AND_SYNC_ERROR: i32 = 4;
const MODE_ALLOC_ERROR: i32 = 5;

struct Fixture {
    _dir: TempDir,
    path: PathBuf,
    library: Option<Library>,
}
impl Fixture {
    fn new(defines: &[&str]) -> Self {
        let dir = tempdir().unwrap();
        let path = dir
            .path()
            .join(format!("runtime{}", std::env::consts::DLL_SUFFIX));
        let mut command = Command::new("cc");
        command.arg(if cfg!(target_os = "macos") {
            "-dynamiclib"
        } else {
            "-shared"
        });
        let result = command
            .args(["-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror"])
            .args(defines)
            .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/runtime.c"))
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let library = Some(unsafe { Library::new(&path) }.unwrap());
        Self {
            _dir: dir,
            path,
            library,
        }
    }
    fn mode(&self, mode: i32) {
        unsafe {
            self.library
                .as_ref()
                .unwrap()
                .get::<unsafe extern "C" fn(i32)>(b"laya_test_mode\0")
                .unwrap()(mode);
        }
    }
    fn trace(&self) -> String {
        unsafe {
            let f = self
                .library
                .as_ref()
                .unwrap()
                .get::<unsafe extern "C" fn() -> *const std::ffi::c_char>(b"laya_test_trace\0")
                .unwrap();
            CStr::from_ptr(f()).to_string_lossy().into_owned()
        }
    }
    fn live(&self) -> i32 {
        unsafe {
            self.library
                .as_ref()
                .unwrap()
                .get::<unsafe extern "C" fn() -> i32>(b"laya_test_live\0")
                .unwrap()()
        }
    }
    fn load(&self, device: i32) -> anyhow::Result<Cuda> {
        unsafe { Cuda::load(&self.path, device) }
    }
}

#[test]
fn loading_rejects_incompatible_libraries_before_creating_resources() {
    for defines in [vec!["-DLAYA_TEST_ABI=2"], vec!["-DLAYA_TEST_NO_DOWNLOAD"]] {
        let fixture = Fixture::new(&defines);
        assert!(fixture.load(0).is_err());
        assert_eq!(fixture.live(), 0);
        assert!(!fixture.trace().contains("create"));
    }
    let fixture = Fixture::new(&[]);
    assert!(fixture.load(-1).is_err());
    fixture.mode(MODE_CREATE_ERROR);
    let error = fixture.load(0).err().unwrap().to_string();
    assert!(error.contains("23"), "{error}");
    assert_eq!(fixture.live(), 0);
    assert!(!fixture.trace().contains("destroy"));
}

#[test]
fn copies_round_trip_and_bounds_are_checked() {
    let fixture = Fixture::new(&[]);
    let cuda = fixture.load(0).unwrap();
    assert!(cuda.alloc(0).is_err());
    let buffer = cuda.upload(&[1, 2, 3, 4]).unwrap();
    assert_eq!(buffer.bytes(), 4);
    assert_eq!(buffer.read(4).unwrap(), [1, 2, 3, 4]);
    buffer.write(&[9, 8]).unwrap();
    assert_eq!(buffer.read(4).unwrap(), [9, 8, 3, 4]);
    buffer.write(&[]).unwrap();
    assert!(buffer.read(0).unwrap().is_empty());
    let before = fixture.trace();
    assert!(buffer.write(&[0; 5]).is_err());
    assert!(buffer.read(5).is_err());
    assert_eq!(fixture.trace(), before);
    drop(buffer);
    drop(cuda);
    assert_eq!(fixture.live(), 0);
}

#[test]
fn copy_errors_still_synchronize_and_async_errors_are_reported() {
    let fixture = Fixture::new(&[]);
    let cuda = fixture.load(0).unwrap();
    let buffer = cuda.alloc(3).unwrap();
    fixture.mode(MODE_COPY_ERROR);
    let error = buffer.write(&[4, 5, 6]).unwrap_err().to_string();
    assert!(error.contains("fixture-error-41"), "{error}");
    assert!(fixture.trace().contains("upload0 sync0 "));
    fixture.mode(MODE_NORMAL);
    assert_eq!(buffer.read(3).unwrap(), [4, 5, 6]);
    fixture.mode(MODE_COPY_ERROR);
    assert!(buffer.read(3).unwrap_err().to_string().contains("41"));
    assert!(fixture.trace().contains("download0 sync0 "));
    fixture.mode(MODE_SYNC_ERROR);
    assert!(
        buffer
            .write(&[7, 8, 9])
            .unwrap_err()
            .to_string()
            .contains("42")
    );
    assert!(buffer.read(3).unwrap_err().to_string().contains("42"));
    assert!(cuda.sync().unwrap_err().to_string().contains("42"));
    fixture.mode(MODE_COPY_AND_SYNC_ERROR);
    assert!(
        buffer
            .write(&[1, 2, 3])
            .unwrap_err()
            .to_string()
            .contains("41")
    );
    fixture.mode(MODE_NORMAL);
    assert_eq!(buffer.read(3).unwrap(), [1, 2, 3]);
    drop(buffer);
    drop(cuda);
    assert_eq!(fixture.live(), 0);
}

#[test]
fn failed_allocations_and_uploads_release_partial_resources() {
    let fixture = Fixture::new(&[]);
    let cuda = fixture.load(0).unwrap();
    fixture.mode(MODE_ALLOC_ERROR);
    assert!(cuda.alloc(4).err().unwrap().to_string().contains("31"));
    assert_eq!(fixture.live(), 1);
    fixture.mode(MODE_COPY_ERROR);
    assert!(
        cuda.upload(&[1, 2])
            .err()
            .unwrap()
            .to_string()
            .contains("41")
    );
    assert_eq!(fixture.live(), 1);
    fixture.mode(MODE_NORMAL);
    drop(cuda);
    assert_eq!(fixture.live(), 0);
}

#[test]
fn buffers_keep_the_library_and_stream_alive_after_cuda_is_dropped() {
    let mut fixture = Fixture::new(&[]);
    let cuda = fixture.load(0).unwrap();
    let cloned = cuda.clone();
    let buffer = cuda.upload(&[7, 8, 9]).unwrap();
    // Remove the test's dlopen handle too: only the buffer may keep this library alive.
    drop(fixture.library.take());
    drop(cuda);
    drop(cloned);
    assert_eq!(buffer.read(3).unwrap(), [7, 8, 9]);
    fixture.library = Some(unsafe { Library::new(&fixture.path) }.unwrap());
    assert_eq!(fixture.live(), 2);
    assert!(!fixture.trace().contains("destroy"));
    drop(buffer);
    assert_eq!(fixture.live(), 0);
    let trace = fixture.trace();
    assert!(trace.rfind("sync0 ").unwrap() < trace.rfind("destroy0 ").unwrap());
    assert!(trace.rfind("free0 ").unwrap() < trace.rfind("destroy0 ").unwrap());
}

#[test]
fn operations_and_drops_restore_the_owning_device() {
    let fixture = Fixture::new(&[]);
    let cuda0 = fixture.load(0).unwrap();
    let buffer0 = cuda0.upload(&[1, 2]).unwrap();
    let cuda1 = fixture.load(1).unwrap();
    let buffer1 = cuda1.upload(&[3, 4]).unwrap();
    assert_eq!(buffer0.read(2).unwrap(), [1, 2]);
    buffer1.write(&[5, 6]).unwrap();
    cuda0.sync().unwrap();
    drop(buffer1);
    drop(cuda1);
    drop(buffer0);
    drop(cuda0);
    assert_eq!(fixture.live(), 0);
    let trace = fixture.trace();
    for event in [
        "device0 download0",
        "device1 upload1",
        "device0 sync0",
        "device1 sync1 free1",
        "device0 sync0 free0",
    ] {
        assert!(trace.contains(event), "missing {event}: {trace}");
    }
}

#[test]
#[ignore = "requires an approved GPU and LAYA_CUDA_LIBRARY plus LAYA_CUDA_DEVICE"]
fn real_gpu_round_trip() {
    let path = PathBuf::from(std::env::var_os("LAYA_CUDA_LIBRARY").expect("set LAYA_CUDA_LIBRARY"));
    let device = std::env::var("LAYA_CUDA_DEVICE")
        .expect("set LAYA_CUDA_DEVICE")
        .parse()
        .unwrap();
    {
        let library = unsafe { Library::new(&path) }.unwrap();
        let free = unsafe {
            library.get::<unsafe extern "C" fn(*mut std::ffi::c_void) -> i32>(b"laya_free\0")
        }
        .unwrap();
        assert_eq!(unsafe { free(std::ptr::null_mut()) }, 1000);
    }
    let cuda = unsafe { Cuda::load(&path, device) }.unwrap();
    let expected: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
    let buffer = cuda.upload(&expected).unwrap();
    assert_eq!(buffer.read(expected.len()).unwrap(), expected);
    drop(cuda);
    assert_eq!(buffer.read(expected.len()).unwrap(), expected);
}

#[test]
fn kernels_validate_context_and_shape_and_keep_the_library_alive() {
    use omni_cuda::kernels::Kernels;
    let f = Fixture::new(&["-DLAYA_TEST_KERNELS"]);
    let cuda = unsafe { Cuda::load(&f.path, 0) }.unwrap();
    let buffer = cuda.alloc(1).unwrap();
    let kernels = unsafe { Kernels::load(&cuda, &f.path, &["fill"]) }.unwrap();
    assert!(unsafe { Kernels::load(&cuda, &f.path, &["absent"]) }.is_err());
    f.mode(MODE_CREATE_ERROR);
    assert!(unsafe { Kernels::load(&cuda, &f.path, &["fill"]) }.is_err());
    f.mode(MODE_NORMAL);
    let other = unsafe { Cuda::load(&f.path, 1) }.unwrap();
    let foreign = other.alloc(1).unwrap();
    let same_device = unsafe { Cuda::load(&f.path, 0) }.unwrap();
    let foreign_stream = same_device.alloc(1).unwrap();
    let before = f.trace();
    for args in [&foreign, &foreign_stream] {
        assert!(unsafe { kernels.launch("fill", &[args], 1, 16) }.is_err());
    }
    for (b, l) in [(0, 16), (3, 16), (32, 16), (1, 0), (1, 17), (1, 528)] {
        assert!(unsafe { kernels.launch("fill", &[&buffer], b, l) }.is_err());
    }
    assert!(unsafe { kernels.launch("absent", &[&buffer], 1, 16) }.is_err());
    assert_eq!(
        before,
        f.trace(),
        "rejected calls must not enter the runtime"
    );
    f.mode(MODE_COPY_ERROR);
    assert!(unsafe { kernels.launch("fill", &[&buffer], 1, 16) }.is_err());
    f.mode(MODE_NORMAL);
    drop(cuda);
    unsafe { kernels.launch("fill", &[&buffer], 1, 16) }.unwrap();
    assert!(f.trace().ends_with("device0 kernel0 "));
    drop(kernels);
    assert!(f.trace().ends_with("device0 sync0 "));
    assert_eq!(buffer.read(1).unwrap(), [73]);
}
