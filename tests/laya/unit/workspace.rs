use super::*;
use libloading::Library;
use std::{path::PathBuf, process::Command};
use tempfile::{TempDir, tempdir};

struct Fixture {
    _dir: TempDir,
    library: Library,
    cuda: Cuda,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempdir().unwrap();
        let path = dir
            .path()
            .join(format!("runtime{}", std::env::consts::DLL_SUFFIX));
        let output = Command::new("cc")
            .args([
                if cfg!(target_os = "macos") {
                    "-dynamiclib"
                } else {
                    "-shared"
                },
                "-fPIC",
            ])
            .arg(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../tests/backends/cuda/fixtures/runtime.c"),
            )
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let library = unsafe { Library::new(&path) }.unwrap();
        let cuda = unsafe { Cuda::load(&path, 0) }.unwrap();
        Self {
            _dir: dir,
            library,
            cuda,
        }
    }
    fn live(&self) -> i32 {
        live(&self.library)
    }
    fn trace(&self) -> String {
        unsafe {
            let trace = self
                .library
                .get::<unsafe extern "C" fn() -> *const std::ffi::c_char>(b"laya_test_trace\0")
                .unwrap();
            std::ffi::CStr::from_ptr(trace())
                .to_string_lossy()
                .into_owned()
        }
    }
}
fn live(library: &Library) -> i32 {
    unsafe {
        library
            .get::<unsafe extern "C" fn() -> i32>(b"laya_test_live\0")
            .unwrap()()
    }
}
fn buffers(s: &Workspace) -> [&Buffer; 17] {
    let s = s.buffers();
    [
        &s.ids,
        &s.lengths,
        &s.types,
        &s.residual,
        &s.hidden,
        &s.qkv,
        &s.attention,
        &s.gated,
        &s.feed_forward,
        &s.indices,
        &s.offsets,
        &s.markers,
        &s.scored,
        &s.logits,
        &s.features,
        &s.action_hidden,
        &s.actions,
    ]
}

#[test]
fn capacities_match_consumer_layouts_at_both_bounds() {
    let fixture = Fixture::new();
    for (batch, sequence, expected) in [
        (
            1,
            16,
            [
                128, 4, 8, 65536, 32768, 98304, 32768, 83968, 131072, 8192, 8, 4194304, 4194304,
                4096, 2056, 512, 4,
            ],
        ),
        (
            16,
            512,
            [
                65536, 64, 128, 33554432, 16777216, 50331648, 16777216, 42991616, 67108864, 8192,
                68, 4194304, 4194304, 4096, 32896, 8192, 64,
            ],
        ),
    ] {
        let workspace = Workspace::new(&fixture.cuda, batch, sequence).unwrap();
        assert_eq!(workspace.batch(), batch);
        assert_eq!(workspace.sequence(), sequence);
        assert_eq!(buffers(&workspace).map(Buffer::bytes), expected);
        assert_eq!(workspace.bytes(), expected.iter().sum::<usize>());
        assert_eq!(fixture.live(), 18);
        drop(workspace);
        assert_eq!(fixture.live(), 1);
    }
}

#[test]
fn invalid_shapes_allocate_nothing() {
    let fixture = Fixture::new();
    let before = fixture.trace();
    for batch in [0, 3, 17, 32, usize::MAX] {
        assert!(Workspace::new(&fixture.cuda, batch, 16).is_err());
    }
    for sequence in [0, 1, 15, 17, 511, 513, usize::MAX] {
        assert!(Workspace::new(&fixture.cuda, 1, sequence).is_err());
    }
    assert_eq!(fixture.live(), 1);
    assert_eq!(fixture.trace(), before);
}

#[test]
fn failed_allocations_release_the_partial_workspace() {
    for completed in [0, 1, 8, 16] {
        let fixture = Fixture::new();
        unsafe {
            fixture
                .library
                .get::<unsafe extern "C" fn(i32)>(b"laya_test_fail_alloc_after\0")
                .unwrap()(completed);
        }
        assert!(Workspace::new(&fixture.cuda, 1, 16).is_err());
        assert_eq!(fixture.live(), 1, "failed after {completed} allocations");
    }
}

#[test]
fn workspaces_do_not_alias_and_outlive_the_cuda_handle() {
    let fixture = Fixture::new();
    let first = Workspace::new(&fixture.cuda, 1, 16).unwrap();
    let second = Workspace::new(&fixture.cuda, 2, 32).unwrap();
    assert_eq!(fixture.live(), 35);
    drop(fixture.cuda);
    for (i, b) in buffers(&first)
        .into_iter()
        .chain(buffers(&second))
        .enumerate()
    {
        b.write(&vec![i as u8; b.bytes().min(32)]).unwrap();
    }
    for (i, b) in buffers(&first)
        .into_iter()
        .chain(buffers(&second))
        .enumerate()
    {
        assert_eq!(
            b.read(b.bytes().min(32)).unwrap(),
            vec![i as u8; b.bytes().min(32)]
        );
    }
    drop(first);
    assert_eq!(live(&fixture.library), 18);
    drop(second);
    assert_eq!(live(&fixture.library), 0);
}

#[test]
#[ignore = "requires approved GPU, LAYA_CUDA_LIBRARY and LAYA_CUDA_DEVICE"]
fn real_gpu_workspace_capacity_and_reuse() {
    let path = PathBuf::from(std::env::var_os("LAYA_CUDA_LIBRARY").unwrap());
    let device = std::env::var("LAYA_CUDA_DEVICE").unwrap().parse().unwrap();
    for (batch, sequence, expected_bytes) in
        [(1, 16, 8848032), (1, 512, 22628896), (16, 512, 236048836)]
    {
        let cuda = unsafe { Cuda::load(&path, device) }.unwrap();
        let workspace = Workspace::new(&cuda, batch, sequence).unwrap();
        drop(cuda);
        assert_eq!(workspace.bytes(), expected_bytes);
        for pass in 0..2 {
            let pattern = |size, index| {
                (0..size)
                    .map(|offset| ((offset % 251 + index * 7 + pass * 89) % 256) as u8)
                    .collect::<Vec<_>>()
            };
            for (i, b) in buffers(&workspace).into_iter().enumerate() {
                b.write(&pattern(b.bytes(), i)).unwrap();
            }
            for (i, b) in buffers(&workspace).into_iter().enumerate() {
                assert_eq!(
                    b.read(b.bytes()).unwrap(),
                    pattern(b.bytes(), i),
                    "buffer {i}, pass {pass}"
                );
            }
        }
        println!(
            "batch={batch} sequence={sequence} buffers=17 bytes={expected_bytes} verified_passes=2"
        );
    }
}
