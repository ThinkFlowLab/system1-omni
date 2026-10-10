use super::*;

#[test]
fn prepared_plan_preserves_original_launch_trace_for_all_shape_selectors() {
    let fixture = fixture::Fixture::new(&[]);
    let lib = unsafe { Cuda::load(fixture.path()) }.unwrap();
    let reset = unsafe {
        lib.symbol::<unsafe extern "C" fn()>(b"test_reset\0")
            .unwrap()
    };
    let trace = unsafe {
        lib.symbol::<unsafe extern "C" fn() -> *const i8>(b"test_trace\0")
            .unwrap()
    };
    for original_rope in [false, true] {
        let model = fixture_model_at(original_rope, fixture.path());
        for b in [1, 2, 4, 8, 16] {
            for l in [16, 512] {
                let s = Workspace::new(&model.cuda, b, l).unwrap();
                unsafe { reset() };
                model.encode_original(&s, original_rope).unwrap();
                let expected = unsafe { std::ffi::CStr::from_ptr(trace()) }
                    .to_bytes()
                    .to_vec();
                unsafe { reset() };
                model.encode(&s).unwrap();
                let actual = unsafe { std::ffi::CStr::from_ptr(trace()) }
                    .to_bytes()
                    .to_vec();
                assert_eq!(
                    expected, actual,
                    "B={b}, L={l}, original_rope={original_rope}"
                );
                assert_eq!(actual.split(|x| *x == b'\n').count() - 1, 242);
            }
        }
    }
}

#[test]
fn missing_specialized_attention_only_rejects_its_shape() {
    let fixture = fixture::Fixture::new(&["OMIT_SPECIALIZED_ATTN"]);
    let model = fixture_model_at(false, fixture.path());
    let short = Workspace::new(&model.cuda, 1, 16).unwrap();
    model.encode(&short).unwrap();
    let generic_long = Workspace::new(&model.cuda, 2, 512).unwrap();
    model.encode(&generic_long).unwrap();
    for b in [1, 4] {
        let specialized = Workspace::new(&model.cuda, b, 512).unwrap();
        assert!(
            model
                .encode(&specialized)
                .unwrap_err()
                .to_string()
                .contains("missing specialized attention")
        );
    }
}
