use super::*;

#[test]
fn grouped_model_inputs_reuse_staging_and_update_all_values() {
    let fixture = fixture::Fixture::new(&[]);
    let mut model = fixture_model_at(false, fixture.path());
    let mut batch = Batch {
        b: 1,
        l: 16,
        input_ids: vec![0; 16],
        lens: vec![16],
        qtypes: vec![0],
        markers: vec![vec![0]],
    };
    let syncs = unsafe {
        model
            .cuda
            .symbol::<unsafe extern "C" fn() -> i32>(b"test_syncs\0")
            .unwrap()
    };
    let before = unsafe { syncs() };
    model.infer(&batch).unwrap();
    assert_eq!(
        unsafe { syncs() },
        before + 5,
        "one grouped input sync, two marker uploads, two synchronized readbacks"
    );
    let first = model.cache.back().unwrap().staging.as_ptr();
    batch.input_ids.fill(7);
    batch.lens[0] = 12;
    batch.qtypes[0] = 2;
    model.infer(&batch).unwrap();
    let workspace = model.cache.back().unwrap();
    assert_eq!(first, workspace.staging.as_ptr());
    assert_eq!(
        workspace.ids.read(16 * 8).unwrap(),
        [7i64.to_le_bytes(); 16].concat()
    );
    assert_eq!(workspace.lens.read(4).unwrap(), 12i32.to_le_bytes());
    assert_eq!(workspace.types.read(8).unwrap(), 2i64.to_le_bytes());
    batch.input_ids.fill(0);
    batch.lens[0] = 16;
    batch.qtypes[0] = 0;
    model.infer(&batch).unwrap();
    assert_eq!(
        model.cache.back().unwrap().ids.read(16 * 8).unwrap(),
        [0; 16 * 8]
    );
    let before = unsafe { syncs() };
    batch.qtypes[0] = 3;
    assert!(model.infer(&batch).is_err());
    assert_eq!(
        unsafe { syncs() },
        before,
        "invalid request does not submit or synchronize"
    );
    assert_eq!(first, model.cache.back().unwrap().staging.as_ptr());
}
