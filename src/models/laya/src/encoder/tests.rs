use super::*;

pub(super) type Checkpoints = std::cell::RefCell<Option<Vec<(String, Vec<u8>)>>>;

#[test]
fn input_bounds_cover_padding_and_vocabulary_edges() {
    assert!(validate_inputs(&[0; 32], &[16, 0], &[0, 2], 2, 16).is_ok());
    assert!(validate_inputs(&[50367; 16], &[16], &[1], 1, 16).is_ok());
    for ids in [vec![-1; 16], vec![50368; 16], vec![0; 15]] {
        assert!(validate_inputs(&ids, &[16], &[0], 1, 16).is_err());
    }
    for lens in [vec![-1], vec![17], vec![]] {
        assert!(validate_inputs(&[0; 16], &lens, &[0], 1, 16).is_err());
    }
    for types in [vec![-1], vec![3], vec![]] {
        assert!(validate_inputs(&[0; 16], &[16], &types, 1, 16).is_err());
    }
}

#[test]
#[ignore = "requires approved GPU, checkpoint, trusted kernel bundle and encoder oracle"]
fn real_encoder_matches_official_hidden_states() -> Result<()> {
    let path = |name| {
        std::env::var_os(name)
            .map(std::path::PathBuf::from)
            .expect(name)
    };
    let cuda = unsafe { Cuda::load(&path("LAYA_CUDA_LIBRARY"), 0) }?;
    let encoder =
        unsafe { Encoder::load(&cuda, &path("LAYA_CHECKPOINT"), &path("LAYA_KERNEL_BUNDLE")) }?;
    let root = path("LAYA_ENCODER_ORACLE");
    let cases: serde_json::Value = serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    for case in cases
        .as_array()
        .unwrap()
        .iter()
        .chain(cases.as_array().unwrap().iter().rev())
    {
        let name = case["name"].as_str().unwrap();
        let b = case["batch"].as_u64().unwrap() as usize;
        let l = case["sequence"].as_u64().unwrap() as usize;
        let ids: Vec<i64> = serde_json::from_value(case["ids"].clone())?;
        let lengths: Vec<i32> = serde_json::from_value(case["lengths"].clone())?;
        let types: Vec<i64> = serde_json::from_value(case["types"].clone())?;
        let workspace = Workspace::new(&cuda, b, l)?;
        // Kernels are asynchronous; readback must observe the completed final layer.
        *encoder.checkpoints.borrow_mut() = Some(Vec::new());
        encoder.run(&ids, &lengths, &types, &workspace)?;
        let got = workspace.buffers().residual.read(b * l * 1024 * 4)?;
        check_output(
            name,
            &got,
            &fs::read(root.join(format!("{name}.f32")))?,
            &lengths,
            l,
        )?;
        for (stage, output) in encoder.checkpoints.take().unwrap() {
            check_output(
                &format!("{name}/{stage}"),
                &output,
                &fs::read(root.join(format!("{name}-{stage}.f32")))?,
                &lengths,
                l,
            )?;
        }
        // Reusing the same allocations must not accumulate residuals across requests.
        encoder.run(&ids, &lengths, &types, &workspace)?;
        assert_eq!(
            got,
            workspace.buffers().residual.read(got.len())?,
            "{name}: repeat changed output"
        );
    }
    Ok(())
}

impl Encoder {
    pub(super) fn record(&self, stage: &str, buffer: &Buffer) -> Result<()> {
        if let Some(stages) = self.checkpoints.borrow_mut().as_mut() {
            stages.push((stage.into(), buffer.read(buffer.bytes())?));
        }
        Ok(())
    }
}

fn check_output(name: &str, got: &[u8], expected: &[u8], lengths: &[i32], l: usize) -> Result<()> {
    assert_eq!(expected.len(), got.len(), "{name}");
    let mut err2 = 0f64;
    let mut ref2 = 0f64;
    let mut max_error = 0f32;
    let mut max_ref = 0f32;
    for (j, (g, r)) in got
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expected.as_chunks::<4>().0)
        .enumerate()
    {
        let g = f32::from_le_bytes(*g);
        let r = f32::from_le_bytes(*r);
        assert!(g.is_finite(), "{name}: nonfinite output at {j}");
        // Padding queries have no observable output; patched empty-key attention differs there.
        if (j / 1024) % l >= lengths[j / (l * 1024)] as usize {
            continue;
        }
        assert!(r.is_finite());
        let error = g - r;
        err2 += f64::from(error).powi(2);
        ref2 += f64::from(r).powi(2);
        max_error = max_error.max(error.abs());
        max_ref = max_ref.max(r.abs());
    }
    let nrms = (err2 / ref2.max(1e-24)).sqrt();
    assert!(
        nrms <= 0.005 && max_error <= 0.02 * max_ref.max(1.),
        "{name}: nrms={nrms}, max_abs={max_error}, ref_max={max_ref}"
    );
    println!("ENCODER {name} nrms={nrms} max_abs={max_error} ref_max={max_ref}");
    Ok(())
}
