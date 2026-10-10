//! GPU checks of the attention and Gated DeltaNet kernels on random inputs. They need
//! a GPU and CUA_S1_CUDA_LIB pointing at libqwen3_5_cuda.so, so they only run when
//! asked for:
//!
//!     CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
//!       cargo test --release -p omni-qwen3-5-native --test kernels -- --ignored
//!
//! The prefix-state test additionally needs QWEN3_5_CHECKPOINT pointing to a
//! supported multimodal checkpoint; it loads two instances sequentially.

use std::path::PathBuf;

use half::bf16;
use omni_qwen3_5_native::cuda::{self, DeviceBuffer, Stream, api, check};

fn setup() -> Stream {
    let lib = std::env::var_os("CUA_S1_CUDA_LIB")
        .map(PathBuf::from)
        .expect("CUA_S1_CUDA_LIB must point at libqwen3_5_cuda.so");
    cuda::load(&lib).unwrap();
    cuda::set_device(0).unwrap();
    cuda::new_stream().unwrap()
}

/// Uniform values in [-amp, amp), rounded to bfloat16, from a fixed seed.
fn random(n: usize, seed: u64, amp: f32) -> Vec<bf16> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            bf16::from_f32(((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp)
        })
        .collect()
}

fn to_device(v: &[bf16], st: Stream) -> DeviceBuffer {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let buf = DeviceBuffer::new(bytes.len()).unwrap();
    // SAFETY: the buffer was allocated for these bytes.
    unsafe { cuda::upload(buf.at(0), &bytes, st).unwrap() };
    buf
}

fn f32_to_device(v: &[f32], st: Stream) -> DeviceBuffer {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let buf = DeviceBuffer::new(bytes.len()).unwrap();
    // SAFETY: the buffer was allocated for these bytes.
    unsafe { cuda::upload(buf.at(0), &bytes, st).unwrap() };
    buf
}

fn from_device(buf: &DeviceBuffer, n: usize, st: Stream) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 2];
    // SAFETY: the buffer holds n bfloat16 values.
    unsafe { cuda::download(&mut bytes, buf.at(0), st).unwrap() };
    let (pairs, _) = bytes.as_chunks::<2>();
    pairs
        .iter()
        .map(|&b| bf16::from_le_bytes(b).to_f32())
        .collect()
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn residual_rms_norm_matches_rounded_reference() {
    let st = setup();
    let eps = 1e-6f32;
    // Cua-S1 and Open-Jev widths, scalar fallbacks for other widths, and a BF16
    // element offset to check that cached loads need no packed alignment.
    for (d, offset) in [
        (2560usize, 0usize),
        (5120, 0),
        (8192, 0),
        (257, 0),
        (8448, 0),
        (5120, 1),
    ] {
        for rows in [1usize, 3, 107] {
            let n = rows * d;
            let mut residual = random(n + offset, 5, 3.0);
            let mut delta = random(n + offset, 6, 0.5);
            let mut weights = random(d + offset, 7, 0.25);
            // Exercise the BF16 rounding between addition and normalization.
            residual[offset] = bf16::ONE;
            delta[offset] = bf16::from_f32(1.0 / 256.0);
            weights[offset + 1] = bf16::from_f32(-1.0);
            let rounded: Vec<bf16> = residual[offset..]
                .iter()
                .zip(&delta[offset..])
                .map(|(r, d)| bf16::from_f32(r.to_f32() + d.to_f32()))
                .collect();
            let input = to_device(&residual, st);
            let change = to_device(&delta, st);
            let weight = to_device(&weights, st);
            let output = DeviceBuffer::new((n + offset) * 2).unwrap();
            // SAFETY: complete BF16 rows and d weights after the optional one-
            // element offset, with separate input, delta, weight, and output.
            unsafe {
                check(
                    (api().cs1_add_rms_norm)(
                        input.at(offset * 2),
                        change.at(offset * 2),
                        weight.at(offset * 2),
                        output.at(offset * 2),
                        rows as i32,
                        d as i32,
                        eps,
                        st,
                    ),
                    "residual norm",
                )
                .unwrap();
            }
            let added = from_device(&input, n + offset, st);
            assert_eq!(
                added[offset..],
                rounded.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                "residual addition changed BF16 rounding at d={d}, offset={offset}"
            );
            if offset != 0 {
                assert_eq!(added[0], residual[0].to_f32());
            }
            let got = from_device(&output, n + offset, st);
            for (row, values) in rounded.chunks_exact(d).enumerate() {
                let ss: f64 = values.iter().map(|v| (v.to_f32() as f64).powi(2)).sum();
                let inv = ((ss / d as f64) as f32 + eps).sqrt().recip();
                for (i, value) in values.iter().enumerate() {
                    let want =
                        bf16::from_f32(value.to_f32() * inv * (1.0 + weights[offset + i].to_f32()));
                    let actual = bf16::from_f32(got[offset + row * d + i]);
                    assert!(actual.is_finite());
                    // Reduction order and CUDA rsqrt may move a value by one
                    // BF16 ULP; residual addition above must remain exact.
                    assert!(
                        actual.to_bits().abs_diff(want.to_bits()) <= 1,
                        "norm mismatch d={d}, offset={offset}, row={row}, i={i}: {actual} vs {want}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn silu_mul_matches_rounded_reference() {
    let st = setup();
    // Packed model widths, padded rows, and scalar fallbacks for an odd width
    // or stride. A one-element pointer offset forces the scalar path on the
    // same inputs, which must agree exactly with packed execution.
    for (width, ld) in [(8192usize, 16384usize), (17408, 34816), (64, 136), (37, 79)] {
        for rows in [1usize, 3, 107] {
            let input = random(rows * ld, 23, 12.0);
            let mut results = Vec::new();
            for offset in [0usize, 1] {
                let mut padded = vec![bf16::ZERO; offset];
                padded.extend_from_slice(&input);
                let source = to_device(&padded, st);
                let output = DeviceBuffer::new((rows * width + offset) * 2).unwrap();
                // SAFETY: complete gate/up rows and an output of rows*width
                // BF16 elements after the optional one-element offset.
                unsafe {
                    check(
                        (api().cs1_silu_mul)(
                            source.at(offset * 2),
                            ld as i32,
                            output.at(offset * 2),
                            rows as i32,
                            width as i32,
                            st,
                        ),
                        "silu_mul",
                    )
                    .unwrap();
                }
                results.push(from_device(&output, rows * width + offset, st)[offset..].to_vec());
            }
            assert_eq!(
                results[0], results[1],
                "packed/scalar mismatch at {width}/{ld}"
            );
            for (row, values) in input.chunks_exact(ld).enumerate() {
                for j in 0..width {
                    let gate = values[j].to_f32() as f64;
                    let activated = bf16::from_f64(gate / (1.0 + (-gate).exp()));
                    let want = bf16::from_f32(activated.to_f32() * values[width + j].to_f32());
                    let actual = bf16::from_f32(results[0][row * width + j]);
                    assert!(actual.is_finite());
                    assert!(
                        actual.to_bits().abs_diff(want.to_bits()) <= 1,
                        "SiLU rounding mismatch at {width}/{ld}, row={row}, j={j}: {actual} vs {want}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn graph_replay_reads_updated_inputs_after_failed_capture() {
    let st = setup();
    // A failed recording must end capture so this stream can be captured again.
    assert!(cuda::Graph::capture(st, || anyhow::bail!("recording failed")).is_err());
    let table: Vec<bf16> = (0..24).map(|i| bf16::from_f32(i as f32)).collect();
    let weights = to_device(&table, st);
    let ids = DeviceBuffer::new(8).unwrap();
    let output = DeviceBuffer::new(32).unwrap();
    let embed = || {
        // SAFETY: two int32 ids, three embedding rows of width eight, two output rows.
        check(
            unsafe { (api().cs1_embed)(ids.at(0).cast(), weights.at(0), output.at(0), 2, 8, st) },
            "capture embed",
        )
    };
    let graph = cuda::Graph::capture(st, embed).unwrap();
    let assert_replay = |graph: &cuda::Graph, rows: [i32; 2]| {
        let bytes: Vec<u8> = rows.iter().flat_map(|id| id.to_le_bytes()).collect();
        // SAFETY: ids holds two int32 values; every id is a valid embedding row.
        unsafe { cuda::upload(ids.at(0), &bytes, st).unwrap() };
        graph.launch(st).unwrap();
        let expected: Vec<f32> = rows
            .iter()
            .flat_map(|&row| (row * 8..row * 8 + 8).map(|i| i as f32))
            .collect();
        assert_eq!(from_device(&output, 16, st), expected);
    };
    for rows in [[0i32, 1], [2, 0], [1, 2]] {
        assert_replay(&graph, rows);
    }
    // SAFETY: zero is CUDA's valid legacy default stream handle. Capturing it
    // is unsupported and must report an error without poisoning this thread.
    let default_stream: Stream = unsafe { std::mem::zeroed() };
    assert_ne!(unsafe { (api().cs1_graph_begin)(default_stream) }, 0);
    let recovered = cuda::Graph::capture(st, embed).unwrap();
    assert_replay(&recovered, [0, 1]);
    // SAFETY: a null graph handle deliberately exercises CUDA's argument error.
    assert_ne!(
        unsafe { (api().cs1_graph_launch)(std::ptr::null_mut(), st) },
        0
    );
    let recovered = cuda::Graph::capture(st, embed).unwrap();
    assert_replay(&recovered, [2, 0]);
    for propagate in [true, false] {
        let error = cuda::Graph::capture(st, || {
            embed()?;
            // Synchronizing a capturing stream invalidates the capture (900).
            // EndCapture then reports 901, even if the closure returns Ok.
            // SAFETY: st is a live stream created by setup.
            let code = unsafe { (api().cs1_stream_sync)(st) };
            assert_eq!(code, 900);
            if propagate {
                check(code, "invalidate capture")
            } else {
                Ok(())
            }
        })
        .err()
        .expect("synchronization must invalidate capture");
        let expected_code = if propagate { "(900)" } else { "(901)" };
        assert!(error.to_string().contains(expected_code), "{error}");

        // Retained-graph replay and download do not consume CUDA's last error.
        // Recapture must work on this same thread without clearing it here.
        assert_replay(&graph, [2, 1]);
        let recovered = cuda::Graph::capture(st, embed).unwrap();
        assert_replay(&recovered, [0, 2]);
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn fused_attention_gate_matches_separate_pass() {
    let st = setup();
    let dh = 256usize;
    for (hq, hk) in [(24usize, 4usize), (16, 4), (4, 4), (16, 1)] {
        for t in [1usize, 63, 64, 65, 139, 712, 2048] {
            let ldv = hk * dh + 16; // exercise a strided V buffer
            let q = to_device(&random(t * hq * dh, 1, 2.0), st);
            let k = to_device(&random(t * hk * dh, 2, 2.0), st);
            let v = to_device(&random(t * ldv, 3, 1.0), st);
            let n = t * hq * dh;
            let mut gates = random(n, 4, 12.0);
            // Saturated sigmoid, zero, and ordinary values in every output row.
            for row in gates.chunks_exact_mut(dh) {
                row[0] = bf16::from_f32(-40.0);
                row[1] = bf16::from_f32(40.0);
                row[2] = bf16::ZERO;
            }
            let gate = to_device(&gates, st);
            let separate = DeviceBuffer::new(n * 2).unwrap();
            let fused = DeviceBuffer::new(n * 2).unwrap();
            // SAFETY: all buffers have complete rows of the widths above, with
            // disjoint output and gate allocations, and use the same stream.
            unsafe {
                check(
                    (api().cs1_attention)(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        ldv as i32,
                        separate.at(0),
                        t as i32,
                        hq as i32,
                        hk as i32,
                        dh as i32,
                        0.0625,
                        st,
                    ),
                    "separate attention",
                )
                .unwrap();
                check(
                    (api().cs1_sigmoid_gate)(separate.at(0), gate.at(0), n, st),
                    "separate gate",
                )
                .unwrap();
                check(
                    (api().cs1_attention_gated)(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        ldv as i32,
                        gate.at(0),
                        fused.at(0),
                        t as i32,
                        hq as i32,
                        hk as i32,
                        dh as i32,
                        0.0625,
                        st,
                    ),
                    "fused attention gate",
                )
                .unwrap();
            }
            let want = from_device(&separate, n, st);
            let got = from_device(&fused, n, st);
            assert!(
                got.iter().all(|x| x.is_finite()),
                "non-finite result at t={t}"
            );
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "fused gate changed BF16 output at t={t}, hq={hq}, hk={hk}"
            );
        }
    }
    // Empty work is valid without a gate buffer; nonempty work must have one.
    // SAFETY: these invalid/empty shapes return before any kernel is launched.
    unsafe {
        let null = std::ptr::null();
        assert_eq!(
            (api().cs1_attention_gated)(
                null,
                null,
                null,
                256,
                null,
                std::ptr::null_mut(),
                0,
                1,
                1,
                256,
                0.0625,
                st
            ),
            0
        );
        assert_ne!(
            (api().cs1_attention_gated)(
                null,
                null,
                null,
                256,
                null,
                std::ptr::null_mut(),
                1,
                1,
                1,
                256,
                0.0625,
                st
            ),
            0
        );
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn flash_attention_matches_float64_reference() {
    let st = setup();
    let (hq, hk, dh) = (16usize, 4usize, 256usize);
    for (t, amp) in [
        (1, 8.0),
        (63, 8.0),
        (65, 0.5),
        (139, 2.0),
        (700, 8.0),
        (2048, 0.5),
    ] {
        let (qh, kh) = (random(t * hq * dh, 1, amp), random(t * hk * dh, 2, amp));
        // v is read in place from the q|k|v projection output, rows of 10240 as in the model
        let (ldv, v_at) = (10240usize, (hq * 2 + hk) * dh);
        let qkvh = random(t * ldv, 3, 1.0);
        let (q, k, qkv) = (to_device(&qh, st), to_device(&kh, st), to_device(&qkvh, st));
        let out = DeviceBuffer::new(t * hq * dh * 2).unwrap();
        // SAFETY: every buffer holds t rows of the given widths.
        let code = unsafe {
            (api().cs1_attention)(
                q.at(0),
                k.at(0),
                qkv.at(v_at * 2),
                ldv as i32,
                out.at(0),
                t as i32,
                hq as i32,
                hk as i32,
                dh as i32,
                0.0625,
                st,
            )
        };
        check(code, "attention").unwrap();
        let got = from_device(&out, t * hq * dh, st);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "non-finite output at t = {t}"
        );
        // about 64 query rows per length, each against causal attention in float64;
        // per (row, head): the largest difference over the largest magnitude
        let mut worst = 0f64;
        for i in (0..t).step_by(t.div_ceil(64)).chain([t - 1]) {
            for h in 0..hq {
                let g = h / (hq / hk);
                let qi = &qh[(i * hq + h) * dh..][..dh];
                let s: Vec<f64> = (0..=i)
                    .map(|j| {
                        let kj = &kh[(j * hk + g) * dh..][..dh];
                        qi.iter()
                            .zip(kj)
                            .map(|(a, b)| a.to_f64() * b.to_f64())
                            .sum::<f64>()
                            * 0.0625
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = w.iter().sum();
                let want: Vec<f64> = (0..dh)
                    .map(|d| {
                        (0..=i)
                            .map(|j| w[j] * qkvh[j * ldv + v_at + g * dh + d].to_f64())
                            .sum::<f64>()
                            / z
                    })
                    .collect();
                let row = &got[(i * hq + h) * dh..][..dh];
                let diff = row
                    .iter()
                    .zip(&want)
                    .map(|(&x, y)| (x as f64 - y).abs())
                    .fold(0f64, f64::max);
                worst = worst.max(diff / want.iter().map(|y| y.abs()).fold(1e-3, f64::max));
            }
        }
        eprintln!("attention t = {t}, amplitude {amp}: largest relative difference {worst:.2e}");
        assert!(worst < 1.6e-2, "t = {t}: {worst}");
    }
}

/// Transformers' torch_recurrent_gated_delta_rule in float64, one token at a time,
/// with the L2 norms of q and k and q scaled by K^-1/2.
#[allow(clippy::too_many_arguments)]
fn gated_delta_reference(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    g: &[f32],
    beta: &[bf16],
    t: usize,
    h: usize,
    hk: usize,
    d: usize,
) -> Vec<f64> {
    let mut out = vec![0f64; t * h * d];
    for head in 0..h {
        let kh = head / (h / hk);
        let mut s = vec![0f64; d * d]; // [K][V]
        for tok in 0..t {
            let norm = |x: &[bf16]| {
                let x: Vec<f64> = x.iter().map(|v| v.to_f64()).collect();
                let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
                x.into_iter().map(|v| v * inv).collect::<Vec<f64>>()
            };
            let qv: Vec<f64> = norm(&q[(tok * hk + kh) * d..][..d])
                .into_iter()
                .map(|x| x / (d as f64).sqrt())
                .collect();
            let kv = norm(&k[(tok * hk + kh) * d..][..d]);
            let vv: Vec<f64> = v[(tok * h + head) * d..][..d]
                .iter()
                .map(|x| x.to_f64())
                .collect();
            let decay = (g[tok * h + head] as f64).exp();
            let b = beta[tok * h + head].to_f64();
            s.iter_mut().for_each(|x| *x *= decay);
            for j in 0..d {
                let mem: f64 = (0..d).map(|i| kv[i] * s[i * d + j]).sum();
                let delta = (vv[j] - mem) * b;
                for i in 0..d {
                    s[i * d + j] += kv[i] * delta;
                }
            }
            for j in 0..d {
                out[(tok * h + head) * d + j] = (0..d).map(|i| qv[i] * s[i * d + j]).sum();
            }
        }
    }
    out
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn gated_delta_rule_matches_recurrent_reference() {
    let st = setup();
    let d = 128usize;
    // Tile/chunk boundaries, weak decay, grouped heads, and near-zero q/k.
    // Keep the float64 recurrent reference and its tolerance.
    for (t, h, hk, decay_scale, qk_amp) in [
        (1usize, 4usize, 2usize, 1.0f32, 1.0f32),
        (16, 4, 2, 1.0, 1.0),
        (17, 4, 2, 1.0, 1.0),
        (32, 4, 2, 1.0, 1.0),
        (33, 4, 2, 1.0, 1.0),
        (63, 4, 2, 1.0, 1.0),
        (64, 4, 2, 1.0, 1.0),
        (65, 4, 2, 1.0, 1.0),
        (107, 4, 2, 1.0, 1.0),
        (127, 4, 2, 1.0, 1.0),
        (128, 4, 2, 1.0, 1.0),
        (129, 4, 2, 1.0, 1.0),
        (150, 4, 2, 1.0, 1.0),
        (107, 4, 2, 0.01, 1.0),
        (936, 4, 2, 0.01, 1.0),
        (936, 6, 2, 0.01, 1.0),
        (3399, 3, 1, 0.01, 1.0),
        (65, 4, 2, 0.01, 1e-9),
        (65, 4, 2, 0.01, 1e-18),
        (65, 4, 2, 0.01, 0.0),
    ] {
        // q close to k, so that q.k and the outputs are of order one as in the model
        let k = random(t * hk * d, 12, qk_amp);
        let q: Vec<bf16> = k
            .iter()
            .zip(random(t * hk * d, 11, qk_amp))
            .map(|(k, n)| bf16::from_f32(0.8 * k.to_f32() + 0.2 * n.to_f32()))
            .collect();
        let v = random(t * h * d, 13, 1.0);
        // log decays in (-2, 0) and learning rates in (0, 1), as sigmoid and -exp * softplus give
        let g: Vec<f32> = random(t * h, 14, 1.0)
            .iter()
            .map(|x| (x.to_f32() - 1.0) * decay_scale)
            .collect();
        let beta: Vec<bf16> = random(t * h, 15, 0.5)
            .iter()
            .map(|x| bf16::from_f32(x.to_f32() + 0.5))
            .collect();
        let want = gated_delta_reference(&q, &k, &v, &g, &beta, t, h, hk, d);
        let (qd, kd, vd, gd, bd) = (
            to_device(&q, st),
            to_device(&k, st),
            to_device(&v, st),
            f32_to_device(&g, st),
            to_device(&beta, st),
        );
        let o = DeviceBuffer::new(t * h * d * 2).unwrap();
        // SAFETY: pure function of its arguments.
        let floats = unsafe { (api().cs1_gdn_workspace_floats)(t as i32, h as i32) };
        let ws = DeviceBuffer::new(floats * 4).unwrap();
        // SAFETY: every buffer holds t rows of the given widths, the workspace its size.
        unsafe {
            check(
                (api().cs1_gdn_prefill)(
                    qd.at(0),
                    kd.at(0),
                    vd.at(0),
                    gd.at(0).cast::<f32>(),
                    bd.at(0),
                    o.at(0),
                    ws.at(0).cast::<f32>(),
                    t as i32,
                    h as i32,
                    hk as i32,
                    (d as f32).powf(-0.5),
                    st,
                ),
                "gdn prefill",
            )
            .unwrap();
        }
        let got = from_device(&o, t * h * d, st);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "non-finite GDN at t = {t}"
        );
        let scale = want.iter().fold(0f64, |m, x| m.max(x.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0f64, f64::max);
        eprintln!(
            "gated delta t = {t}, amplitude {qk_amp}: largest difference {worst:.2e}, largest |reference| {scale:.2e}"
        );
        assert!(worst <= 2e-2 * scale, "t = {t}: {worst} vs scale {scale}");
    }
}

fn f32_from_device(buf: &DeviceBuffer, n: usize, st: Stream) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    // SAFETY: the buffer holds n float32 values.
    unsafe { cuda::download(&mut bytes, buf.at(0), st).unwrap() };
    let (quad, _) = bytes.as_chunks::<4>();
    quad.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB, and a multimodal QWEN3_5_CHECKPOINT"]
fn prefix_state_rejects_uncaptured_foreign_and_oversized_prompts() {
    use omni_qwen3_5_native::inputs::MultimodalInput;
    use omni_qwen3_5_native::model::Model;

    let library = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("set CUA_S1_CUDA_LIB"));
    let checkpoint =
        PathBuf::from(std::env::var_os("QWEN3_5_CHECKPOINT").expect("set QWEN3_5_CHECKPOINT"));
    let mut first = Model::load(&checkpoint, &library).unwrap();
    for len in [(first.cfg.max_positions / 64 + 1) * 64, usize::MAX & !63] {
        let error = first.alloc_prefix(len).err().expect("oversized prefix");
        assert!(error.to_string().contains("exceeds the configured maximum"));
    }
    let mut state = first.alloc_prefix(64).unwrap();
    let ids = vec![0u32; state.token_count()];
    let positions: Vec<i64> = (0..state.token_count() as i64).collect();
    let input = MultimodalInput {
        token_ids: &ids,
        image_token_indices: &[],
        image_embeddings: &[],
        position_ids: [&positions; 3],
    };
    let error = first
        .forward_multimodal_continue(&input, &state)
        .unwrap_err();
    assert!(error.to_string().contains("has not completed capture"));

    first
        .forward_multimodal_capture(&input, &mut state)
        .unwrap();
    // The suffix fits by itself; the cached prefix pushes the full prompt over
    // the limit. Reject it before allocating scratch or submitting CUDA work.
    let suffix_ids = vec![0u32; first.cfg.max_positions];
    let suffix_positions = vec![0i64; suffix_ids.len()];
    let suffix = MultimodalInput {
        token_ids: &suffix_ids,
        image_token_indices: &[],
        image_embeddings: &[],
        position_ids: [&suffix_positions; 3],
    };
    let error = first
        .forward_multimodal_continue(&suffix, &state)
        .unwrap_err();
    assert!(error.to_string().contains("cached prompt exceeds"));

    // Retain only the prefix while replacing the model; never hold two sets of
    // checkpoint weights on the GPU. The old identity must remain distinct.
    drop(first);
    let mut second = Model::load(&checkpoint, &library).unwrap();
    let error = second
        .forward_multimodal_capture(&input, &mut state)
        .unwrap_err();
    assert!(error.to_string().contains("different model instance"));
    let error = second
        .forward_multimodal_continue(&input, &state)
        .unwrap_err();
    assert!(error.to_string().contains("different model instance"));
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn gdn_empty_window_captures_state_copy_and_zero_initial_state() {
    let st = setup();
    let h = 2usize;
    let n = h * 128 * 128;
    let state = f32_to_device(&vec![1.0; n], st);
    let output = f32_to_device(&vec![2.0; n], st);
    let empty_window = |input: *const std::ffi::c_void| {
        // SAFETY: T=0 touches only the optional input and output state buffers,
        // both sized [H,128,128] FP32; token and workspace pointers are unused.
        unsafe {
            check(
                (api().cs1_gdn_prefill_x)(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    h as i32,
                    1,
                    1.0,
                    input,
                    output.at(0),
                    st,
                ),
                "empty gdn window",
            )
        }
    };
    let copy = cuda::Graph::capture(st, || empty_window(state.at(0))).unwrap();
    let updated: Vec<u8> = vec![3.0f32; n]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    // Change the source after capture: a default-stream copy during capture
    // cannot substitute for the copy that must execute on every graph launch.
    // SAFETY: state holds exactly n float32 values.
    unsafe { cuda::upload(state.at(0), &updated, st).unwrap() };
    copy.launch(st).unwrap();
    assert_eq!(f32_from_device(&output, n, st), vec![3.0; n]);

    let zero = cuda::Graph::capture(st, || empty_window(std::ptr::null())).unwrap();
    // SAFETY: output holds exactly n float32 values; poison it after capture.
    unsafe { cuda::upload(output.at(0), &updated, st).unwrap() };
    zero.launch(st).unwrap();
    assert_eq!(f32_from_device(&output, n, st), vec![0.0; n]);
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn device_copy_helpers_roundtrip() {
    let st = setup();
    // copy_dd keeps every bit (odd size exercises plain memcpy).
    let a = to_device(&random(4099, 1, 2.0), st);
    let b = DeviceBuffer::new(4099 * 2).unwrap();
    // SAFETY: same size, non-overlapping device allocations.
    unsafe { cuda::copy_dd(b.at(0), a.at(0), 4099 * 2, st).unwrap() };
    assert_eq!(from_device(&a, 4099, st), from_device(&b, 4099, st));
    // copy2d copies width-sized windows out of pitched rows.
    let (rows, ld, w) = (5usize, 300usize, 256usize);
    let src = to_device(&random(rows * ld, 2, 1.0), st);
    let dst = DeviceBuffer::new(rows * w * 2).unwrap();
    // SAFETY: dst holds rows of w, src rows of ld.
    unsafe {
        cuda::copy2d(dst.at(0), w * 2, src.at(0), ld * 2, w * 2, rows, st).unwrap();
    }
    let full = from_device(&src, rows * ld, st);
    let want: Vec<f32> = full
        .chunks_exact(ld)
        .flat_map(|r| &r[..w])
        .copied()
        .collect();
    assert_eq!(from_device(&dst, rows * w, st), want);
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn gdn_conv_continuation_uses_cached_history() {
    let st = setup();
    let prefix = 64usize;
    // Decider2B uses sixteen key/value heads; preserve the 27B control too.
    for (key_dim, value_dim, heads) in [(2048usize, 2048usize, 16usize), (2048, 6144, 48)] {
        let channels = 2 * key_dim + value_dim;
        let ld = channels + value_dim + 2 * heads;
        let weights = to_device(&vec![bf16::from_f32(0.25); channels * 4], st);
        let conv = |source: &DeviceBuffer, start: usize, rows: usize| {
            let q = DeviceBuffer::new(rows * key_dim * 2).unwrap();
            let k = DeviceBuffer::new(rows * key_dim * 2).unwrap();
            let v = DeviceBuffer::new(rows * value_dim * 2).unwrap();
            // SAFETY: source contains start + rows complete pitched rows; outputs
            // contain rows of the corresponding Q/K/V widths.
            unsafe {
                check(
                    (api().cs1_gdn_conv)(
                        source.at(start * ld * 2),
                        ld as i32,
                        weights.at(0),
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        rows as i32,
                        key_dim as i32,
                        value_dim as i32,
                        st,
                    ),
                    "gdn conv continuation",
                )
                .unwrap();
            }
            [
                from_device(&q, rows * key_dim, st),
                from_device(&k, rows * key_dim, st),
                from_device(&v, rows * value_dim, st),
            ]
        };
        for rows in [1usize, 2, 3, 4, 65] {
            let total = prefix + rows;
            // Positive inputs and taps make omission of any history row observable.
            let projection: Vec<bf16> = random(total * ld, 37, 0.5)
                .iter()
                .map(|x| bf16::from_f32(x.to_f32().abs() + 0.25))
                .collect();
            let source = to_device(&projection, st);
            let full = conv(&source, 0, total);
            let tail = DeviceBuffer::new(3 * ld * 2).unwrap();
            let window = DeviceBuffer::new((rows + 3) * ld * 2).unwrap();
            // SAFETY: capture three prefix rows, restore them ahead of the suffix,
            // and copy the suffix into the remaining non-overlapping window rows.
            unsafe {
                cuda::copy_dd(tail.at(0), source.at((prefix - 3) * ld * 2), 3 * ld * 2, st)
                    .unwrap();
                cuda::copy_dd(window.at(0), tail.at(0), 3 * ld * 2, st).unwrap();
                cuda::copy_dd(
                    window.at(3 * ld * 2),
                    source.at(prefix * ld * 2),
                    rows * ld * 2,
                    st,
                )
                .unwrap();
            }
            let restored = conv(&window, 0, rows + 3);
            let missing_history = conv(&window, 3, rows);
            for (i, width) in [key_dim, key_dim, value_dim].into_iter().enumerate() {
                let expected = &full[i][prefix * width..];
                assert_eq!(
                    &restored[i][3 * width..],
                    expected,
                    "cached conv history diverges for output {i}, suffix rows={rows}"
                );
                assert_ne!(
                    &missing_history[i][..rows.min(3) * width],
                    &expected[..rows.min(3) * width],
                    "fixture must detect the old suffix-only conv call"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn windowed_attention_matches_full_pass_bit_for_bit() {
    let st = setup();
    // Canonical window indexing covers Decider2B and the existing 27B shape.
    for (hq, hk, dh) in [(8usize, 2usize, 256usize), (24, 4, 256)] {
        for (t, bases) in [
            (139usize, vec![64usize]),
            (712, vec![64, 128]),
            (972, vec![64, 896]),
            (2048, vec![64, 1024]),
        ] {
            // gated path with a strided V buffer, exactly the model's shape
            let ldv = hk * dh + 16;
            let q = to_device(&random(t * hq * dh, 1, 2.0), st);
            let k = to_device(&random(t * hk * dh, 2, 2.0), st);
            let v = to_device(&random(t * ldv, 3, 1.0), st);
            let gate = to_device(&random(t * hq * dh, 4, 3.0), st);
            let full = DeviceBuffer::new(t * hq * dh * 2).unwrap();
            // SAFETY: every buffer has complete rows of the shapes above.
            unsafe {
                check(
                    (api().cs1_attention_gated)(
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        ldv as i32,
                        gate.at(0),
                        full.at(0),
                        t as i32,
                        hq as i32,
                        hk as i32,
                        dh as i32,
                        0.0625,
                        st,
                    ),
                    "full pass",
                )
                .unwrap();
            }
            let full_rows = from_device(&full, t * hq * dh, st);
            for qb in &bases {
                let win = DeviceBuffer::new(t * hq * dh * 2).unwrap();
                // SAFETY: the window reads the same buffers; rows < q_base stay unwritten.
                unsafe {
                    check(
                        (api().cs1_attention_gated_prefix)(
                            q.at(0),
                            k.at(0),
                            v.at(0),
                            ldv as i32,
                            gate.at(0),
                            win.at(0),
                            t as i32,
                            hq as i32,
                            hk as i32,
                            dh as i32,
                            0.0625,
                            *qb as i32,
                            st,
                        ),
                        "windowed pass",
                    )
                    .unwrap();
                }
                let win_rows = from_device(&win, t * hq * dh, st);
                let mut bad = 0usize;
                for row in *qb..t {
                    for e in 0..hq * dh {
                        if full_rows[row * hq * dh + e] != win_rows[row * hq * dh + e] {
                            bad += 1;
                        }
                    }
                }
                assert_eq!(bad, 0, "t = {t}, q_base = {qb}: {bad} values differ");
                eprintln!(
                    "windowed attention t = {t}, q_base = {qb}: rows >= q_base bitwise equal"
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn gdn_prefill_two_phase_matches_one_shot_bit_for_bit() {
    let st = setup();
    let d = 128usize;
    for (t, h, hk, qk_amp, decay_scale) in [
        (107usize, 4usize, 2usize, 1.0f32, 1.0f32),
        (129, 16, 16, 0.01, 1.0),
        (936, 16, 16, 0.01, 1.0),
        (936, 4, 2, 0.01, 1.0),
        (3399, 3, 1, 0.01, 1.0),
        (65, 4, 2, 0.01, 1e-9),
    ] {
        let k = random(t * hk * d, 12, qk_amp);
        let q: Vec<bf16> = k
            .iter()
            .zip(random(t * hk * d, 11, qk_amp))
            .map(|(k, n)| bf16::from_f32(0.8 * k.to_f32() + 0.2 * n.to_f32()))
            .collect();
        let v = random(t * h * d, 13, 1.0);
        let g: Vec<f32> = random(t * h, 14, 1.0)
            .iter()
            .map(|x| (x.to_f32() - 1.0) * decay_scale)
            .collect();
        let beta: Vec<bf16> = random(t * h, 15, 0.5)
            .iter()
            .map(|x| bf16::from_f32(x.to_f32() + 0.5))
            .collect();
        let (qd, kd, vd, gd, bd) = (
            to_device(&q, st),
            to_device(&k, st),
            to_device(&v, st),
            f32_to_device(&g, st),
            to_device(&beta, st),
        );
        let floats = unsafe { (api().cs1_gdn_workspace_floats)(t as i32, h as i32) };
        let ws0 = DeviceBuffer::new(floats * 4).unwrap();
        let o_full = DeviceBuffer::new(t * h * d * 2).unwrap();
        // SAFETY: every buffer holds t rows of the widths above; workspace its size.
        unsafe {
            check(
                (api().cs1_gdn_prefill)(
                    qd.at(0),
                    kd.at(0),
                    vd.at(0),
                    gd.at(0).cast(),
                    bd.at(0),
                    o_full.at(0),
                    ws0.at(0).cast(),
                    t as i32,
                    h as i32,
                    hk as i32,
                    (d as f32).powf(-0.5),
                    st,
                ),
                "one-shot prefill",
            )
            .unwrap();
        }
        // phase 1 over rows [0, p) with the float32 state dump, phase 2 continues.
        let p = ((t as f64 * 0.6) as usize / 64 * 64).max(64);
        let state = DeviceBuffer::new(h * d * d * 4).unwrap();
        let o_p1 = DeviceBuffer::new(p * h * d * 2).unwrap();
        let ws1 =
            DeviceBuffer::new(unsafe { (api().cs1_gdn_workspace_floats)(p as i32, h as i32) } * 4)
                .unwrap();
        // SAFETY: the p-row prefix of the same inputs; the state buffer is [H][K][V] f32.
        unsafe {
            check(
                (api().cs1_gdn_prefill_x)(
                    qd.at(0),
                    kd.at(0),
                    vd.at(0),
                    gd.at(0).cast(),
                    bd.at(0),
                    o_p1.at(0),
                    ws1.at(0).cast(),
                    p as i32,
                    h as i32,
                    hk as i32,
                    (d as f32).powf(-0.5),
                    std::ptr::null(),
                    state.at(0).cast(),
                    st,
                ),
                "capture phase",
            )
            .unwrap();
        }
        let t2 = t - p;
        let o_p2 = DeviceBuffer::new(t2 * h * d * 2).unwrap();
        let ws2 =
            DeviceBuffer::new(unsafe { (api().cs1_gdn_workspace_floats)(t2 as i32, h as i32) } * 4)
                .unwrap();
        // SAFETY: the t2-row suffix of the same inputs; the captured state feeds it.
        unsafe {
            check(
                (api().cs1_gdn_prefill_x)(
                    qd.at(p * hk * d * 2),
                    kd.at(p * hk * d * 2),
                    vd.at(p * h * d * 2),
                    gd.at(p * h * 4).cast(),
                    bd.at(p * h * 2),
                    o_p2.at(0),
                    ws2.at(0).cast(),
                    t2 as i32,
                    h as i32,
                    hk as i32,
                    (d as f32).powf(-0.5),
                    state.at(0).cast_const(),
                    std::ptr::null_mut(),
                    st,
                ),
                "continuation phase",
            )
            .unwrap();
        }
        let full = from_device(&o_full, t * h * d, st);
        let first = from_device(&o_p1, p * h * d, st);
        let second = from_device(&o_p2, t2 * h * d, st);
        assert_eq!(first, full[..p * h * d], "capture rows diverge at t = {t}");
        assert_eq!(
            second,
            full[p * h * d..],
            "continuation rows diverge at t = {t}"
        );
        eprintln!("gdn two-phase t = {t}, split = {p}: both phases bitwise equal to one-shot");
        // capture-only, T = 0: the state passes through untouched.
        let passthru = DeviceBuffer::new(h * d * d * 4).unwrap();
        let empty = DeviceBuffer::new(4).unwrap();
        // SAFETY: T = 0 touches only the state buffers.
        unsafe {
            check(
                (api().cs1_gdn_prefill_x)(
                    empty.at(0),
                    empty.at(0),
                    empty.at(0),
                    empty.at(0).cast(),
                    empty.at(0),
                    empty.at(0),
                    empty.at(0).cast(),
                    0,
                    h as i32,
                    hk as i32,
                    1.0,
                    state.at(0).cast_const(),
                    passthru.at(0).cast(),
                    st,
                ),
                "state passthrough",
            )
            .unwrap();
        }
        assert_eq!(
            f32_from_device(&state, h * d * d, st),
            f32_from_device(&passthru, h * d * d, st),
            "T = 0 must copy the state through"
        );
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn fixed_gemm_rows_do_not_depend_on_m() {
    let st = setup();
    // Every projection of the 2B, 4B, 9B and 27B backbones as (N, K): the GDN input and
    // output, the attention input and output, and the MLP's gate|up and down.
    let shapes = [
        (8224usize, 2048usize),
        (2048, 2048),
        (5120, 2048),
        (12288, 2048),
        (2048, 6144),
        (12352, 2560),
        (2560, 4096),
        (10240, 2560),
        (18432, 2560),
        (2560, 9216),
        (12352, 4096),
        (4096, 4096),
        (10240, 4096),
        (24576, 4096),
        (4096, 12288),
        (16480, 5120),
        (5120, 6144),
        (14336, 5120),
        (34816, 5120),
        (5120, 17408),
    ];
    let max_m = 4096usize;
    // SAFETY: plain handle creation; checked for null below.
    let fixed = unsafe { (api().cs1_gemm_create_fixed)(32 << 20, 64) };
    assert!(!fixed.is_null());
    // SAFETY: a non-positive reference M is refused before anything is allocated.
    assert!(unsafe { (api().cs1_gemm_create_fixed)(32 << 20, 0) }.is_null());
    let nan = vec![0xffu8; max_m * 34816 * 2];
    for (i, (n, k)) in shapes.into_iter().enumerate() {
        let x = random(max_m * k, 70 + i as u64, 2.0);
        let w = random(n * k, 90 + i as u64, 0.05);
        let (xd, wd) = (to_device(&x, st), to_device(&w, st));
        let y = DeviceBuffer::new(max_m * n * 2).unwrap();
        // the bfloat16 output of m rows starting at input row `at`, over a NaN-filled buffer
        let rows = |at: usize, m: usize| {
            let mut out = vec![0u8; m * n * 2];
            // SAFETY: x holds max_m rows of k, w is [n, k], y has room for max_m rows of n,
            // and at + m <= max_m.
            unsafe {
                cuda::upload(y.at(0), &nan[..max_m * n * 2], st).unwrap();
                check(
                    (api().cs1_gemm)(
                        fixed,
                        xd.at(at * k * 2),
                        wd.at(0),
                        y.at(0),
                        m as i32,
                        n as i32,
                        k as i32,
                        n as i32,
                        st,
                    ),
                    "fixed gemm",
                )
                .unwrap();
                cuda::download(&mut out, y.at(0), st).unwrap();
            }
            out
        };
        let all = rows(0, max_m);
        // every output was written (the comparisons below are byte for byte against this)
        let (values, _) = all.as_chunks::<2>();
        assert!(values.iter().all(|b| !bf16::from_le_bytes(*b).is_nan()));
        let row = |r: usize| &all[r * n * 2..(r + 1) * n * 2];
        // fewer rows, and rows that sit lower in the full call, as a branch's tokens do
        for (at, m) in [
            (0usize, 1usize),
            (0, 2),
            (0, 3),
            (0, 8),
            (0, 63),
            (0, 64),
            (0, 65),
            (0, 256),
            (0, 1000),
            (0, 1024),
            (0, 3109),
            (1, 2),
            (37, 64),
            (64, 65),
            (1000, 200),
            (3000, 1096),
        ] {
            assert!(
                rows(at, m) == all[at * n * 2..(at + m) * n * 2],
                "rows {at}..{} changed at {n} x {k}",
                at + m
            );
        }
        // the first and last rows against float64, so the product itself is right
        for r in [0, max_m - 1] {
            let want: Vec<f64> = (0..n)
                .map(|j| {
                    (0..k)
                        .map(|c| x[r * k + c].to_f64() * w[j * k + c].to_f64())
                        .sum()
                })
                .collect();
            let (got, _) = row(r).as_chunks::<2>();
            let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
            let worst = got
                .iter()
                .zip(&want)
                .map(|(b, v)| (bf16::from_le_bytes(*b).to_f64() - v).abs())
                .fold(0f64, f64::max);
            assert!(
                worst <= 1e-2 * scale,
                "{n} x {k}, row {r}: {worst} vs {scale}"
            );
        }
    }
    // SAFETY: created above and not destroyed before.
    unsafe { (api().cs1_gemm_destroy)(fixed) };
}
