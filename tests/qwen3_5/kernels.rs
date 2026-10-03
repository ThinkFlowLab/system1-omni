//! GPU checks of the attention and Gated DeltaNet kernels on random inputs. They need
//! a GPU and CUA_S1_CUDA_LIB pointing at libqwen3_5_cuda.so, so they only run when
//! asked for:
//!
//!     CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
//!       cargo test --release -p omni-qwen3-5-native --test kernels -- --ignored

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
    let (h, hk, d) = (4usize, 2usize, 128usize);
    // Exercise inverse tile/chunk boundaries, the short model request, and
    // accumulated state with weak decay over many chunks.
    for (t, decay_scale) in [
        (1usize, 1.0f32),
        (16, 1.0),
        (32, 1.0),
        (63, 1.0),
        (64, 1.0),
        (65, 1.0),
        (107, 1.0),
        (127, 1.0),
        (128, 1.0),
        (129, 1.0),
        (150, 1.0),
        (107, 0.01),
        (936, 0.01),
    ] {
        // q close to k, so that q.k and the outputs are of order one as in the model
        let k = random(t * hk * d, 12, 1.0);
        let q: Vec<bf16> = k
            .iter()
            .zip(random(t * hk * d, 11, 1.0))
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
        let scale = want.iter().fold(0f64, |m, x| m.max(x.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0f64, f64::max);
        eprintln!(
            "gated delta t = {t}: largest difference {worst:.2e}, largest |reference| {scale:.2}"
        );
        assert!(worst <= 2e-2 * scale, "t = {t}: {worst} vs scale {scale}");
    }
}
