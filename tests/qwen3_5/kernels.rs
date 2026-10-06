//! GPU checks of the attention and Gated DeltaNet kernels on random inputs. They need
//! a GPU and CUA_S1_CUDA_LIB pointing at libqwen3_5_cuda.so, so they only run when
//! asked for:
//!
//!     CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
//!       cargo test --release -p omni-qwen3-5-native --test kernels -- --ignored

use std::ffi::c_void;
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

fn f32_from_device(buf: &DeviceBuffer, n: usize, st: Stream) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    // SAFETY: the buffer holds n float32 values.
    unsafe { cuda::download(&mut bytes, buf.at(0), st).unwrap() };
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
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
/// with the L2 norms of q and k and q scaled by K^-1/2. Also returns the state
/// [h, K, V] after each count of tokens in `states_at`.
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
    states_at: &[usize],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut out = vec![0f64; t * h * d];
    let mut states = vec![vec![0f64; h * d * d]; states_at.len()];
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
            for (at, state) in states_at.iter().zip(&mut states) {
                if *at == tok + 1 {
                    state[head * d * d..][..d * d].copy_from_slice(&s);
                }
            }
        }
    }
    (out, states)
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
        let (want, _) = gated_delta_reference(&q, &k, &v, &g, &beta, t, h, hk, d, &[]);
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

/// Prefix lengths around and inside 64-token chunks, and the branch lengths after them.
const PREFIXES: [usize; 12] = [1, 2, 3, 31, 63, 64, 65, 127, 128, 129, 1000, 1024];
const BRANCHES: [usize; 5] = [1, 2, 64, 65, 200];

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn conv_with_history_matches_unsplit_conv() {
    let st = setup();
    // 4B/9B (16 key heads, 32 value heads) and 27B (48 value heads); each GDN input
    // row also carries z, b and a after the conv channels.
    for (key_dim, value_dim, heads) in [(2048usize, 4096usize, 32usize), (2048, 6144, 48)] {
        let channels = 2 * key_dim + value_dim;
        let ld = channels + value_dim + 2 * heads;
        let total = 1300usize;
        let input = random(total * ld, 31, 4.0);
        let weights = random(channels * 4, 32, 0.5);
        let (qkv, w) = (to_device(&input, st), to_device(&weights, st));
        // q|k|v of rows [at, at + t), each token's channels in order
        let conv = |at: usize, t: usize, history: *const c_void, history_out: *mut c_void| {
            let q = DeviceBuffer::new(t * key_dim * 2).unwrap();
            let k = DeviceBuffer::new(t * key_dim * 2).unwrap();
            let v = DeviceBuffer::new(t * value_dim * 2).unwrap();
            // SAFETY: qkv holds rows [at, at + t) of width ld, the outputs t rows each,
            // and history/history_out are null or 3 rows of the conv channels.
            unsafe {
                check(
                    (api().cs1_gdn_conv_history)(
                        qkv.at(at * ld * 2),
                        ld as i32,
                        w.at(0),
                        history,
                        history_out,
                        q.at(0),
                        k.at(0),
                        v.at(0),
                        t as i32,
                        key_dim as i32,
                        value_dim as i32,
                        st,
                    ),
                    "conv with history",
                )
                .unwrap();
            }
            let (q, k, v) = (
                from_device(&q, t * key_dim, st),
                from_device(&k, t * key_dim, st),
                from_device(&v, t * value_dim, st),
            );
            (0..t)
                .flat_map(|r| {
                    [
                        &q[r * key_dim..][..key_dim],
                        &k[r * key_dim..][..key_dim],
                        &v[r * value_dim..][..value_dim],
                    ]
                    .concat()
                })
                .collect::<Vec<f32>>()
        };
        // the conv inputs of the three positions before `end`, zeros before the start
        let source = &input;
        let history_before = |end: usize| -> Vec<f32> {
            (0..3)
                .flat_map(|r| {
                    let s = end as isize - 3 + r;
                    (0..channels).map(move |c| {
                        if s < 0 {
                            0.0
                        } else {
                            source[s as usize * ld + c].to_f32()
                        }
                    })
                })
                .collect()
        };
        let full = conv(0, total, std::ptr::null(), std::ptr::null_mut());
        let rows = |a: usize, b: usize| bits(&full[a * channels..b * channels]);
        let null = std::ptr::null();
        for p in PREFIXES {
            let history = DeviceBuffer::new(3 * channels * 2).unwrap();
            assert_eq!(bits(&conv(0, p, null, history.at(0))), rows(0, p));
            assert_eq!(
                from_device(&history, 3 * channels, st),
                history_before(p),
                "history after {p}"
            );
            for b in BRANCHES {
                let branch = conv(p, b, history.at(0), std::ptr::null_mut());
                assert_eq!(bits(&branch), rows(p, p + b), "split {p} + {b}");
            }
        }
        // request prefix, question prefix, then a branch; also chains shorter than the kernel
        for (p1, p2, b) in [
            (1usize, 2usize, 1usize),
            (2, 3, 2),
            (900, 960, 65),
            (64, 65, 1),
        ] {
            let (h1, h2) = (
                DeviceBuffer::new(3 * channels * 2).unwrap(),
                DeviceBuffer::new(3 * channels * 2).unwrap(),
            );
            conv(0, p1, null, h1.at(0));
            assert_eq!(bits(&conv(p1, p2 - p1, h1.at(0), h2.at(0))), rows(p1, p2));
            assert_eq!(from_device(&h2, 3 * channels, st), history_before(p2));
            assert_eq!(
                bits(&conv(p2, b, h2.at(0), std::ptr::null_mut())),
                rows(p2, p2 + b),
                "chain {p1}, {p2}, {b}"
            );
        }
        // no tokens: history_out is the history itself, shifted by nothing
        let (h1, h2) = (
            DeviceBuffer::new(3 * channels * 2).unwrap(),
            DeviceBuffer::new(3 * channels * 2).unwrap(),
        );
        conv(0, 5, null, h1.at(0));
        conv(5, 0, h1.at(0), h2.at(0));
        assert_eq!(from_device(&h2, 3 * channels, st), history_before(5));
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn gated_delta_state_continues_unsplit_prefill() {
    let st = setup();
    let d = 128usize;
    // 4B/9B (32 value heads over 16 key heads) and 27B (48 over 16)
    for (h, hk) in [(32usize, 16usize), (48, 16)] {
        let total = 1300usize;
        let k = random(total * hk * d, 42, 1.0);
        let q: Vec<bf16> = k
            .iter()
            .zip(random(total * hk * d, 41, 1.0))
            .map(|(k, n)| bf16::from_f32(0.8 * k.to_f32() + 0.2 * n.to_f32()))
            .collect();
        let v = random(total * h * d, 43, 1.0);
        // slow decays, so that the state carries far across the split
        let g: Vec<f32> = random(total * h, 44, 1.0)
            .iter()
            .map(|x| (x.to_f32() - 1.0) * 0.01)
            .collect();
        let beta: Vec<bf16> = random(total * h, 45, 0.5)
            .iter()
            .map(|x| bf16::from_f32(x.to_f32() + 0.5))
            .collect();
        // reference states after each prefix and after each question prefix below
        let marks: Vec<usize> = PREFIXES.iter().copied().chain([192, 1100]).collect();
        let (want, ref_states) =
            gated_delta_reference(&q, &k, &v, &g, &beta, total, h, hk, d, &marks);
        let ref_state = |at: usize| &ref_states[marks.iter().position(|&m| m == at).unwrap()];
        let (qd, kd, vd, gd, bd) = (
            to_device(&q, st),
            to_device(&k, st),
            to_device(&v, st),
            f32_to_device(&g, st),
            to_device(&beta, st),
        );
        // SAFETY: pure function of its arguments.
        let floats = unsafe { (api().cs1_gdn_workspace_floats)(total as i32, h as i32) };
        let ws = DeviceBuffer::new(floats * 4).unwrap();
        let state_floats = h * d * d;
        let run = |at: usize, t: usize, initial: *const f32, last: *mut f32| {
            let o = DeviceBuffer::new((t * h * d).max(1) * 2).unwrap();
            // SAFETY: the inputs hold rows [at, at + t), the workspace fits the longest
            // run, and the states are null or [h, 128, 128] floats.
            unsafe {
                check(
                    (api().cs1_gdn_prefill_state)(
                        qd.at(at * hk * d * 2),
                        kd.at(at * hk * d * 2),
                        vd.at(at * h * d * 2),
                        gd.at(at * h * 4).cast::<f32>(),
                        bd.at(at * h * 2),
                        o.at(0),
                        ws.at(0).cast::<f32>(),
                        initial,
                        last,
                        t as i32,
                        h as i32,
                        hk as i32,
                        (d as f32).powf(-0.5),
                        st,
                    ),
                    "gdn prefill with state",
                )
                .unwrap();
            }
            from_device(&o, t * h * d, st)
        };
        let null = std::ptr::null();
        let full = run(0, total, null, std::ptr::null_mut());
        let rows = |a: usize, b: usize| &full[a * h * d..b * h * d];
        let scale_of = |a: usize, b: usize| {
            want[a * h * d..b * h * d]
                .iter()
                .fold(0f64, |m, x| m.max(x.abs()))
        };
        let worst = |got: &[f32], want: &[f64]| {
            got.iter()
                .zip(want)
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0f64, f64::max)
        };
        for (i, p) in PREFIXES.into_iter().enumerate() {
            let state = DeviceBuffer::new(state_floats * 4).unwrap();
            let prefix = run(0, p, null, state.at(0).cast());
            assert_eq!(
                bits(&prefix),
                bits(rows(0, p)),
                "writing the state changed the output, p = {p}"
            );
            let s = f32_from_device(&state, state_floats, st);
            let s_scale = ref_states[i].iter().fold(0f64, |m, x| m.max(x.abs()));
            let s_worst = worst(&s, ref_state(p));
            eprintln!(
                "gated delta state after {p} ({h} heads): largest difference {s_worst:.2e}, largest |reference| {s_scale:.2e}"
            );
            assert!(
                s_worst <= 2e-2 * s_scale,
                "state after {p}: {s_worst} vs {s_scale}"
            );
            for b in BRANCHES {
                let branch = run(p, b, state.at(0).cast(), std::ptr::null_mut());
                assert!(branch.iter().all(|x| x.is_finite()));
                if p % 64 == 0 {
                    assert_eq!(
                        bits(&branch),
                        bits(rows(p, p + b)),
                        "aligned split {p} + {b}"
                    );
                    continue;
                }
                let scale = scale_of(p, p + b);
                let to_ref = worst(&branch, &want[p * h * d..(p + b) * h * d]);
                let to_unsplit = branch
                    .iter()
                    .zip(rows(p, p + b))
                    .map(|(a, b)| (a - b).abs() as f64)
                    .fold(0f64, f64::max);
                eprintln!(
                    "gated delta split {p} + {b} ({h} heads): to float64 {to_ref:.2e}, to unsplit {to_unsplit:.2e}, largest |reference| {scale:.2e}"
                );
                assert!(
                    to_ref <= 2e-2 * scale,
                    "split {p} + {b}: {to_ref} vs {scale}"
                );
            }
        }
        // request prefix -> question prefix -> branch, also with the question state
        // updated in place
        for (p1, p2, b) in [
            (64usize, 128usize, 65usize),
            (100, 192, 65),
            (1000, 1100, 100),
        ] {
            let (s1, s2, s1b) = (
                DeviceBuffer::new(state_floats * 4).unwrap(),
                DeviceBuffer::new(state_floats * 4).unwrap(),
                DeviceBuffer::new(state_floats * 4).unwrap(),
            );
            run(0, p1, null, s1.at(0).cast());
            run(0, p1, null, s1b.at(0).cast());
            let question = run(p1, p2 - p1, s1.at(0).cast(), s2.at(0).cast());
            let in_place = run(p1, p2 - p1, s1b.at(0).cast(), s1b.at(0).cast());
            assert_eq!(bits(&question), bits(&in_place));
            assert_eq!(
                bits(&f32_from_device(&s2, state_floats, st)),
                bits(&f32_from_device(&s1b, state_floats, st)),
                "in-place state {p1}, {p2}"
            );
            let branch = run(p2, b, s2.at(0).cast(), std::ptr::null_mut());
            if p1 % 64 == 0 && p2 % 64 == 0 {
                assert_eq!(bits(&question), bits(rows(p1, p2)));
                assert_eq!(
                    bits(&branch),
                    bits(rows(p2, p2 + b)),
                    "aligned chain {p1}, {p2}"
                );
            } else {
                let q_ref = worst(&question, &want[p1 * h * d..p2 * h * d]);
                assert!(
                    q_ref <= 2e-2 * scale_of(p1, p2),
                    "question {p1}, {p2}: {q_ref}"
                );
                let s2_ref = ref_state(p2);
                let s2_worst = worst(&f32_from_device(&s2, state_floats, st), s2_ref);
                let s2_scale = s2_ref.iter().fold(0f64, |m, x| m.max(x.abs()));
                assert!(s2_worst <= 2e-2 * s2_scale, "state {p1}, {p2}: {s2_worst}");
                let scale = scale_of(p2, p2 + b);
                let to_ref = worst(&branch, &want[p2 * h * d..(p2 + b) * h * d]);
                eprintln!(
                    "gated delta chain {p1}, {p2} + {b} ({h} heads): question to float64 {q_ref:.2e}, state {s2_worst:.2e}, branch {to_ref:.2e}"
                );
                assert!(to_ref <= 2e-2 * scale, "chain {p1}, {p2} + {b}");
            }
        }
        // no tokens: the final state is the initial one, or zeros without one
        let (s1, s2) = (
            DeviceBuffer::new(state_floats * 4).unwrap(),
            DeviceBuffer::new(state_floats * 4).unwrap(),
        );
        run(0, 200, null, s1.at(0).cast());
        run(200, 0, s1.at(0).cast(), s2.at(0).cast());
        assert_eq!(
            bits(&f32_from_device(&s2, state_floats, st)),
            bits(&f32_from_device(&s1, state_floats, st))
        );
        run(200, 0, null, s2.at(0).cast());
        assert!(
            f32_from_device(&s2, state_floats, st)
                .iter()
                .all(|x| x.to_bits() == 0)
        );
        // a misaligned state is rejected before any work is queued
        for (initial, last) in [
            (s1.at(4).cast::<f32>().cast_const(), std::ptr::null_mut()),
            (std::ptr::null(), s2.at(4).cast::<f32>()),
        ] {
            // SAFETY: the call returns before launching anything.
            let code = unsafe {
                (api().cs1_gdn_prefill_state)(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    initial,
                    last,
                    1,
                    h as i32,
                    hk as i32,
                    1.0,
                    st,
                )
            };
            assert_ne!(code, 0);
        }
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn cached_attention_matches_unsplit_attention() {
    let st = setup();
    let dh = 256usize;
    // 4B/9B (16 query heads) and 27B (24) over 4 KV heads
    for (hq, hk) in [(16usize, 4usize), (24, 4)] {
        let total = 1300usize;
        let ldv = hk * dh + 16; // a strided V buffer, as read from the projection output
        let q = to_device(&random(total * hq * dh, 51, 2.0), st);
        let k = to_device(&random(total * hk * dh, 52, 2.0), st);
        let v = to_device(&random(total * ldv, 53, 1.0), st);
        let gate = to_device(&random(total * hq * dh, 54, 12.0), st);
        let row = hq * dh;
        let attend = |at: usize, tq: usize, tk: usize| {
            let out = DeviceBuffer::new((tq * row).max(1) * 2).unwrap();
            // SAFETY: q and gate hold rows [at, at + tq), k and v the first tk rows.
            unsafe {
                check(
                    (api().cs1_attention_gated_cached)(
                        q.at(at * row * 2),
                        k.at(0),
                        v.at(0),
                        ldv as i32,
                        gate.at(at * row * 2),
                        out.at(0),
                        tq as i32,
                        tk as i32,
                        hq as i32,
                        hk as i32,
                        dh as i32,
                        0.0625,
                        st,
                    ),
                    "cached attention",
                )
                .unwrap();
            }
            from_device(&out, tq * row, st)
        };
        // with every position as a query, the cached call runs the same code as the
        // unsplit one; the check below only guards the two entry points
        let full = attend(0, total, total);
        let unsplit = DeviceBuffer::new(total * row * 2).unwrap();
        // SAFETY: as above, with total rows everywhere.
        unsafe {
            check(
                (api().cs1_attention_gated)(
                    q.at(0),
                    k.at(0),
                    v.at(0),
                    ldv as i32,
                    gate.at(0),
                    unsplit.at(0),
                    total as i32,
                    hq as i32,
                    hk as i32,
                    dh as i32,
                    0.0625,
                    st,
                ),
                "gated attention",
            )
            .unwrap();
        }
        assert_eq!(bits(&full), bits(&from_device(&unsplit, total * row, st)));
        for p in PREFIXES {
            for b in BRANCHES {
                let got = attend(p, b, p + b);
                assert!(got.iter().all(|x| x.is_finite()));
                assert_eq!(
                    bits(&got),
                    bits(&full[p * row..(p + b) * row]),
                    "split {p} + {b}, {hq} heads"
                );
            }
        }
        // no queries is valid; more queries than keys is not
        // SAFETY: both calls return before launching anything.
        unsafe {
            let null = std::ptr::null();
            let out = std::ptr::null_mut();
            let call = |tq: i32, tk: i32| {
                (api().cs1_attention_gated_cached)(
                    null, null, null, 1024, null, out, tq, tk, 16, 4, 256, 0.0625, st,
                )
            };
            assert_eq!(call(0, 5), 0);
            assert_ne!(call(2, 1), 0);
        }
    }
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB"]
fn copy_rows_copies_pitched_rows() {
    let st = setup();
    let (rows, width, src_pitch, dst_pitch) = (37usize, 1024usize, 10240usize, 2048usize);
    let src = random(rows * src_pitch / 2, 61, 1.0);
    let source = to_device(&src, st);
    let target = to_device(&vec![bf16::from_f32(7.0); rows * dst_pitch / 2], st);
    // SAFETY: both buffers hold `rows` rows of their pitch, each wider than `width` bytes.
    unsafe {
        check(
            (api().cs1_copy_rows)(
                target.at(0),
                dst_pitch,
                source.at(0),
                src_pitch,
                width,
                rows as i32,
                st,
            ),
            "copy rows",
        )
        .unwrap();
    }
    let got = from_device(&target, rows * dst_pitch / 2, st);
    for r in 0..rows {
        for i in 0..dst_pitch / 2 {
            let want = if i < width / 2 {
                src[r * src_pitch / 2 + i].to_f32()
            } else {
                7.0
            };
            assert_eq!(got[r * dst_pitch / 2 + i], want, "row {r}, element {i}");
        }
    }
}
