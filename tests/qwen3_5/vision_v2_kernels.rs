//! Direct 27B vision operators against host references, without model weights.
use super::*;

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB with vision v2 symbols"]
fn position_1152_preserves_interpolation_rounding() {
    let st = setup();
    let (rows, hidden) = (5usize, 1152usize);
    let original = random(rows * hidden, 101, 0.5);
    let table = random(6 * hidden, 102, 0.25);
    let indices: Vec<i32> = (0..rows)
        .flat_map(|row| (0..4).map(move |corner| ((row + corner) % 6) as i32))
        .collect();
    let weights: Vec<f32> = (0..rows).flat_map(|_| [0.125, 0.25, 0.5, 0.125]).collect();
    let input = to_device(&original, st);
    let positions = to_device(&table, st);
    let bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let index = DeviceBuffer::new(bytes.len()).unwrap();
    let weight = f32_to_device(&weights, st);
    // SAFETY: all four indices per row address complete 1152-wide table rows.
    unsafe {
        cuda::upload(index.at(0), &bytes, st).unwrap();
        check(
            (api().vision_v2().unwrap().position)(
                input.at(0),
                positions.at(0),
                index.at(0).cast(),
                weight.at(0).cast(),
                rows as i32,
                hidden as i32,
                st,
            ),
            "27B vision position",
        )
        .unwrap();
    }
    let got = from_device(&input, rows * hidden, st);
    for row in 0..rows {
        for col in 0..hidden {
            let mut value = 0.0f32;
            for corner in 0..4 {
                value += table[indices[row * 4 + corner] as usize * hidden + col].to_f32()
                    * weights[row * 4 + corner];
            }
            let want = bf16::from_f32(
                original[row * hidden + col].to_f32() + bf16::from_f32(value).to_f32(),
            )
            .to_f32();
            assert_eq!(got[row * hidden + col], want, "row={row}, col={col}");
        }
    }
    // SAFETY: synchronized download completed every operation on this stream.
    unsafe { (api().cs1_stream_destroy)(st) };
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB with vision v2 symbols"]
fn rope_72_preserves_head_pairs_and_interleaved_qkv() {
    let st = setup();
    let (hidden, head_dim) = (1152usize, 72usize);
    for rows in [3usize, 65] {
        let qkv = random(rows * hidden * 3, rows as u64, 0.75);
        // Dyadic angles make the separately materialized FP32 products exact.
        let co: Vec<f32> = (0..rows * head_dim / 2)
            .map(|i| if i % 3 == 0 { -0.5 } else { 0.75 })
            .collect();
        let si: Vec<f32> = (0..co.len())
            .map(|i| if i % 5 == 0 { 0.25 } else { -0.5 })
            .collect();
        let input = to_device(&qkv, st);
        let cos = f32_to_device(&co, st);
        let sin = f32_to_device(&si, st);
        let q = DeviceBuffer::new(rows * hidden * 2).unwrap();
        let k = DeviceBuffer::new(rows * hidden * 2).unwrap();
        // SAFETY: compact outputs, 3*hidden interleaved input and 36 angles per row.
        unsafe {
            check(
                (api().vision_v2().unwrap().rope)(
                    input.at(0),
                    cos.at(0).cast(),
                    sin.at(0).cast(),
                    q.at(0),
                    k.at(0),
                    rows as i32,
                    hidden as i32,
                    head_dim as i32,
                    st,
                ),
                "27B vision rotary",
            )
            .unwrap();
        }
        for (part, output) in [(0usize, &q), (1, &k)] {
            let got = from_device(output, rows * hidden, st);
            for row in 0..rows {
                for head in 0..16 {
                    for channel in 0..head_dim {
                        let half = head_dim / 2;
                        let partner = (channel + half) % head_dim;
                        let start = (row * 3 + part) * hidden + head * head_dim;
                        let angle = row * half + channel % half;
                        let sign = if channel < half { -1.0 } else { 1.0 };
                        let want = bf16::from_f32(
                            qkv[start + channel].to_f32() * co[angle]
                                + sign * qkv[start + partner].to_f32() * si[angle],
                        )
                        .to_f32();
                        assert_eq!(
                            got[row * hidden + head * head_dim + channel],
                            want,
                            "rows={rows}, row={row}, head={head}, channel={channel}, part={part}"
                        );
                    }
                }
            }
        }
    }
    // SAFETY: synchronized downloads completed every operation on this stream.
    unsafe { (api().cs1_stream_destroy)(st) };
}

#[test]
#[ignore = "needs a GPU and CUA_S1_CUDA_LIB with vision v2 symbols"]
fn attention_72_matches_float64_with_strided_values_and_dirty_workspace() {
    let st = setup();
    let (heads, head_dim) = (16usize, 72usize);
    let hidden = heads * head_dim;
    for rows in [1usize, 7, 65] {
        let workspace_bytes = rows * heads * 80 * 4 * 2;
        let workspace = DeviceBuffer::new(workspace_bytes).unwrap();
        let output = DeviceBuffer::new(rows * hidden * 2).unwrap();
        for seed in [201u64, 301] {
            let q = random(rows * hidden, seed, 0.6);
            let k = random(rows * hidden, seed + 1, 0.6);
            let qkv = random(rows * hidden * 3, seed + 2, 0.8);
            let dq = to_device(&q, st);
            let dk = to_device(&k, st);
            let dv = to_device(&qkv, st);
            // Poison all padded columns; a missing zero-fill must not pass on fresh memory.
            unsafe { cuda::upload(workspace.at(0), &vec![0xff; workspace_bytes], st).unwrap() };
            // SAFETY: V begins at the third projection, with stride 3*hidden;
            // workspace contains four padded [rows,16,80] BF16 matrices.
            unsafe {
                check(
                    (api().vision_v2().unwrap().attention)(
                        dq.at(0),
                        dk.at(0),
                        dv.at(2 * hidden * 2),
                        output.at(0),
                        rows as i32,
                        heads as i32,
                        head_dim as i32,
                        workspace.at(0),
                        st,
                    ),
                    "27B vision attention",
                )
                .unwrap();
            }
            let got = from_device(&output, rows * hidden, st);
            for row in 0..rows {
                for head in 0..heads {
                    let scores: Vec<f64> = (0..rows)
                        .map(|key| {
                            (0..head_dim)
                                .map(|d| {
                                    q[(row * heads + head) * head_dim + d].to_f64()
                                        * k[(key * heads + head) * head_dim + d].to_f64()
                                })
                                .sum::<f64>()
                                / (head_dim as f64).sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|v| (v - max).exp()).collect();
                    let sum: f64 = weights.iter().sum();
                    for d in 0..head_dim {
                        let want: f64 = weights
                            .iter()
                            .enumerate()
                            .map(|(key, weight)| {
                                weight * qkv[(key * 3 + 2) * hidden + head * head_dim + d].to_f64()
                            })
                            .sum::<f64>()
                            / sum;
                        let actual = got[(row * heads + head) * head_dim + d] as f64;
                        assert!(
                            actual.is_finite() && (actual - want).abs() <= 0.004,
                            "rows={rows}, row={row}, head={head}, d={d}, seed={seed}: {actual} != {want}"
                        );
                    }
                }
            }
        }
    }
    // SAFETY: synchronized downloads completed every operation on this stream.
    unsafe { (api().cs1_stream_destroy)(st) };
}
