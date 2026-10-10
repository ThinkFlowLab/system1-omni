//! Frozen numeric pins for the verbalizer readout math (constants computed with
//! CPython over the same f64 formula against this test's pattern).

use super::{LabelHead, Readout};

fn head() -> LabelHead {
    // rows[3][64]; row t element i = ((t*64+i)*5 % 11 - 5) / 7.
    let mut rows = Vec::with_capacity(3 * 64);
    for t in 0..3u32 {
        for i in 0..64u32 {
            rows.push((((t * 64 + i) * 5 % 11) as i32 - 5) as f32 / 7.0);
        }
    }
    LabelHead {
        width: 64,
        rows,
        index: [(15, 0), (16, 1), (17, 2)].into_iter().collect(),
    }
}

#[test]
fn probabilities_match_python_f64() {
    let hidden: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) - 6) as f32 / 9.0).collect();
    let readout = Readout {
        token_ids: vec![15, 16, 17],
        bias: vec![0.001, -0.002, 0.0005],
        temperature: 1.0143134751376188,
    };
    let p = head().probabilities(&hidden, &readout).unwrap();
    let expected = [0.6256388379121548, 0.1869499071806715, 0.18741125490717364];
    // 1e-12: f32 division rounding order differs between CPython and Rust patterns;
    // both far below the f32 head precision, so this pins the formula, not ulps.
    for (&a, &e) in p.iter().zip(&expected) {
        assert!((a - e).abs() < 1e-12, "{a} != {e}");
    }
}

#[test]
fn softmax_is_shift_invariant_to_minus_logz() {
    // The official readout subtracts per-request -logZ before bias/T; the stable
    // softmax must be invariant, which is why logits suffice.
    let hidden: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) - 6) as f32 / 9.0).collect();
    let readout = |shift: f64| Readout {
        token_ids: vec![15, 16, 17],
        bias: vec![0.001 + shift, -0.002 + shift, 0.0005 + shift],
        temperature: 1.0143134751376188,
    };
    let p0 = head().probabilities(&hidden, &readout(0.0)).unwrap();
    let p1 = head().probabilities(&hidden, &readout(-42.5)).unwrap();
    for (&a, &b) in p0.iter().zip(&p1) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
}

#[test]
fn missing_exported_label_is_rejected() {
    let mut h = head();
    h.index.remove(&17);
    let readout = Readout {
        token_ids: vec![15, 16, 17],
        bias: vec![0.0; 3],
        temperature: 1.0,
    };
    let hidden = vec![0.0f32; 64];
    assert!(h.probabilities(&hidden, &readout).is_err());
}
