use omni_laya::{
    packing::pack,
    preprocess::{Prepared, Question},
};
use serde_json::Value;

fn prepared(n: usize) -> Prepared {
    Prepared {
        questions: (0..n)
            .map(|i| Question {
                id: i.to_string(),
                kind: "noul".into(),
                criteria: Value::Null,
                ids: vec![1, 2, 3],
                markers: vec![1, 2],
                qtype: 2,
            })
            .collect(),
        usage: n * 3,
    }
}

#[test]
fn padding_preserves_order_lengths_and_markers() {
    let prepared = prepared(3);
    let batch = pack(&prepared).unwrap();
    assert_eq!((batch.b, batch.l), (4, 16));
    assert_eq!(batch.lens, [3, 3, 3, 0]);
    assert_eq!(batch.qtypes, [2, 2, 2, 0]);
    assert_eq!(batch.markers, vec![vec![1, 2]; 3]);
    for i in 0..3 {
        assert_eq!(&batch.input_ids[i * 16..i * 16 + 3], &[1, 2, 3]);
    }
    assert!(batch.input_ids[48..].iter().all(|id| *id == 0));
    assert_eq!(prepared.usage, 9);
}

#[test]
fn rejects_executor_capacity_and_invalid_markers() {
    assert!(pack(&prepared(17)).is_err());
    let mut input = prepared(1);
    input.questions[0].markers.push(3);
    assert!(pack(&input).is_err());
    input.questions[0].markers = vec![1; 2049];
    assert!(pack(&input).is_err());
}

#[test]
fn empty_request_needs_no_device_rows() {
    let batch = pack(&prepared(0)).unwrap();
    assert_eq!((batch.b, batch.l), (0, 0));
    assert!(batch.markers.is_empty());
}

#[test]
fn padding_matches_native_sequence_alignment() {
    let mut input = prepared(3);
    input.questions[1].ids = vec![7; 257];
    let batch = pack(&input).unwrap();
    assert_eq!((batch.b, batch.l), (4, 320));
    assert_eq!(batch.lens, [3, 257, 3, 0]);
    assert!(batch.input_ids[3..257].iter().all(|id| *id == 50283));
    assert!(batch.input_ids[257..320].iter().all(|id| *id == 0));
    assert!(batch.input_ids[960..].iter().all(|id| *id == 0));
}
