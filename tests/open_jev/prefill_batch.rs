//! Full-checkpoint packing, sequence isolation and graph-shape regression check.
//! Run inside a GPU reservation with OPEN_JEV_MODEL and OPEN_JEV_CUDA_LIB set.

use std::path::PathBuf;

use omni_open_jev_native::engine::Engine;
use serde_json::json;

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs the pinned Open-Jev export, CUDA library and a GPU reservation"]
async fn packed_candidates_preserve_isolation_order_and_graph_shapes() {
    let model = PathBuf::from(std::env::var_os("OPEN_JEV_MODEL").unwrap());
    let library = PathBuf::from(std::env::var_os("OPEN_JEV_CUDA_LIB").unwrap());
    let engine = Engine::load(&model, &library).await.unwrap();
    let request = json!({
        "state": "Refunds require a receipt. This customer has no receipt.",
        "questions": {
            "short": {"type": "noul", "instructions": "Is a refund permitted?"},
            "long": {"type": "noul", "instructions": format!("{}Is a refund permitted?", "Use only the stated policy. ".repeat(20))},
            "other": {"type": "noul", "instructions": "Does the customer have the required receipt?"}
        }
    });
    let prepared = engine
        .processor
        .prepare(&serde_json::to_vec(&request).unwrap())
        .unwrap();
    let inputs = prepared.inputs;
    assert_ne!(inputs[0][0].len(), inputs[1][0].len());
    let mut independent = Vec::new();
    for candidate in &inputs {
        independent.push(
            engine
                .executor
                .execute(&engine.scheduler, vec![candidate.clone()])
                .await
                .unwrap()[0][0],
        );
    }
    for order in [
        vec![0, 1, 2],
        vec![1, 0, 2],
        vec![2, 1, 0],
        vec![0, 1, 2],
        (0..17).map(|index| index % 3).collect(),
        vec![0, 1, 2],
        vec![0, 1, 2],
    ] {
        let grouped = order.iter().map(|&index| inputs[index].clone()).collect();
        let packed = engine
            .executor
            .execute(&engine.scheduler, grouped)
            .await
            .unwrap();
        assert_eq!(packed.len(), order.len());
        for (row, index) in packed.iter().zip(order) {
            assert_eq!(row.len(), 1);
            // Probability gates in the HTTP benchmark are stricter and cover
            // complete typed outputs. This logit check detects sequence leakage.
            assert!(
                (row[0] - independent[index]).abs() <= 0.01,
                "candidate {index}: packed={} independent={}",
                row[0],
                independent[index]
            );
        }
    }
    // A later singleton must not retain a packed sequence's recurrent state.
    let again = engine
        .executor
        .execute(&engine.scheduler, vec![inputs[0].clone()])
        .await
        .unwrap();
    assert_eq!(again[0][0], independent[0]);

    // Reuse both singleton and packed graph shapes with changed token IDs.
    // The new embeddings must replace the residual left by the previous replay.
    let mut changed = inputs.clone();
    changed[0][0].fill(42);
    let expected = engine
        .executor
        .execute(&engine.scheduler, vec![changed[0].clone()])
        .await
        .unwrap()[0][0];
    assert_ne!(expected, independent[0]);
    let packed = engine
        .executor
        .execute(&engine.scheduler, changed)
        .await
        .unwrap();
    assert_eq!(packed.len(), 3);
    for (row, expected) in packed
        .iter()
        .zip([expected, independent[1], independent[2]])
    {
        assert_eq!(row.len(), 1);
        assert!((row[0] - expected).abs() <= 0.01);
    }
}
