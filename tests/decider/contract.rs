use omni_decider_native::{Limits, Processor};
use serde_json::{Value, json};

fn processor() -> Processor {
    Processor::load(
        std::env::var("DECIDER_MODEL").expect("set DECIDER_MODEL"),
        Limits::default(),
    )
    .unwrap()
}
#[test]
#[ignore = "requires the pinned tokenizer/config; CPU only, no tensor loading"]
fn pinned_tokens_rows_usage_and_all_labels() {
    let processor = processor();
    let golden: Value = serde_json::from_str(include_str!("data/cpu.json")).unwrap();
    let labels: Vec<u32> = serde_json::from_value(golden["label_ids"].clone()).unwrap();
    assert_eq!(
        processor
            .labels()
            .iter()
            .map(|label| label.id)
            .collect::<Vec<_>>(),
        labels
    );
    assert_eq!(processor.labels().last().unwrap().name, "JT");
    for fixture in golden["fixtures"].as_array().unwrap() {
        let request = json!({"state":fixture["state"],"questions":fixture["questions"]});
        let prepared = processor.prepare(request.to_string().as_bytes()).unwrap();
        assert_eq!(
            prepared.rows.len(),
            fixture["items"].as_array().unwrap().len()
        );
        assert_eq!(
            prepared.context.input_tokens(),
            fixture["unique_tokens"].as_u64().unwrap() as usize
        );
        assert_eq!(
            prepared.processed_tokens(),
            fixture["processed_tokens"].as_u64().unwrap() as usize
        );
        for (row, item) in prepared
            .rows
            .iter()
            .zip(fixture["items"].as_array().unwrap())
        {
            assert_eq!(
                row.ids,
                serde_json::from_value::<Vec<u32>>(item["ids"].clone()).unwrap(),
                "{}",
                fixture["name"]
            );
            let count = item["nopts"][0].as_u64().unwrap() as usize;
            assert_eq!(row.candidate_ids, labels[..count]);
            assert_eq!(row.readout_position, row.ids.len() - 1);
        }
    }
}
#[test]
#[ignore = "requires the pinned tokenizer/config; CPU only, no tensor loading"]
fn pinned_assemblies_and_state_truncation() {
    let processor = processor();
    let golden: Value = serde_json::from_str(include_str!("data/responses.json")).unwrap();
    for fixture in golden.as_array().unwrap() {
        let prepared = processor
            .prepare(fixture["request"].to_string().as_bytes())
            .unwrap();
        let logits = serde_json::from_value(fixture["logits"].clone()).unwrap();
        assert_eq!(
            prepared.context.finish(logits).unwrap()["answers"],
            fixture["expected_answers"],
            "{}",
            fixture["name"]
        );
    }
    let request = json!({"state":"hello ".repeat(40000),"questions":{"q":{"instructions":"Choose.","criteria":["A","B"]}}});
    let row = processor
        .prepare(request.to_string().as_bytes())
        .unwrap()
        .rows
        .pop()
        .unwrap();
    assert!(row.ids.len() > 32768 && row.ids.len() <= 36864);
    let first = row.ids[..32768].to_vec();
    let request = json!({"state":"hello ".repeat(45000),"questions":{"q":{"instructions":"Choose.","criteria":["A","B"]}}});
    assert_eq!(
        processor
            .prepare(request.to_string().as_bytes())
            .unwrap()
            .rows[0]
            .ids[..32768],
        first
    );
}
