use super::*;
use crate::caches::CacheCfg;
use tokenizers::{
    AddedToken, models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
};

fn processor() -> Processor {
    let vocab = [("[UNK]".to_owned(), 0)].into_iter().collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|vision_start|>", true),
        AddedToken::from("<|image_pad|>", true),
        AddedToken::from("<|vision_end|>", true),
    ]);
    let caches = Caches::new(CacheCfg {
        enabled: true,
        l1: true,
        l2: true,
        l3: false,
        l1_max: 8,
        l2_bytes: 1 << 20,
        l3_bytes: 0,
    });
    caches.l2_insert(
        Processor::url_key("prepared://image"),
        Arc::new(ImageAsset {
            grid_thw: [1, 16, 16],
            embeddings: vec![half::bf16::ONE; 64 * 5120],
        }),
    );
    Processor {
        image_pad: tokenizer.token_to_id("<|image_pad|>").unwrap(),
        vision_end: tokenizer.token_to_id("<|vision_end|>").unwrap(),
        tokenizer,
        labels: vec!["A".into(), "B".into()],
        max_length: 256,
        source: ImageSource::Prepared(None),
        model_index_hash: "test".into(),
        caches,
    }
}

#[test]
fn image_placeholders_are_rejected_before_and_after_l1_warmup() {
    let processor = processor();
    let valid = serde_json::json!({
        "kind": "choice", "state": [{"image": "prepared://image"}],
        "question": "Pick one.", "options": ["yes", "no"],
    });
    let prepare = |request: &Value| {
        let compiled =
            contract::compile(&serde_json::to_vec(request).unwrap(), &processor.labels).unwrap();
        processor.prepare_cached(
            &compiled,
            Readout {
                token_ids: vec![0, 1],
                bias: vec![0.0; 2],
                temperature: 1.0,
            },
            Instant::now(),
        )
    };
    let mut bad_question = valid.clone();
    bad_question["question"] = serde_json::json!("Pick <|image_pad|>.");
    let mut bad_option = valid.clone();
    bad_option["options"][0] = serde_json::json!("<|image_pad|>");
    for request in [&bad_question, &bad_option] {
        assert_eq!(
            prepare(request)
                .err()
                .expect("cold request must fail")
                .status,
            400
        );
    }
    assert_eq!(processor.caches.snapshot().l1_records, 0);
    assert_eq!(prepare(&valid).unwrap().cache_note, "l1=miss,l3=off,p=-");
    assert_eq!(prepare(&valid).unwrap().cache_note, "l1=hit,l3=off,p=-");
    for request in [&bad_question, &bad_option] {
        assert_eq!(
            prepare(request)
                .err()
                .expect("warm request must fail")
                .status,
            400
        );
    }
}

const RED: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";
const GREEN: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNg+M8AAAICAQB7CYF4AAAAAElFTkSuQmCC";

fn image_request(images: &[&str]) -> Value {
    serde_json::json!({
        "kind": "choice", "state": images.iter().map(|url| serde_json::json!({"image": url})).collect::<Vec<_>>(),
        "question": "Pick one.", "options": ["yes", "no"],
    })
}

fn prepare_request(processor: &Processor, raw: &Value) -> Result<PreparedRequest, Reject> {
    use crate::executor::LabelHead;
    use safetensors::{Dtype, tensor::TensorView};
    use std::sync::OnceLock;

    // Exercise the public cache-on/off dispatch with a CPU-only label-head export.
    static HEAD: OnceLock<(LabelHead, Value)> = OnceLock::new();
    let (head, manifest) = HEAD.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            include_str!("../qwen3_5/data/config.json"),
        )
        .unwrap();
        let rows = vec![0u8; 2 * 5120 * 4];
        let ids: Vec<_> = [0i64, 1].into_iter().flat_map(i64::to_le_bytes).collect();
        safetensors::serialize_to_file(
            [
                (
                    "rows",
                    TensorView::new(Dtype::F32, vec![2, 5120], &rows).unwrap(),
                ),
                ("ids", TensorView::new(Dtype::I64, vec![2], &ids).unwrap()),
            ],
            None,
            &dir.path().join("head.safetensors"),
        )
        .unwrap();
        let manifest = serde_json::json!({
            "label_head": {"file": "head.safetensors", "ids": [0, 1]},
            "label_ids": [0, 1], "verbalizer_ids": vec![0; 24],
            "verbalizer_bias": vec![0.0; 24], "temperatures": {"choice": 1.0},
        });
        (LabelHead::load(dir.path(), &manifest).unwrap(), manifest)
    });
    processor.prepare(head, manifest, &serde_json::to_vec(raw).unwrap())
}

fn prepare_online(processor: &Processor, images: &[&str]) -> Result<PreparedRequest, Reject> {
    prepare_request(processor, &image_request(images))
}

fn online_processor(enabled: bool, l2: bool) -> Processor {
    let mut processor = processor();
    processor.source = ImageSource::Online("unused-by-cpu".into());
    processor.caches = Caches::new(CacheCfg {
        enabled,
        l1: true,
        l2,
        l3: false,
        l1_max: 8,
        l2_bytes: 1 << 20,
        l3_bytes: 0,
    });
    processor
}

#[test]
fn online_preparation_has_pixels_not_empty_embeddings_and_preserves_l1_geometry() {
    let mut processor = processor();
    processor.source = ImageSource::Online("unused-by-cpu".into());
    let cold = prepare_online(&processor, &[RED]).unwrap();
    let warm = prepare_online(&processor, &[RED]).unwrap();
    assert!(cold.cache_note.contains("l1=miss"));
    assert!(warm.cache_note.contains("l1=hit"));
    let (MmPlan::Full(cold), MmPlan::Full(warm)) = (cold.plan, warm.plan) else {
        panic!("expected full plans")
    };
    assert_eq!(cold.ids, warm.ids);
    assert_eq!(cold.positions, warm.positions);
    let ImageInput::Inline { pixels, .. } = &warm.blocks[0].asset else {
        panic!("CPU must not claim ready embeddings")
    };
    assert_eq!(pixels.image_grid_thw, [1, 16, 16]);
    assert_eq!(pixels.image_tokens(), 64);
    assert_eq!(warm.blocks[0].end - warm.blocks[0].start, 64);
}

#[test]
fn online_rejects_unsupported_urls_and_full_context_overflow_before_admission() {
    let mut processor = processor();
    processor.source = ImageSource::Online("unused-by-cpu".into());
    assert!(
        processor
            .load_image("https://example.com/image.png")
            .is_err()
    );
    assert!(processor.load_image("prepared://unknown").is_err());
    processor.max_length = 63;
    assert_eq!(
        prepare_online(&processor, &[RED]).err().unwrap().status,
        400
    );
    // A rejected cold request must not create an L1 record.
    assert_eq!(processor.caches.snapshot().l1_records, 0);
    assert_eq!(
        prepare_online(&processor, &[RED]).err().unwrap().status,
        400
    );
}

#[test]
fn request_budget_stops_before_later_images_with_caches_on_or_off() {
    for (enabled, l2) in [(true, true), (true, false), (false, false)] {
        for second in [RED, GREEN] {
            let mut processor = online_processor(enabled, l2);
            let raw = image_request(&[RED, second, "data:image/png;base64,%%%"]);
            let compiled =
                contract::compile(&serde_json::to_vec(&raw).unwrap(), &processor.labels).unwrap();
            let text_tokens = processor.tokenize(&compiled.prompt).unwrap().len();
            // Exactly one expanded image fits; the second reference must reject
            // before the third image's deliberately invalid bytes are decoded.
            processor.max_length = text_tokens + 63;
            let error = prepare_request(&processor, &raw).err().unwrap();
            let message = error.body.to_string();
            assert_eq!(error.status, 400);
            assert!(message.contains("maximum context length"), "{message}");
            assert!(
                message.contains(&format!("value={}", text_tokens + 126)),
                "{message}"
            );
            assert!(!message.contains("invalid base64"), "{message}");
            assert_eq!(processor.caches.snapshot().l2_records, 0);
        }
    }
}

#[test]
fn repeated_references_share_pixels_without_losing_blocks_or_positions() {
    for (enabled, l2) in [(true, true), (true, false), (false, false)] {
        let processor = online_processor(enabled, l2);
        let raw = image_request(&[RED, GREEN, RED]);
        let MmPlan::Full(mm) = prepare_request(&processor, &raw).unwrap().plan else {
            panic!("expected full plan")
        };
        let images: Vec<_> = mm
            .blocks
            .iter()
            .map(|b| match &b.asset {
                ImageInput::Inline { key, pixels } => (key, pixels),
                _ => panic!("expected inline pixels"),
            })
            .collect();
        assert_eq!(images.len(), 3);
        assert_eq!(images[0].0, images[2].0);
        assert_ne!(images[0].0, images[1].0);
        assert!(Arc::ptr_eq(images[0].1, images[2].1));
        assert!(!Arc::ptr_eq(images[0].1, images[1].1));
        assert_eq!(
            mm.blocks.iter().map(|b| b.end - b.start).sum::<usize>(),
            192
        );
        let compiled =
            contract::compile(&serde_json::to_vec(&raw).unwrap(), &processor.labels).unwrap();
        let text_ids = processor.tokenize(&compiled.prompt).unwrap();
        let expected =
            images::expand_grids(&text_ids, processor.image_pad, &[[1, 16, 16]; 3]).unwrap();
        assert_eq!(mm.ids, expected.ids);
        assert_eq!(mm.positions, expected.positions);
        assert_eq!(
            mm.blocks
                .iter()
                .map(|b| (b.start, b.end))
                .collect::<Vec<_>>(),
            expected
                .blocks
                .iter()
                .map(|b| (b.start, b.end))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn text_over_budget_is_rejected_before_any_image_decode() {
    for enabled in [true, false] {
        let mut processor = online_processor(enabled, false);
        processor.max_length = 1;
        let error = prepare_online(&processor, &["data:image/png;base64,%%%"])
            .err()
            .unwrap();
        assert!(error.body.to_string().contains("maximum context length"));
    }
}

#[test]
fn warm_single_image_checks_full_length_before_reloading_pixels() {
    let mut processor = online_processor(true, false);
    let accepted = prepare_online(&processor, &[RED]).unwrap();
    assert_eq!(processor.caches.snapshot().l1_records, 1);
    processor.max_length = accepted.context.input_tokens - 1;
    // Preserve the warm prefix record but make image loading fail if reached.
    processor.source = ImageSource::Prepared(None);
    let error = prepare_online(&processor, &[RED]).err().unwrap();
    let message = error.body.to_string();
    assert!(message.contains("maximum context length"), "{message}");
    assert!(!message.contains("no imgcache"));
}

#[test]
fn disabling_l2_retains_typed_images_and_does_not_publish_embeddings_on_cpu() {
    let processor = online_processor(true, false);
    for _ in 0..2 {
        let MmPlan::Full(mm) = prepare_online(&processor, &[RED, RED]).unwrap().plan else {
            panic!("expected full plan")
        };
        assert_eq!(mm.blocks.len(), 2);
        assert!(
            mm.blocks
                .iter()
                .all(|b| matches!(b.asset, ImageInput::Inline { .. }))
        );
        assert_eq!(processor.caches.snapshot().l2_records, 0);
    }
}

#[test]
fn aligned_image_end_still_loads_images_without_a_live_prefix() {
    for l3 in [false, true] {
        let mut processor = online_processor(true, false);
        processor.caches = Caches::new(CacheCfg {
            enabled: true,
            l1: true,
            l2: false,
            l3,
            l1_max: 8,
            l2_bytes: 0,
            l3_bytes: 1 << 20,
        });
        let request = (0..64)
            .map(|padding| {
                let mut request = image_request(&[RED]);
                request["state"] = serde_json::json!(["align ".repeat(padding), {"image": RED}]);
                request
            })
            .find(|request| {
                let compiled =
                    contract::compile(&serde_json::to_vec(request).unwrap(), &processor.labels)
                        .unwrap();
                let ids = processor.tokenize(&compiled.prompt).unwrap();
                (ids.iter()
                    .position(|&id| id == processor.image_pad)
                    .unwrap()
                    + 64)
                    % 64
                    == 0
            })
            .unwrap();
        let cold = prepare_request(&processor, &request).unwrap();
        let before = processor.caches.snapshot();
        let warm = prepare_request(&processor, &request).unwrap();
        assert!(warm.cache_note.contains("l1=hit"));
        let after = processor.caches.snapshot();
        assert!(after.l2_miss > before.l2_miss);
        assert_eq!(after.l3_miss, before.l3_miss + u64::from(l3));
        assert_eq!(matches!(warm.plan, MmPlan::Populate { .. }), l3);
        assert!(
            warm.cache_note
                .contains(if l3 { "l3=miss" } else { "l3=off" })
        );
        let full = |plan| match plan {
            MmPlan::Full(mm) | MmPlan::Populate { mm, .. } => mm,
            _ => panic!("no live GPU prefix exists"),
        };
        let (cold, warm) = (full(cold.plan), full(warm.plan));
        assert_eq!(cold.ids, warm.ids);
        assert_eq!(cold.positions, warm.positions);
        assert_eq!(warm.blocks[0].end % 64, 0);
        assert_eq!(warm.blocks[0].asset.n_tokens(), 64);
    }
}
