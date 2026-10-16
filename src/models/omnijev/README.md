# OmniJev

OmniJev-4B v1.1 ([#114](https://github.com/ThinkFlowLab/system1-omni/issues/114))
answers Choice, Noul and Score questions about one image with a Qwen3.5-4B backbone,
a merged LoRA adapter, two option-marker tokens and trained FP32 heads. This directory
holds the native worker: the request contract, image decoding and preparation, the
branch layout, execution on the shared Qwen3.5 vision and language CUDA path, the
heads, and response finishing, each checked against the pinned reference. The
[preparation recipe](../../../recipe/omnijev/README.md) exports the checkpoint and
regenerates the fixtures; [the worker recipe](../../../recipe/omnijev/native.md) builds,
runs and validates the worker.

## Pinned artifacts

| Artifact | Revision |
| --- | --- |
| Reference code: [`tinnel123666888/OmniJev`](https://github.com/tinnel123666888/OmniJev/tree/14dbec4f71e194852c8d7b88ab36ef639493f400) | `14dbec4f71e194852c8d7b88ab36ef639493f400` |
| Adapter, heads, tokenizer and processor: [`tinnel123/OmniJev`](https://huggingface.co/tinnel123/OmniJev/tree/ffe5f436eaf22e20e2f041f8e74e121fd057a6cb), tag `v1.1` | `ffe5f436eaf22e20e2f041f8e74e121fd057a6cb` |
| Base model: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |

The adapter names its base without a revision; `851bf6e` is the one Cua-S1 pins. The
reference environment is transformers 5.17.0, torch 2.14.0, torchvision 0.29.0 and
PEFT 0.19.1, without `fla-core`, so transformers runs its PyTorch Gated DeltaNet code.
The reference's branch path needs transformers 5.7.0 or later: 5.5 and 5.6 start the
linear-attention state of every row from zero, and 5.2 to 5.4 fail in it.

## Request and response

```json
{"model": "tinnel123/OmniJev",
 "state": {"images": ["data:image/png;base64,..."]},
 "questions": {
   "visible": {"type": "noul", "instructions": "A person is visible."},
   "next": {"type": "choice", "instructions": "Which operation comes next?",
            "criteria": {"click": null, "type": "Enter text into the focused field"}},
   "risk": {"type": "score", "instructions": "How risky is acting on this screen?",
            "levels": ["safe", "check first", "dangerous"]}}}
```

- `model` may be omitted, `tinnel123/OmniJev`, `omnijev` or `omnijev-4b-v1.1`.
- `state.images` holds exactly one inline PNG or JPEG.
- Noul takes `instructions` and an optional `region` (`{"box": [x1, y1, x2, y2]}`).
- Choice takes Jev's `criteria` map (a key alone, or `key: rubric`), or the reference's
  `options` list, whose entries carry `key`, `text`, `region` or `abstain`. Abstain
  entries are skipped, since the head has its own abstain probability; a missing key
  falls back to the text, then to `option_<index>`.
- Score takes `levels` in order, or the keys of `criteria`.

Option text is cut to 200 characters, and region coordinates are rounded half to even,
both as the reference does. Answers keep the reference's fields: `noul`; `choice`,
`probabilities`, `abstain`, `valid` and `confidence`; or `score`, `probabilities` and
`confidence`; each with `latency_s` and `latency_total_s`, which time the request's
preparation, its vision and language passes and its heads, as the reference times `ask_branch`
from encoding to the heads. Probabilities are
rounded to four decimals. The response is
`{"model": "tinnel123/OmniJev", "answers": {id: answer}, "usage": {"input_tokens": n,
"output_tokens": 0}}`, with the reference's input-token count: the shared prefix once
plus every row.

Deliberate differences from the reference:

- One image per request. The reference tiles several still images into a numbered
  panel and reads a video as a frame mosaic; both come later.
- Text containing `<|opt|>`, `<|/opt|>`, `<|image_pad|>`, `<|video_pad|>`,
  `<|vision_start|>` or `<|vision_end|>` is refused. The reference would read it as
  option or image markers and misplace its readouts.
- Images arrive as data URLs instead of file paths. PNG files decode to the same RGB
  values as Pillow's `convert("RGB")`, including its 16-bit conversions; JPEG files use
  the `image` crate's decoder, whose values can differ from Pillow's libjpeg-turbo by
  rounding (by at most 3 in the committed sample).
- Explicit limits: a 12 MiB body; an 8 MiB image of at most 4096 pixels per side and
  an aspect ratio of at most 200, which the processor also requires; 64 questions with
  ids of at most 256 characters; 1 to 255 options per question and 1,024 in all;
  8,192 characters per text field; 8,192 tokens per question; 65,536 processed
  tokens per request; and 16,384 option-text tokens whose log-probabilities are read.
- Fields the reference ignores (Noul `criteria`, `criteria` next to Choice `options` or
  Score `levels`, unknown fields), and values it would turn into text (numbers as
  levels, for example), are refused.

## Preparation and layout

[`processing.rs`](native/src/processing.rs) renders each question as the reference's
chat template does:

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{instructions}<|im_end|>
<|im_start|>assistant
<think>
<|opt|>{option 1}<|/opt|><|opt|>{option 2}<|/opt|>...
```

The template trims the user content, so trailing whitespace in the instructions goes,
by Python's definition of whitespace. The image is decoded and preprocessed by the
shared Qwen crate (`image_decode` and `image_preprocess`, the port of the reference's
`Qwen2VLImageProcessor`) with `MSO1`'s 602,112-pixel budget (it overrides the processor
config's 401,408) and the 65,536-pixel minimum, and its placeholder expands to one token
per 2×2 merged patch.

The layout follows `MSO1.ask_branch`. The shared prefix is the questions' common token
prefix, cut at least one token before any question's first `<|opt|>`. Each row is the
rest of its question plus one option block. A row's readouts are `u` at its closing
marker, `zq` at the token before its option marker, and, for the LM features, each
option-text token with the position that predicts it. Rotary positions are Qwen3.5's
T/H/W positions (the shared `inputs::image_positions`); after the prefix, rows continue
from its largest position plus one. Since every row equals a plain forward over prefix
plus row up to rounding, execution may choose its own split points.

## Execution

[`executor.rs`](native/src/executor.rs) runs the vision tower once per request, then the
request's prefix up to its last multiple of 64 tokens once, each question's text up to the
last multiple of 64 before its first option once, and each option's block with the
tokens that remain before it, each as a continuation of the state before it
(`Model::readout_window`). The passes use the fixed-algorithm GEMMs and split at
multiples of 64, so every answer equals that of one plain pass per (question, option)
row over the prefix and that row, which `Executor::execute_rows` keeps for the checks.
The passes read the final-norm hidden states at
`zq` and `u`, which is what the reference's `hidden_states[-1]` holds, and the
option-text tokens' log-probabilities: BF16 logits from the tied embedding over the
248,079-entry vocabulary, converted to float32, with a float64 sum in the log-softmax. Score rows skip the
log-probabilities, which its ordinal head does not use. The heads run on the CPU; the
question state `zq` is the first row's, as in the reference.

## Heads and finishing

[`heads.rs`](native/src/heads.rs) runs the trained heads in FP32 on the CPU. Choice
and Noul use `OptionScorer`: gated option and question MLPs, the LM features, a
per-type temperature, and for Choice a softmax that includes the abstain logit; Noul
is a sigmoid. Score uses only the cumulative-link ordinal head. The six LM
features of an option come from its text tokens' log-probabilities. A plain Noul
option has no text, so its features are zero.

`contract::answer` follows `MSO1._finish`: Noul's log-odds bias before its
temperature, the saved temperatures for each type, Choice's abstain and validity,
Score's renormalization, the float32 values the reference stores in between, Jev's
confidence, and four-decimal rounding. The float32 sum of more than four head outputs
can add in a different order than PyTorch's, which moves the fourth decimal by one at
most; the reference's own sum on a GPU has the same freedom.

## Validation

[`generate_fixtures.py`](../../../tests/omnijev/generate_fixtures.py) runs the
reference's own image loading and processor, preparation, layout, rotary positions,
heads and finishing on the CPU without loading the backbone. The tests in
[`tests/omnijev/`](../../../tests/omnijev/) compare against those fixtures:

- prompts, answer keys and option text, marker positions, image grids, rotary
  positions, the prefix, rows, readouts and token counts for seven requests, exactly;
- decoded RGB values and pixel values for nine images (patterns at four sizes, RGBA,
  grayscale, palette, 16-bit grayscale and JPEG), exactly for PNG, and for JPEG the
  decoder within 3 of Pillow and the processor exactly on Pillow's values;
- the LM features within 1e-6, and finished answers for eighteen head outputs, exactly
  up to four options and within one unit of the fourth decimal beyond;
- refused requests, with the reason, the limits, and the export manifest checks.

Opt-in checks need the checkpoint or an export: token ids against the checkpoint's
tokenizer, exactly; the heads against an export, within 1e-5 (relative for values above
one); and on a GPU, the readout pass (`tests/qwen3_5/readout.rs`: equal to
`forward_fixed` for text, independent of the other rows and targets read,
log-probabilities against the CPU, and windows continuing captured prefixes equal to one
pass) and the worker (repeatable answers, independent of question order, and the reused
prefix equal to the per-row passes bit for bit for questions that split three ways
against the 64-token chunks). The two prefix states the executor keeps take about
50 MiB each plus 32 KiB per token of their longest prefix. [`validate.py`](../../../recipe/omnijev/validate.py) compares the
worker's answers with the reference's.

The Rust code adapts OmniJev's Apache-2.0 code; its copyright and license are retained
in [`native/LICENSE.omnijev`](native/LICENSE.omnijev). No model weights are
distributed here.
