# Cua-S1 4B 0.2 model engine

This directory owns Cua-S1 4B 0.2 ([#10](https://github.com/ThinkFlowLab/system1-omni/issues/10)): request mapping, prompt construction, adapter selection, execution, and the answer-letter readout. This page records the pinned upstream revisions, the inference contract an implementation must match, and how its outputs will be compared with the upstream reference.

Status: a reference worker for the `text` adapter loads the model through Hugging Face Transformers and PEFT: [`text/`](text/), served by [`src/frontend/cua_s1_text.py`](../../frontend/cua_s1_text.py), with setup in [`recipe/cua_s1/text.md`](../../../recipe/cua_s1/text.md). It is the correctness reference for the native worker in [`native/`](native/): Rust, with the Qwen3.5 forward pass on the CUDA kernels in [`src/backends/cuda/qwen3_5/`](../../backends/cuda/qwen3_5/), set up as in [`recipe/cua_s1/native.md`](../../../recipe/cua_s1/native.md). The `multimodal` adapter is deferred; see [Not covered yet](#not-covered-yet).

## Pinned revisions

| Artifact | Revision |
| --- | --- |
| Reference code: [`trycua/cua`](https://github.com/trycua/cua/tree/0e75660ce4c2edda519e0c795fa3ad98abf4e76f/libs/cua-s1) `libs/cua-s1` | `0e75660ce4c2edda519e0c795fa3ad98abf4e76f` |
| Base model: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Adapters: [`cua-ai/cua-s1-4b-0.2`](https://huggingface.co/cua-ai/cua-s1-4b-0.2) | `16818868b0cc7813808aae4e87b417657046ab79` |

The model revisions are the ones upstream pins in `libs/cua-s1/ci/weights.lock.json`, which lists the size and SHA-256 of every file. Downloads are checked against that file with upstream's `ci/fetch_pinned_weights.py --verify-only`.

The reference implementation is `cua_s1.four_b.FourBModel`. The reference environment is upstream's `four-b` lock (`libs/cua-s1/python/uv.lock`) on Python 3.12, as upstream measured: torch 2.14.0, Transformers 5.17.0, PEFT 0.21.0 and torchvision 0.29.0. It has neither `flash-linear-attention` nor `causal-conv1d`, so Transformers runs the reference PyTorch implementations of those operations.

## Model

- **Base.** Qwen3.5-4B has 32 decoder layers: 24 Gated DeltaNet (linear attention) layers and 8 full-attention layers (layers 3, 7, ..., 31, counting from 0). Hidden size is 2560, the MLP intermediate size is 9216, the embedding has 248,320 rows, and the input and output embeddings are tied. The checkpoint also contains one MTP layer, which is not used.
- **Adapters.** `text/` and `multimodal/` are two independently trained PEFT LoRA adapters, both rank 16 and alpha 32 (scale 2), stored in fp32.
  - `text/` targets `q_proj`, `k_proj`, `v_proj` and `o_proj` in the 8 full-attention layers, and `gate_proj`, `up_proj` and `down_proj` in all 32 MLPs. The Gated DeltaNet attention blocks keep their base weights; only the MLPs in those layers are adapted.
  - `multimodal/` targets the same language modules, plus `linear_fc1` and `linear_fc2` in the 24 vision blocks and in the vision merger. Its keys follow the image-text model layout (`model.language_model.*`, `model.visual.*`), so the two adapters are not interchangeable.
- **Model classes.** Upstream loads `text` with `AutoModelForCausalLM` (`Qwen3_5ForCausalLM`) and `multimodal` with `AutoModelForImageTextToText` (`Qwen3_5ForConditionalGeneration`). `FourBModel` defaults to bfloat16. PEFT keeps the adapter weights in fp32 and, while the adapter is not merged, computes each LoRA branch in fp32 and casts the sum back to bfloat16.

## Inference contract

A decision is one forward pass over one prompt, with no decoding.

1. Options get the letters `A` to `Z` in request order, so a question has at most 26 options.
2. The prompt has a fixed system message and a user message in this layout (upstream `build_prompt`, text modality):

   ```text
   Goal: <goal>

   App: <app>
   Task family: <task family>

   Accessibility tree:
   <state>

   Options:
   A. <role> "<label>" -> <action>
   B. ...

   Answer with a single letter.
   ```

   The `Goal` line and the blank line after it are left out when the goal is empty. For `multimodal`, the user message starts with the image, and the `Accessibility tree` block is replaced by `The current screenshot is attached.`
3. The base model's chat template is applied with `add_generation_prompt=True` and without an `enable_thinking` argument, so the prompt ends with `<|im_start|>assistant\n<think>\n`. Upstream's training scripts (`training/train_4b*.py`) build prompts the same way. Passing `enable_thinking=False` produces a different suffix (`<think>\n\n</think>\n\n`) and must not be used.
4. Tokenization adds no special tokens. Text in the request that spells a special token, such as `<|im_end|>`, is encoded as that token, as upstream does. Each of the letters `A` to `Z` is a single token, with ids 32 to 57.
5. The readout takes the logits at the last position, which leave `lm_head` in the model dtype, keeps the ids of the letters in use, casts them to fp32 and applies a softmax. The results are the option probabilities, in option order.

For upstream's positive fixture (`libs/cua-driver/examples/jev-use/fixtures/jev-choice-request-v1.json`, 3 options), the prompt is 218 tokens.

## Mapping `/v1/systemone` requests

The worker follows upstream's closed-candidate chooser (`libs/cua-driver/examples/jev-use/python/decision_models.py`), which produced upstream's published fixture results. It uses the same fixed values and label escaping. The request `state` takes the place of the chooser's rendered screen regions.

| `/v1/systemone` field | Prompt field |
| --- | --- |
| `state` | Accessibility tree. A string is used as is. An object or array is serialized as by Python's `json.dumps(value, ensure_ascii=False)`: separators `, ` and `: `, keys in request order, non-ASCII text kept. |
| Question `instructions` | Goal, serialized in the same way as `state`. |
| `criteria` keys, in request order | Options `A`, `B`, ... The keys themselves are not shown to the model unless their value is `null`. |
| `criteria` values | Option label. A string is escaped with `json.dumps(value, ensure_ascii=False)[1:-1]`, as in the chooser. An object or array is serialized as above and then escaped the same way. `null` uses the key, escaped the same way. |
| Fixed values | `App: Cua Driver`, `Task family: closed-candidate decision`, role `Decision`, action `select`. |

The request `model` is `cua-s1-4b-0.2`. Each question in a request is a separate prompt and forward pass over the shared `state`. Each answer uses the Jev choice format, `{"type": "choice", "choice": ..., "probabilities": ..., "confidence": ...}`:

- `choice` is the option with the highest probability. Ties go to the earliest option in request order. Upstream's chooser returns an error on a tie instead.
- `probabilities` maps every option key to its probability.
- `confidence` is `1 - H(p) / ln(n)` for `n` options, and 1 when `n` is 1, the same normalized entropy the LAYA worker reports. The Jev API only says that confidence is derived from `probabilities`. Upstream's chooser reports `p_max`, which can still be read from `probabilities`.

The response `model` is `cua-ai/cua-s1-4b-0.2@<adapter revision>:<modality>`, in upstream's identity format, for example `cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:text`. `usage.input_tokens` is the total prompt length over all questions, and `usage.output_tokens` is 0.

An error rejects the whole request. Its body is `{"detail": "<message>"}`, as the LAYA worker returns, and the message names the problem.

The status is `400` when the body is not a usable JSON object: invalid JSON or UTF-8, `NaN`, `Infinity` or a number out of range, a lone surrogate such as `\ud800`, nesting too deep to parse, or a key repeated in any object.

The status is `422` when a well-formed request cannot be answered:

- a `score` or `noul` question, since the adapters were trained only on closed-option choices;
- a question without an `instructions` field (`null` is allowed), or a `choice` with no options or more than 26 options;
- a `criteria` value that is a number or a boolean;
- an empty `state` (`""`, `{}` or `[]`);
- a `model` other than `cua-s1-4b-0.2`.

## Validation

**Inputs.** The fixed input set is upstream's two checked-in fixtures, converted to `/v1/systemone` requests with the chooser's rendered regions as `state`, plus `/v1/systemone` choice requests. These cover 1 to 26 options, short and long states, string and structured `state`, `instructions` and `criteria`, `null` criteria, non-ASCII text, and text that spells a special token. Each input is scored once per configuration.

**Tolerances.** These are declared before any comparison is run:

| Comparison | Setup | Pass condition |
| --- | --- | --- |
| Worker prompt vs upstream `build_prompt` | Same inputs | Identical token ids |
| Worker vs upstream `FourBModel` | Same GPU and reference environment, bfloat16, adapter not merged, full logits, one unpadded prompt per forward pass | Identical fp32 probabilities |
| Through the frontend vs direct to the worker | Same worker | Identical status, content type and body bytes |
| Native engine vs fp32 worker | Same GPU; the engine runs in bfloat16; the fp32 worker runs with TF32 disabled | (1) Over the whole input set, the largest per-option probability difference is at most twice the bfloat16 worker's largest difference from the fp32 worker, plus 0.01. (2) The top option matches wherever the fp32 worker's top-two margin is at least 0.05. |

The bfloat16 worker's own difference from the fp32 worker is reported next to each native-engine result.

**Performance.** Performance results will state the hardware, revisions, commands and raw results. Load and warmup are reported separately from warm latency, and warm latency is broken down by prompt length and option count.

## Not covered yet

- The `multimodal` adapter: image preprocessing, the vision tower and the vision LoRA. This is tracked in [#10](https://github.com/ThinkFlowLab/system1-omni/issues/10).
- `score` and `noul` questions.
- More than 26 options per question.
- The Metal backend.
- The 0.1 adapters, which are a separate artifact.
