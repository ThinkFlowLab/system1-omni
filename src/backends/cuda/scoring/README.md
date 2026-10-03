# Fused candidate scoring

One CUDA kernel that turns a query vector and a matrix of candidate vectors into
a probability distribution, without materialising the similarities in between.

This is the readout both models that need it share the shape of:

| Model | Similarity | Notes |
| --- | --- | --- |
| Kev ([#27](https://github.com/ThinkFlowLab/system1-omni/issues/27)) | `scale * <k_proj(c_i), q_proj(s)>` | candidates are hidden states, one question per prefill row |
| CLM ([#9](https://github.com/ThinkFlowLab/system1-omni/issues/9)) | `exp(logit_scale) * cos(state_head(s), action_head(c))` | cosine, so `normalize = 1` |

They differ in the similarity and in whether a projection happens first; what
they share is the primitive. `#9` describes CLM's answer as "the softmax over the
question's candidates" of `exp(logit_scale) * cos(state_head(s), action_head(c))`,
and `#27` describes Kev's as `scale * (k(h_opts) @ q(h_decide))` with
`scale = 1/sqrt(256)`, softmaxed over that question's candidates. Neither is in
`#19`'s `gemm.cu`, which is a bf16 cuBLASLt GEMM.

## What is fused, precisely

Not merely "fewer launches". When the similarity is a cosine, each candidate's
norm and its dot with the query need **the same elements**, so both accumulate in
a single read of `C`. For Kev that is 255 rows of 2560 floats read twice per
question in the unfused version and once here. Beyond that:

- One thread block per question; the stable softmax runs over shared memory, so
  there is no second kernel, no atomics, and no global round trip for the
  similarities.
- `K ≤ 255` (the serving API's limit), which is what makes the in-block softmax
  possible at all — the whole similarity vector fits in shared memory.
- A batch is a grid over questions, not a loop of launches.

The query norm is computed once per block rather than once per candidate.

## Files

| File | What it is | Verified |
| --- | --- | --- |
| `reference.py` | The oracle: float64, deliberately slow and obvious | **yes**, 17 tests |
| `kernel_simulation.py` | The kernel's algorithm in numpy: float32, warp tree order, same norm flooring | **yes** |
| `candidate_scoring.cu` | The kernel and its C ABI | **yes**, compiled and checked against the reference on sm_89 |

## Verification state

**Verified on real hardware**, on two driver stacks. RTX 4090 (sm_89), CUDA 13.0
(V13.0.88), compiled with `./build.sh ./out 89`:

| driver | worst absolute difference (tolerance 1e-4) |
| --- | --- |
| 595.58.03 | 1.835e-07 |
| 580.76.05 | 1.835e-07 |

`gpu_parity.py` checks three things, and all three run:

- **13 fixed cases**, including the degenerate ones (single candidate, identical
  candidates, a zero candidate) and the one whose raw logits overflow a naive
  softmax.
- **The batch path**, as a grid over questions: 1, 8, 4 and 3 questions with `K`
  up to 255 and `D` up to 256. Each row is compared with the reference and checked
  to sum to 1. The single-question call goes through `cs_score_candidates` and the
  rest through `cs_score_candidates_batch`, so the two entry points have to agree.
- **Rejected arguments**: `K` above 255, `K` of zero, `D` of zero, a zero scale, a
  negative temperature and a null query must each return an error rather than a
  scored answer. `K` above 255 is the one that matters: the shared array is sized
  `MAX_K`, so scoring the first 255 of a larger set would be a confident wrong
  answer instead of a failure.

### Two defects the first GPU run found

Both were in how the kernel was called or reduced; neither was visible without a
GPU, and neither was caught by the simulation below.

1. **The parity harness passed host pointers.** A numpy buffer is host memory, so
   the kernel dereferenced it as device memory — `cudaErrorIllegalAddress`, not a
   wrong answer. `compute-sanitizer` located it as an invalid global read with
   consecutive lanes on consecutive addresses, which is the coalesced load
   pattern. Fixed by allocating with `cudaMalloc`, copying in, synchronising and
   copying out.
2. **The query norm was counted `warps` times.** The inner loop
   `d = lane; d < D; d += WARP` already covers every element of `D` across one
   warp's 32 lanes, but every warp computed the same partial and the partials
   were then summed across warps — so the norm came out `sqrt(8)` too large on a
   256-thread block. Every unnormalized case sat at float32 rounding while the
   normalize cases were off by ~1e-2, which is what pointed at it. Fixed by
   having warp 0 compute it alone.

That second one is worth recording precisely: **the simulation did not catch it.**
`kernel_simulation.py` computed the query norm correctly, because it was written
from the intent rather than transcribed from the `.cu`. A simulation is only as
good as its fidelity, and the hardware is what settles it.

## What the reference and simulation established before hardware

1. **The reference is right.** All cases sum to 1; a single candidate gets 1.0;
   identical candidates get exactly `1/K`; a zero candidate does not produce NaN;
   logits large enough to overflow a naive softmax give a finite distribution.
2. **The algorithm agrees with the reference.** `kernel_simulation.py` reproduces
   the intended arithmetic — float32 accumulation, the max-subtracted softmax,
   the norm floored as `max(sqrt(sum), eps)` and **not** as
   `sqrt(max(sum, eps))` — and matched the float64 oracle to 1.8e-07 over the
   fixed cases. The measured hardware worst case is the same order, 1.835e-07.
3. **The reduction order is a tree**, matching `__shfl_down_sync`, tested
   structurally and shown to differ from a sequential sum on a concrete input.

## Declared tolerance

`max_abs: 1e-4` on probabilities. The measured worst case on the RTX 4090 is
**1.835e-07**, so the declared bound has 545x of headroom and a failure is a real
failure rather than tolerance noise. Declared in `scoring.backend.json`.

## Reproducing

On a machine with `nvcc` and an NVIDIA GPU:

```sh
src/backends/cuda/scoring/build.sh ./out 89   # or CUDA_ARCH_LIST="80 89 90"
python3 src/backends/cuda/scoring/gpu_parity.py --library ./out/libscoring.so
python3 src/backends/cuda/scoring/test_scoring.py   # the reference, on CPU
```

The inputs are not committed. They are deterministic from a fixed seed, so the
harness regenerates them on every run; `reference.py --json vectors.json` dumps
them if a run needs to be archived. Committing a megabyte of generated floats to
check a kernel that regenerates them is weight without evidence.

`gpu_parity.py` skips rather than fails when there is no library or no device, so
a machine without a GPU reports that it could not check, not that it passed.

## Why not a generic kernel

The two models disagree on details that a generic tensor abstraction would have
to hide and then re-expose: whether there is a projection before the similarity,
whether the norm floor applies to the query, the row, or both, and what the
per-question candidate count is. This kernel takes the *shape* they share and
leaves the projections to the caller, which is where `#6` puts kernel selection
anyway — with the model engine.
