# What fusing the candidate scoring is worth

`src/backends/cuda/scoring/` argues that a cosine needs each candidate's norm and
its dot with the query, that both read the same elements, and that accumulating
them in one pass over `candidates` is therefore cheaper than composing a
normalisation pass with a scoring pass. This measures it, and the measurement
changed the kernel.

## Protocol and controls

The hypothesis was that the fused kernel beats an unfused equivalent that reads
the candidate matrix twice. Stop conditions were any failed correctness check, a
missing library, or no GPU; a correctness failure aborts the run rather than
producing a number, because a kernel that computes the wrong thing can be
arbitrarily fast.

- One shared RTX 4090, 1 of 8 on the host, otherwise idle (0 % utilisation,
  30 °C) for the recorded run. The host is shared and its other GPUs carry
  unrelated work, so the **ratios** are the result here, not the absolute
  microseconds. The fused and unfused measurements are taken in the same
  process, on the same buffers, and were repeated with the order swapped to rule
  out a clock-ramp bias: the ratios moved by at most 0.01.
- Both paths are checked against `reference.py` first, per shape, at a `max_abs`
  of 1e-4. Both passed before anything was timed.
- Per shape: 30 warmup calls, then 300 calls timed individually with CUDA events;
  the median is reported.
- `bench.cu` is the baseline. It is `candidate_scoring.cu`'s own block with the
  row norm loaded from a staged array instead of accumulated beside the dot, plus
  a separate norm pass. The norm pass runs only when the similarity is a cosine,
  because a plain dot has no norm to compute and running it anyway would slow the
  baseline for a reason the fused kernel never removed.
- It is a **conservative** baseline: a real unfused pipeline built from a GEMM
  would also round-trip the similarity matrix through global memory, which this
  does not. The differences below are a floor.

```sh
src/backends/cuda/scoring/build.sh ./out 89
python3 src/backends/cuda/scoring/bench.py --library ./out/libscoring.so --iterations 300 \
    --out docs/benchmarks/scoring-fusion/results.json
```

## The finding: the kernel was occupancy-bound, not bandwidth-bound

At the block size the kernel was written with, 256 threads, fusion **lost** at
every shape that mattered:

| shape | similarity | K | D | questions | fused | unfused | ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clm-cosine-5x512` | cosine | 5 | 512 | 1 | 9.2 µs | 11.3 µs | 1.23× |
| `clm-cosine-64x512` | cosine | 64 | 512 | 1 | 19.4 µs | 19.3 µs | 1.00× |
| `clm-cosine-255x512` | cosine | 255 | 512 | 1 | 52.9 µs | 43.2 µs | **0.82×** |
| `clm-cosine-batch8-255x512` | cosine | 255 | 512 | 8 | 53.1 µs | 44.0 µs | **0.83×** |
| `kev-dot-255x2560` | dot | 255 | 2560 | 1 | 167.8 µs | 96.1 µs | **0.57×** |

Two things said why. The achieved bandwidth was 1 to 190 GB/s against a device
peak near 1000, so nothing was bandwidth-bound and there was no read to save. And
a single question and a batch of eight took **the same 53 µs** — eight times the
work in the same wall clock, which is only possible if each question is running
on its own SM. One thread block per question means a one-question request, the
common serving case, occupies **one of the 128 SMs**, and the norm accumulation
sat on that block's critical path instead of being spread across the device.

Raising the block to 1024 threads — the maximum on sm_89, and one line — moves
the work onto 32 warps instead of 8 and reverses every result:

| shape | similarity | K | D | questions | fused | unfused | ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clm-cosine-5x512` | cosine | 5 | 512 | 1 | 9.2 µs | 11.3 µs | 1.22× |
| `clm-cosine-64x512` | cosine | 64 | 512 | 1 | 12.1 µs | 18.3 µs | 1.51× |
| `clm-cosine-255x512` | cosine | 255 | 512 | 1 | 23.6 µs | 42.2 µs | **1.79×** |
| `clm-cosine-batch8-255x512` | cosine | 255 | 512 | 8 | 23.8 µs | 43.8 µs | **1.84×** |
| `kev-dot-255x2560` | dot | 255 | 2560 | 1 | 55.3 µs | 100.3 µs | **1.81×** |

Kev is the control. It is a plain dot, so there is no norm for the fusion to
remove from the traffic, and its gain is entirely the second kernel and the
global similarity round-trip it does not need. That it lands at the same ~1.8× as
the cosine shapes is the useful part of the result.

Correctness is unchanged by the block size: the parity harness reports the same
`1.835e-07` worst case, and `compute-sanitizer --tool memcheck` reports
`ERROR SUMMARY: 0 errors`.

## Limits

- One GPU, one architecture: sm_89 only. The library builds for whatever
  `build.sh` is given, but nothing else has been run.
- The tiny shapes are launch-bound, not compute-bound — `clm-cosine-5x512` is 9 µs
  either way. The interesting range starts around K = 64.
- CUDA Graph replay is not timed here. It removes launch overhead, which matters
  most exactly where these numbers are worst, so a graph-timed run would flatter
  the fused kernel; that is a separate measurement.
- No end-to-end number. This is the kernel, with the projections left to the
  caller as the design intends.
