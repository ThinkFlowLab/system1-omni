# Recipes

- [Laya text worker](laya/README.md): start the external Python worker, connect the
  Rust frontend and compare direct and proxied responses.
- [Laya on Apple Silicon](laya/apple-silicon.md): serve Laya on the Mac GPU with the Laya
  worker, put the frontend in front of it and run the benchmarks.
- [Cua-S1 4B 0.2 text worker](cua_s1/text.md): download the pinned weights, start
  the worker and connect the Rust frontend.
- [Cua-S1 4B 0.2 native text worker](cua_s1/native.md): build the CUDA library and
  the Rust worker, export the merged weights and start the worker.
- [Open-Jev-27B-v1.1 native text worker](open_jev/native.md): export the merged
  text backbone and trained decision head, then serve with Rust and CUDA.
- [CLM behind the frontend](clm/README.md): run CLM's own server behind the frontend on
  CPU with a stub encoder, and what the response comparison has to allow for.

- [LFM2.5-350M choice worker](lfm2/README.md): score text decisions with independent
  questions, hybrid cache branching and configurable candidate batches.

Recipes contain setup, launch commands and examples. Reusable implementation code
belongs under `src/`.

[`compare_with_backend.py`](compare_with_backend.py) checks that the frontend returns what
the worker returned, for any recipe; [`test_compare_with_backend.py`](test_compare_with_backend.py)
covers it without a model.
