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

Recipes contain setup, launch commands and examples. Reusable implementation code
belongs under `src/`.
