# Legacy merged readout regression oracle

`legacy_merged_readout.json` keeps four requests from
[`benchmarks/e2e/choice.jsonl` at `86926064`](https://github.com/ThinkFlowLab/system1-omni/blob/869260649f7dd5b56cac4d91eaae9087f2e985ee/benchmarks/e2e/choice.jsonl).
The original corpus SHA-256 is recorded in the fixture. Its JevBench-derived
request material is MIT licensed; the original copyright and full terms are
preserved in [`LICENSE.jevbench`](LICENSE.jevbench).

The baseline responses come from A800/ABI-8 parent `5bdd625b`, using the same
legacy merged artifact as candidate `7293c71c`. The exact source, model, artifact
and environment hashes are in the
[public raw-evidence archive](https://github.com/Levius-Fubuki/system1-omni/releases/tag/cua-a800-abi8-20261015):
`evidence/pr143/merged-before/pass0.jsonl` supplies each `expected_response`,
`evidence/pr143/merged-after/pass0.jsonl` supplies `rounded_choice`, and
`evidence/pr143/native-eager/manifest.jsonl` supplies the requests. Both measured
passes produced identical response bytes. Only the four cases whose decisions
changed are retained here as a maintained compatibility oracle. Expected
responses preserve the parent behavior; they are **not task ground truth**.

Consumers:

- `tests/cua_s1/head.rs` uses the actual option order with explicitly synthetic
  BF16 projection vectors to test the CPU readout/response boundary in both modes.
- `tests/cua_s1/legacy_merged_gpu.rs` runs the actual frozen requests twice through
  the legacy model after verifying every recorded artifact file's size and
  SHA-256, requiring baseline decisions, probability error at
  most 0.02, identity/usage/order and exact repeat bytes. Run Graph 0/1 in separate
  processes as documented in `recipe/cua_s1/native.md`.

The GPU oracle is specific to the recorded A800 artifact/environment. Compilation
or a passing synthetic CPU test does not establish a new real-model GPU pass.
