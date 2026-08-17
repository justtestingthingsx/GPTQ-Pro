# Quant-path review base (deliberately empty)

Empty base branch: the companion PR from `quant-path-review` diffs against
this, so the PR's diff IS the complete code path our Qwen3.8-27B
quantization arms execute in this fork (solvers, calibration loop, model
definitions, packing/serialization, drivers, and the house Qronos
additions). Runtime/inference kernels are excluded on purpose — artifacts
are served on vLLM, not on this repo's runtime.
