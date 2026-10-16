# OpenVLA H100 CPU/CUDA drift investigation

This is an investigation, **not a verified CUDA kernel fix**. The component
diagnostic below does not replace an independent original-model reference or
reproduce the complete three-component qualification by itself.

## Observed full-model discrepancy

The FP32 ONNX OpenVLA package exported by
[`rui-ren/mobius@f049872`](https://github.com/rui-ren/mobius/tree/f049872a1e240a360e9120e606abf5fe2a75c627) runs through
ORT 1.31.0 (`402979ac1`) on one H100 80 GB HBM3, SM90, Linux. TF32 is disabled
and graph optimizations are `ORT_ENABLE_ALL`. Against the pinned original CPU
fixture, eight of 8,849,664 prefill logits exceed the combined
`abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)` criterion.
Maximum absolute error is `0.00018551945686340332`.

Separate full-model experiments produced the following observations. Their
weights, request tensors and independent fixture are not bundled with this PR;
the component diagnostic below does not independently reproduce these counts.

| Control | Prefill failures against unchanged original CPU fixture |
|---|---:|
| Previously tested prebuilt ORT 1.31.0 CUDA runtime | 8 |
| Source-built ORT 1.32.0, CUDA 12.8 (details below) | 6 |
| Unchanged original-class Torch CUDA FP32, TF32 disabled | 7 |
| Vision MATH attention instead of efficient attention | 4 |
| Source Q-prescaling plus primitive attention | 3 |
| Frozen source embeddings supplied to the decoder | 0 |

The eight failing coordinates are `[0,14,31913]`, `[0,49,31815]`,
`[0,49,31849]`, `[0,119,31799]`, `[0,122,31809]`, `[0,122,31817]`,
`[0,243,31854]` and `[0,254,31936]`. They are image-token positions.
Matching the seven final actions does not clear the all-prefill numerical gate.

This does not establish an ORT-specific defect: original Torch CUDA also differs
from the frozen CPU fixture. It motivates locating the first vision divergence
with behavior-preserving captures and then comparing identical operator operands.

Neither fused `SkipLayerNormalization` nor `EmbedLayerNormalization` from
[microsoft/onnxruntime#33357](https://github.com/microsoft/onnxruntime/pull/33357)
executes in the observed package. It has 98 distinct ordinary vision LayerNorms
and 65 decoder RMSNorms. Replaying 102 original-source LayerNorm operand sets
through ordinary CUDA LayerNormalization passes the same combined arithmetic
bounds against centered FP64 calculations (maximum absolute error
`1.6502302599974428e-05`). These are source operands, not qualified captures from
the ORT candidate. Final-projection local rounding at the failed coordinates is
at most `2.6945027893354734e-06`, too small to explain the full discrepancy.
FP64 was only an offline arithmetic oracle, not an inference substitution.

## Component diagnostic

Use a GPU-enabled ORT environment with NumPy. Select and coordinate exclusive
use of one otherwise idle physical GPU externally; this script is not a resource
allocator. It neither installs drivers nor silently switches to another device.

```bash
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
python onnxruntime/python/tools/transformers/models/openvla/compare_provider_outputs.py \
  --model /absolute/package/vision_encoder/model.onnx \
  --inputs /absolute/vision-inputs.npz \
  --output-dir /absolute/fresh-provider-comparison
```

The NPZ contains exactly the graph's input names, with their original dtypes
and shapes, captured from the request being investigated. Pickle is disabled.
External initializer files must remain alongside the supplied model with their
original relative paths. Pin their hashes separately: the reported graph hash
explicitly excludes external tensor data. No graph or weights are modified.

The script runs the same inputs through CPU and CUDA sessions, with TF32
disabled and `ORT_ENABLE_ALL`, and saves named-output mappings to separate NPY
files, effective provider options, native build information and profile hashes.
CPU is a diagnostic comparator, **not an independent source oracle**.
CPU-placed nodes are explicitly listed for review; inspect their input/output
types to distinguish integer shape work from floating-point fallback.
Unavailable CUDA, initialization fallback or absence of CUDA kernels fails
explicitly. Runtime fallback is disabled.

To inspect intermediates, separately prepare an identical graph with selected
additional outputs, for example using existing activation-debugging tools, and
pass `--instrumented-model /absolute/instrumented/model.onnx`.
The script first requires **every original output to remain bitwise identical
within each provider**. Even a one-ULP or signed-zero change rejects the capture;
instrumentation can change optimization and fusion. Additional comparisons are
suppressed on rejection. Captured output order is the graph's declared output
order, not a proven topological first-failure order.

Exit codes: **0** means the requested diagnostic comparisons are within their
bounds; **2** means comparisons exceed those bounds, not that a kernel defect
is established; **3** rejects behavior-changing instrumentation. Other input or
runtime errors raise explicitly. Reports never confer model qualification.
The default floating-point diagnostic bounds are `atol=rtol=1e-4`; integer and bool
outputs compare exactly. These CLI options do not
change any external frozen qualification policy.

Reproducing the full-model discrepancies requires the pinned three-component
package, unchanged independent fixture and original own-path prefill sequence;
the component command alone does not reproduce these counts. The unresolved work is a
small matched-input kernel reproducer, not an unverified normalization patch.
Model weights, input tensors and native profiles are not bundled here.
The sanitized [component observation](h100_observation.json) records an actual
run of this tool: all original outputs stayed bitwise unchanged under four
additional taps, and all five CPU/CUDA output comparisons passed. The input to
the first tapped LayerNorm already differed between providers, so those captures
do not establish a normalization defect.

Generated local reports contain file paths and may contain device identifiers.
Review and redact them before publishing. The committed observation excludes
local paths, machine identifiers, weights and input tensors.

## Source-built development baseline

The branch was built at `54be849eb644691d089188b67c0b43e7494d07c8` and its
local GPU wheel was used for the same full-model comparison, not just the
vision-component probe. The [sanitized source-build observation](source_build_observation.json)
records the wheel/native-library hashes, build settings and numerical results.
Later documentation commits are not the compiled source revision.

The build used an isolated CUDA 12.8.93 SDK, CUDA runtime 12.8.90,
cuBLAS 12.8.4.1, cuDNN 9.5.1.17, GCC 11.4.0 and Python 3.12.15, targeting
H100 `90a-real`. CUDA 12.6 did not compile this revision's cuDNN frontend,
which references CUDA 12.8 runtime APIs. Unused FP4-QMoE was disabled;
the FP32 model graphs and numerical policy were unchanged. Explicit
`--cuda_version 12.8` supplied wheel metadata because the isolated SDK
did not contain `version.json`.

The full comparison still **fails qualification**: six prefill values exceed
the unchanged combined `atol=rtol=1e-4` bound, with maximum absolute error
`0.00018596649169921875`. The other 132 of 133 boundaries pass, including all
128 cache comparisons. Seven exact action tokens, actions and a fresh-request
repeat match the original CPU fixture. Placement inspection finds no
floating-point CPU fallback; the CPU nodes perform integer shape work.
The 13 diagnostic unit tests also pass against this wheel.

This is not a controlled kernel-fix experiment: the native revision, ORT
version, CUDA toolchain/libraries and build configuration differ from the
previous prebuilt runtime. Fewer failing values do not establish a fix or its
cause. The original CPU reference was not replaced, no tolerance was relaxed,
and neither all-node GPU qualification nor production admission was granted.

### What the failing gate means

This is a development acceptance rule, not an ORT execution error or a
robot-task accuracy metric. A logit is a raw token score; the gate compares
every initial image/instruction ("prefill") score to the original CPU fixture.
One score outside the combined bound fails the gate even when actions match.
For example, reference `-0.0393329859` versus candidate `-0.0391947106` differs
by `0.0001382753`, above its allowed `0.0001039333`.

A follow-up captured existing outputs without adding graph outputs and
reproduced all 133 comparison metrics. All six failures are at image-token
positions. The highest-scoring token is unchanged at all 276 positions;
the last position used to select the first action has zero failing scores.
Non-image embeddings are bitwise identical to the CPU fixture, so this
observation does not implicate the text-token embedding lookup.

With the same decoder, diagnostic interpolation from CPU image features
toward the candidate features produces 0, 0, 0, 1 and 6 failing scores at
0%, 25%, 50%, 75% and 100% of the feature difference. Endpoint embeddings
and logits match the previous captures bitwise. This supports upstream
feature-error sensitivity, not a proven faulty kernel. Interpolated/oracle
features are counterfactual controls, never an inference fix.

Vision features are also bitwise identical between the prebuilt and
source-built ORT runs: the eight-to-six count change is not a vision-path
fix. These are single-request observations, not proof that small errors
are harmless across other images or robot tasks.

## Tests

Using an environment with NumPy, ONNX and ORT installed, run outside the checkout
so its unbuilt `onnxruntime` package does not shadow the installed wheel:

```bash
cd /tmp
python -m unittest discover \
  -s /absolute/onnxruntime/onnxruntime/test/python/transformers \
  -p test_openvla_provider_drift.py -v
```

Tests include real CPU execution with external initializers and an output named
`file`, combined tolerance, exact large integer comparison, signed zero,
nonfinite/mismatched outputs and rejection of behavior-changing captures.
