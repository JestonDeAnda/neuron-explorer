# MoE Huginn Attribution Design

## Goal

Compute the same four teacher-forced Cognition23 attribution graphs already
produced for Huginn-0125 for the xLLM-native
`IFM/LoopedLM-P1-moe-huginn-500b` checkpoint. Store all 5,844 JSON graphs and
SVGs by the existing 23 task categories, with enough routing and recurrence
metadata to compare expert-neuron functions across graphs.

## Fixed Inputs

- Checkpoint:
  `/mnt/oss-save/modelscope/IFM/LoopedLM-P1-moe-huginn-500b`
- Hub revision: `f1321d5272a11348d5093ad7ab1d310eb9de2e11`
- xLLM-Loop source:
  `755c7b3a91081fe0f83500eabd53bd2490b8b745`
- Dataset:
  `/mnt/oss-save/wangjunqi/neuron-explorer/cognition23_original_text_20261110/按任务`
- DLC task ID and name prefix: `fs0278-xxx-attr-moehuginn500b`
- Resources: Zhongwei, Priority 1, one L20D, one pod, 8 CPU, 32 GiB RAM,
  8 GiB shared memory unless a measured runtime requirement proves this
  insufficient.

The checkpoint and all four BF16 shards have already passed the artifact
manifest SHA-256 checks. The source tree is not included in the checkpoint and
must be staged separately at the exact commit above.

## Executed Model

Native loading constructs the non-hierarchical xLLM `Huginn` class. Although
the configuration contains H=2 and L=3 counters, `huginn_hierarchical_state`
is `none`, so those counters do not control execution. The actual schedule is:

1. physical prelude layers 0–7 once;
2. diagonal input injection followed by physical recurrent layers 8–19,
   repeated eight times;
3. physical coda layers 20–27 once.

This is 112 virtual MoE depths: 8 prelude, 96 recurrent, and 8 coda.
Every physical layer is a top-2 MoE block with 25 experts and 2,432 SwiGLU
neurons per expert. The router applies sigmoid to 25 logits, chooses the two
largest scores, and renormalizes those two scores to sum to one.

## Scientific Semantics

The graph target is the summed teacher-forced completion log-probability under
the fixed prompt-token context. Prompt and continuation tokenization remains
separate, matching the Huginn-0125 protocol.

The production adapter uses a local linearization:

- freeze every RMSNorm inverse standard deviation;
- replace FlashAttention during attribution with the mathematically equivalent
  explicit causal attention pattern and freeze that pattern;
- freeze top-2 expert IDs and their renormalized routing weights;
- treat each routed `SiLU(gate) * up` expert activation as a leaf;
- use the same half-rule product linearization as Huginn-0125 for total-effect
  candidate ranking;
- keep RoPE, output projections, diagonal recurrent injection, residual
  additions, and expert output projections linear;
- treat embeddings and the deterministic row-0 recurrent state as source
  leaves.

The user selected frozen-router attribution. Router choices and weights are
recorded as metadata but are not causal graph nodes and do not receive
attribution edges.

## Implementation Approach

Patch the plain native xLLM modules in place after strict checkpoint loading.
Do not reimplement the production model. Native loading already disables fused
training blocks, so the adapter can patch ordinary RMSNorm, attention, router,
and expert forwards while retaining native weights, recurrence, RoPE,
injection, residuals, and grouped expert GEMMs.

A separate tiny pure-PyTorch reference model exists only for CPU unit tests of
the schedule, routing algebra, expert identity, frozen nonlinearities, and
half-rule. It is not used to load or attribute the real checkpoint.

The adapter must fail closed if an expected xLLM class, parameter shape,
execution count, routing invariant, or source commit differs.

## General MoE Boundary

The reusable part of MoE attribution is not model loading or module patching.
Those details differ between xLLM, Mixtral-style Transformers, DeepSeek MoE,
expert-parallel runtimes, and recurrent models. The reusable boundary is the
data contract between a runtime-specific linearization backend and graph
construction.

Create `src/neuron_explorer/attribution/moe/`:

- `types.py`: immutable `MoEExecutionSlot`, `RoutedNeuron`, `RoutingSnapshot`,
  `MoEContributions`, and `MoEModelIdentity` records.
- `backend.py`: a `MoELinearizationBackend` protocol exposing schedule,
  routed leaves, total/direct contributions, replay, completion target, and
  runtime identity without importing xLLM.
- `graph.py`: backend-independent candidate ranking, direct edges, influence
  pruning, expert-neuron serialization, and schema invariants.
- `reference.py`: dependency-free tiny PyTorch backends used to prove the
  generic graph layer does not assume top-2 sigmoid routing. Tests include one
  top-2 sigmoid backend and one softmax backend with a different expert count
  and top-k value.

The protocol explicitly parameterizes:

- router activation and normalization labels;
- selected expert count, which may differ by token or layer;
- shared experts versus routed experts;
- expert-local width and global serialized neuron identity;
- physical layer, virtual execution, recurrence iteration, and parameter
  sharing;
- runtime-specific source leaves and auxiliary metadata;
- single-device versus expert-parallel capability declarations.

The generic layer never assumes xLLM tensor names, 25 experts, top-2 routing,
sigmoid scores, grouped GEMM, recurrence, or FlashAttention.

Create `src/neuron_explorer/attribution/moe/xllm_huginn/`:

- `schedule.py`: 112 virtual execution slots and expert shared-weight IDs.
- `runtime.py`: pinned xLLM source/runtime checks and strict native loading.
- `linearized.py`: in-place patches, replay cache, frozen patterns/routes,
  expert leaves, contributions, and completion targets; implements
  `MoELinearizationBackend`.
- `runner.py`: benchmark, resumable 23-task batch, progress, and launch status.
- `mount.py`: checkpoint, source, extension, dataset, and mount validation.

Add:

- `scripts/run_moe_huginn_attribution.py`
- `scripts/moe_huginn_mount_check.py`
- `tests/attribution/test_moe_huginn.py`

The existing Huginn batch planner and four-condition protocol are reused. Its
resume helper will accept an explicit implementation version whose default is
the existing dense version 2, so unchanged callers serialize byte-compatible
metadata while MoE graphs explicitly request version 3.

This release implements one production backend, xLLM Huginn. Adding another
real MoE family is outside this run; the two tiny reference backends are the
portability proof for the generic contract.

## Dense Huginn Compatibility

The validated Huginn-0125 adapter is a protected compatibility surface:

- do not move, rewrite, or make it implement the new MoE protocol;
- do not change its default implementation version, graph configuration,
  node fields, checkpoint fingerprint, tokenizer protocol, schedule, renderer
  behavior, or CLI;
- shared batch helpers may gain keyword-only parameters only when all existing
  calls retain their old defaults;
- the complete existing Huginn test suite must pass before every MoE task is
  accepted;
- add explicit regression tests that a dense graph still records version 2,
  resumes with the old metadata contract, and rejects version-3 expectations;
- a real local Raven tiny-model forward/linearization test remains enabled and
  cannot be replaced by a mock.

## Node and Metadata Schema

Expert neurons retain `kind="neuron"` so the generic graph schema and SVG
renderer continue to work. Each expert-neuron node contains:

- `virtual_depth`, `stage`, `recurrence_iteration`, `physical_layer`,
  and `physical_block`;
- `expert_id` in 0–24 and `expert_neuron` in 0–2431;
- global `n = expert_id * 2432 + expert_neuron`, which is unique within one
  physical MoE layer;
- `router_rank` in `{0, 1}` and frozen `routing_weight`;
- `shared_weight_id = "<stage>:<block>:e<expert>:n<expert_neuron>"`;
- activation, gate, influence, residual `other`, and token position.

The same physical expert neuron shares parameters across recurrent
iterations, while `virtual_depth` distinguishes each execution. Expert-local
neuron index is not compared across different experts without the expert ID.

Top-level metadata records checkpoint fingerprint, xLLM source commit,
implementation version 3, schedule, tokenizer protocol, graph configuration,
route-freezing semantics, benchmark settings, and code commit.

## Data Flow

For each dataset row:

1. plan clean/correct, clean/incorrect, corrupted/correct, and
   corrupted/incorrect jobs with one shared row seed;
2. encode fixed prompt tokens and continuation tokens separately and prepend
   native BOS according to the dataset protocol;
3. run total-effect mode once and rank the top `k` routed expert-neuron
   executions;
4. replay the same row-0 recurrent state, frozen RMS denominators, attention
   patterns, routes, and expert activations across direct-edge copies;
5. compute raw signed source-to-target edges, prune by path influence, and
   serialize version-3 JSON atomically;
6. render SVG on CPU and append the manifest only after JSON validation.

Resume accepts a graph only when implementation version, token IDs, row seed,
checkpoint fingerprint, xLLM source commit, project commit, schedule, and full
graph configuration match.

## Correctness Gates

### CPU tests

- schedule is exactly 8 + 8×12 + 8 virtual depths;
- H/L counters do not alter the schedule when hierarchical state is `none`;
- top-2 sigmoid routing renormalizes to one and excludes unselected experts;
- expert IDs are part of neuron identity;
- deterministic state and edge copies use row-0 state;
- forward half-rule values equal the original SwiGLU values;
- frozen RMS, attention, and routes give the expected direct contribution
  identity on the tiny reference;
- edge serialization stays source → destination;
- version-2 dense graphs cannot satisfy version-3 MoE resume expectations.

### Real GPU smoke

Before any full run:

- compile and import `xllm_extension` against the exact DLC image;
- strict-load all 286 checkpoint tensors;
- verify tokenizer length 250,624, BOS 0, and checkpoint fingerprint;
- compare unpatched native logits, explicit-attention control logits, and
  patched forward values on a short row with a measured BF16 tolerance;
- observe exactly 112 MoE executions and eight injections;
- verify two selected routes per token per depth, valid IDs, and route weights
  summing to one;
- verify recurrent shared IDs are stable across all eight invocations;
- build and validate one small graph, including signed edge orientation;
- record peak VRAM and seconds per graph.

Any failed gate stops before the full GPU run.

## Runtime and DLC Sequence

The NeMo 26.06 image contains PyTorch, CUDA, safetensors, transformers, and
FlashAttention 2, but not `xllm_extension` or FlashAttention 3. Stage the pinned
xLLM source and compile its extension in a bounded CPU DLC mount/runtime check
using the same image. The attribution path uses explicit attention, so
FlashAttention 3 is not required for its backward graph; when available, it is
only an additional native-logit control. Absence of FlashAttention 3 must be
recorded as a scientific caveat, not silently ignored.

Use this sequence:

1. CPU runtime/mount check: `fs0278-xxx-attr-moehuginn500b-01`.
2. One-row GPU smoke: `fs0278-xxx-attr-moehuginn500b-02`.
3. Full resumable batch only after smoke approval:
   `fs0278-xxx-attr-moehuginn500b-03`.
4. Continuations increment the sequence and are launched only after the
   previous job reaches a terminal state.

The full job starts with a real median-length benchmark at `k=512` and
`max_edges=3000`. It keeps one resident model and chooses edge-copy batch size
from measured peak memory. It never launches parallel model copies merely to
consume nominal VRAM.

## Outputs

Durable root:

`/mnt/oss-save/wangjunqi/neuron-explorer/attribution/moe-huginn-500b`

Graphs:

- `output/json/<task>/<sample>__<condition>.json`
- `output/svg/<task>/<sample>__<condition>.svg`

Indexes and evidence:

- `output/manifest.jsonl`
- `output/progress.json`
- `output/summary.json`
- `output/benchmark.json`
- runtime-check, smoke-test, launch-status, and terminal-validation JSON files
- logs containing extension build identity, xLLM commit, model fingerprint,
  native/linearized tolerances, resource profile, and failures.

## Failure Handling

- A runtime or source mismatch fails before checkpoint loading.
- A native/linearized equivalence failure aborts the smoke.
- OOM first reduces direct-edge copy batch, then candidate count only if the
  reduced fidelity is explicitly recorded.
- One graph failure is recorded without corrupting completed JSON, but a full
  job exits nonzero when any condition fails.
- JSON is authoritative; missing SVGs can be regenerated without GPU.
- No second GPU job for the same stage/content runs concurrently.

