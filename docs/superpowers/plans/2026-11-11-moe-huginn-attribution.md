# MoE Huginn Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable MoE attribution core and a scientifically verified
xLLM Huginn backend, then compute all 5,844 Cognition23 graphs on one L20D
without changing the existing dense Huginn-0125 behavior.

**Architecture:** A runtime-independent MoE protocol supplies routed expert
leaves and contribution tensors to a generic graph builder. The xLLM Huginn
backend patches the native plain modules to freeze RMS denominators, explicit
attention patterns, and top-k routing while retaining native checkpoint
loading, recurrence, expert grouped GEMMs, and output projections. Existing
Huginn batch helpers gain only keyword-only compatibility parameters whose
defaults preserve version-2 output.

**Tech Stack:** Python 3.12/3.14, PyTorch, xLLM-Loop commit
`755c7b3a91081fe0f83500eabd53bd2490b8b745`, safetensors, pytest, Alibaba PAI
DLC, one Zhongwei L20D.

## Global Constraints

- Keep `src/neuron_explorer/attribution/huginn/` behavior and default graph
  metadata unchanged; all 117 pre-existing tests must remain green.
- MoE graphs use attribution implementation version 3; dense Huginn remains
  version 2.
- Freeze router expert IDs and routing weights. Router values are metadata, not
  causal graph nodes.
- The generic MoE layer must not import xLLM or assume top-2, sigmoid routing,
  25 experts, recurrence, grouped GEMM, or FlashAttention.
- Production xLLM source is pinned to
  `755c7b3a91081fe0f83500eabd53bd2490b8b745`.
- Model revision is the checkpoint artifact fingerprint; source revision and
  project commit are separate required resume identities.
- Every production-code step follows red-green-refactor.
- No full GPU batch starts before CPU tests, CPU runtime/mount check, and a
  real one-row GPU equivalence/completeness smoke all pass.
- DLC names use `fs0278-xxx-attr-moehuginn500b-01`, `-02`, `-03`, then unique
  increasing continuation sequences. Priority is 1 and GPU count is 1.

---

### Task 1: Versioned Batch Compatibility

**Files:**
- Modify: `src/neuron_explorer/attribution/huginn/batch.py`
- Modify: `tests/attribution/test_huginn.py`

**Interfaces:**
- Consumes: existing `run_batch`, `_attach_metadata`, and `_expected_graph`.
- Produces:
  `run_batch` with keyword-only `attribution_implementation_version: int = 2`
  and
  extra_metadata: Mapping[str, Any] | None = None)`.

- [ ] **Step 1: Write failing dense-compatibility and version-3 tests**

Add tests that run the existing `_row()` fixture twice:

```python
def _run_versioned_batch(tmp_path, **version_options):
    return run_batch(
        [_row()],
        "01_指代性别一致",
        tmp_path,
        _Tokenizer(),
        lambda _job: _graph(),
        code_commit="deadbeef",
        model_path="/models/huginn-0125",
        model_revision="checkpoint-sha256:abc",
        recurrence_count=32,
        graph_config={
            "k": 512,
            "max_edges": 3000,
            "target": "teacher_forced_completion_logprob",
        },
        **version_options,
    )


def test_run_batch_keeps_dense_version_two_by_default(tmp_path):
    _run_versioned_batch(tmp_path)
    graph = load_graph(
        graph_paths(
            tmp_path, "01_指代性别一致", _row()["id"], "clean_correct"
        )[0]
    )
    assert graph["metadata"]["attribution_implementation_version"] == 2


def test_run_batch_accepts_explicit_moe_version_and_identity(tmp_path):
    _run_versioned_batch(
        tmp_path,
        attribution_implementation_version=3,
        extra_metadata={"runtime": "xllm", "source_commit": "755c7b3"},
    )
    graph = load_graph(
        graph_paths(
            tmp_path, "01_指代性别一致", _row()["id"], "clean_correct"
        )[0]
    )
    assert graph["metadata"]["attribution_implementation_version"] == 3
    assert graph["metadata"]["runtime"] == "xllm"
    assert graph["metadata"]["source_commit"] == "755c7b3"
```

Also assert the version-3 graph is rebuilt when `source_commit` changes and
that an unchanged dense graph still resumes without invoking `build`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest tests/attribution/test_huginn.py \
  -k 'version_two_by_default or explicit_moe_version_and_identity' -q
```

Expected: `TypeError` for the new keyword arguments.

- [ ] **Step 3: Add keyword-only compatibility parameters**

Thread `attribution_implementation_version=2` and `extra_metadata=None` through
`run_batch`, `_attach_metadata`, and `_expected_graph`. Merge extra metadata
after standard metadata, but reject attempts to override:

```python
reserved = {
    "attribution_implementation_version", "seed", "model_path",
    "model_revision", "recurrence_count", "prompt_protocol",
    "code_commit", "graph_config", "condition", "row_id", "task_slug",
}
overlap = reserved.intersection(extra_metadata or {})
if overlap:
    raise ValueError(f"extra metadata overrides reserved fields: {sorted(overlap)}")
```

Use the explicit version in both attached and expected metadata. Do not change
the module constant or any existing caller.

- [ ] **Step 4: Verify focused and complete dense suites**

Run:

```bash
uv run pytest tests/attribution/test_huginn.py -q
uv run pytest -q
```

Expected: all existing tests plus the new tests pass; the real local Raven
tests are not skipped when the cached checkpoint is present.

- [ ] **Step 5: Commit**

```bash
git add src/neuron_explorer/attribution/huginn/batch.py \
  tests/attribution/test_huginn.py
git commit -m "refactor: version attribution batch metadata safely"
```

---

### Task 2: Generic MoE Contracts and Reference Backends

**Files:**
- Create: `src/neuron_explorer/attribution/moe/__init__.py`
- Create: `src/neuron_explorer/attribution/moe/types.py`
- Create: `src/neuron_explorer/attribution/moe/backend.py`
- Create: `src/neuron_explorer/attribution/moe/reference.py`
- Create: `src/neuron_explorer/attribution/moe/graph.py`
- Create: `tests/attribution/test_moe_core.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `MoEExecutionSlot`
  - `RoutedNeuron`
  - `RoutingSnapshot`
  - `MoEContributions`
  - `MoEModelIdentity`
  - `MoELinearizationBackend`
  - `build_moe_graph(backend, token_ids, completion_start, **config)`

- [ ] **Step 1: Write failing type, routing, and graph tests**

Define desired construction in tests:

```python
slot = MoEExecutionSlot(
    virtual_depth=3,
    stage="recurrent",
    recurrence_iteration=1,
    physical_layer=7,
    physical_block=0,
)
neuron = RoutedNeuron(
    slot=slot,
    position=4,
    expert_id=2,
    expert_neuron=5,
    expert_width=16,
    router_rank=1,
    routing_weight=0.25,
)
assert neuron.serialized_neuron == 37
assert neuron.shared_weight_id == "recurrent:0:e2:n5"
```

Add two tiny backends:

1. four experts, top-2 sigmoid-renormalized routing;
2. three experts, top-1 softmax routing.

For both, call `build_moe_graph` with `k=4`, `batch=2`, and `max_edges=20`
and assert:

- generic code never reads a backend-specific router type;
- node `kind` is `"neuron"`;
- global `n`, `expert_id`, and `expert_neuron` are coherent;
- edges are source → destination and finite;
- schedule and model identity reach metadata;
- unselected experts never appear;
- selected route weights satisfy each backend's declared invariant.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/attribution/test_moe_core.py -q
```

Expected: import failure for `neuron_explorer.attribution.moe`.

- [ ] **Step 3: Implement immutable records and protocol**

Use frozen dataclasses. The backend protocol must expose only runtime-neutral
operations:

```python
class MoELinearizationBackend(Protocol):
    identity: MoEModelIdentity
    schedule: Sequence[MoEExecutionSlot]

    def forward(self, token_ids: Sequence[int], *, copies: int = 1,
                replay: bool = False, mode: str = "direct") -> torch.Tensor:
        raise NotImplementedError
    def routed_neurons(self) -> Sequence[RoutedNeuron]:
        raise NotImplementedError
    def candidate_effects(self, completion_start: int,
                          token_ids: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError
    def direct_targets(self, targets: Sequence[int], completion_start: int,
                       token_ids: Sequence[int]) -> MoEContributions:
        raise NotImplementedError
    def activations(self, neurons: Sequence[RoutedNeuron]) -> torch.Tensor:
        raise NotImplementedError
    def completion_logprob(self, row: int, completion_start: int,
                           token_ids: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError
```

`MoEContributions` contains a routed-neuron matrix, token-embedding matrix,
state vector, and scalar targets. It also carries the exact ordered
`RoutedNeuron` axis so graph code never infers expert mappings from tensor
offsets.

- [ ] **Step 4: Implement reference backends and generic graph**

The tiny reference uses eager PyTorch and deterministic fixed weights. Its
router strategy is injected as a callable, proving the graph builder does not
branch on sigmoid versus softmax or top-k.

Port influence pruning and edge serialization from the verified Huginn graph
without importing xLLM. Serialize expert nodes with:

```python
{
    "kind": "neuron",
    "layer": slot.virtual_depth,
    "virtual_depth": slot.virtual_depth,
    "stage": slot.stage,
    "recurrence_iteration": slot.recurrence_iteration,
    "physical_layer": slot.physical_layer,
    "physical_block": slot.physical_block,
    "pos": routed.position,
    "n": routed.serialized_neuron,
    "physical_neuron": routed.serialized_neuron,
    "expert_id": routed.expert_id,
    "expert_neuron": routed.expert_neuron,
    "router_rank": routed.router_rank,
    "routing_weight": routed.routing_weight,
    "shared_weight_id": routed.shared_weight_id,
}
```

- [ ] **Step 5: Verify genericity and dense compatibility**

Run:

```bash
uv run pytest tests/attribution/test_moe_core.py -q
uv run pytest tests/attribution/test_huginn.py -q
uv run pytest -q
```

Expected: both router fixtures pass and all dense tests remain green.

- [ ] **Step 6: Register and commit the package**

Add `"neuron_explorer.attribution.moe"` to setuptools packages.

```bash
git add pyproject.toml uv.lock src/neuron_explorer/attribution/moe \
  tests/attribution/test_moe_core.py
git commit -m "feat: add generic MoE attribution core"
```

---

### Task 3: xLLM Huginn Schedule, Runtime Identity, and Mount Checks

**Files:**
- Create: `src/neuron_explorer/attribution/moe/xllm_huginn/__init__.py`
- Create: `src/neuron_explorer/attribution/moe/xllm_huginn/schedule.py`
- Create: `src/neuron_explorer/attribution/moe/xllm_huginn/runtime.py`
- Create: `src/neuron_explorer/attribution/moe/xllm_huginn/mount.py`
- Create: `scripts/moe_huginn_mount_check.py`
- Modify: `tests/attribution/test_moe_huginn.py`
- Modify: `tests/scripts/test_cli_help.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `execution_schedule(config: Mapping[str, Any])`
  - `checkpoint_fingerprint(path: Path)`
  - `verify_xllm_source(path: Path, expected_commit: str)`
  - `load_xllm_huginn(model_path, source_path, device)`
  - `check_moe_mounts`

- [ ] **Step 1: Write failing schedule and identity tests**

Use the committed model config fixture or an inline minimal mapping. Assert:

```python
slots = execution_schedule(config)
assert len(slots) == 112
assert [s.virtual_depth for s in slots] == list(range(112))
assert [s.physical_layer for s in slots[:8]] == list(range(8))
assert [s.physical_layer for s in slots[8:20]] == list(range(8, 20))
assert slots[20].recurrence_iteration == 1
assert slots[104].stage == "coda"
```

Change H/L counters while keeping `hierarchical_state="none"` and assert the
schedule is unchanged. Change hierarchical state to a non-supported value and
assert a fail-closed `ValueError`.

For a temporary checkpoint with config, source Python, and safetensors files,
assert any byte change alters the fingerprint. For a temporary Git checkout,
assert a wrong xLLM commit is rejected.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/attribution/test_moe_huginn.py \
  -k 'schedule or fingerprint or source_commit' -q
```

Expected: package import failure.

- [ ] **Step 3: Implement schedule and lazy runtime loading**

Keep all xLLM imports inside `load_xllm_huginn` so local CPU tests do not need
the CUDA extension. Before import:

```python
verify_xllm_source(source_path, XLLM_SOURCE_COMMIT)
sys.path.insert(0, str(source_path))
```

Call `xllm.paper_part1.native_inference.load_native_model`, then verify:

- class name is `Huginn`;
- `num_layers == 28`;
- prelude/recurrent/coda are 8/12/8;
- loop times is 8 and hierarchical state is `none`;
- experts/top-k/width are 25/2/2432;
- tokenizer length is 250624 and BOS is 0;
- state dict loaded strictly.

Return tokenizer, model, mutated model config, fingerprint, and source commit.

- [ ] **Step 4: Implement mount/runtime validation**

Check dataset count 23/1461, staged project commit, xLLM source commit,
checkpoint files and manifest, `xllm_extension` import, grouped-mm presence,
and output-root writability. Emit a JSON report with individual checks and
nonzero exit on any failure.

- [ ] **Step 5: Add CLI help coverage and package registration**

Register:

- `"neuron_explorer.attribution.moe.xllm_huginn"`
- `scripts/moe_huginn_mount_check.py`

The script requires `--dataset`, `--model`, `--xllm-source`, `--stage`,
`--commit`, and `--output-root`.

- [ ] **Step 6: Verify and commit**

Run:

```bash
uv run pytest tests/attribution/test_moe_huginn.py \
  tests/scripts/test_cli_help.py -q
uv run pytest -q
```

Commit:

```bash
git add pyproject.toml uv.lock scripts/moe_huginn_mount_check.py \
  src/neuron_explorer/attribution/moe/xllm_huginn \
  tests/attribution/test_moe_huginn.py tests/scripts/test_cli_help.py
git commit -m "feat: add xLLM Huginn runtime contract"
```

---

### Task 4: xLLM Native Linearization Backend

**Files:**
- Create: `src/neuron_explorer/attribution/moe/xllm_huginn/linearized.py`
- Modify: `tests/attribution/test_moe_huginn.py`

**Interfaces:**
- Produces:
  `XLLMHuginnLinearized(model, config)`, implementing
  `MoELinearizationBackend`.

- [ ] **Step 1: Write failing pure-module patch tests**

Use tiny local modules with the same public attributes as xLLM modules, without
importing xLLM. Tests must prove:

- frozen RMS uses cached `rstd` and preserves forward values;
- explicit causal grouped-query attention preserves tensor shape, freezes the
  pattern, repeats six KV heads to 24 query heads correctly, and masks future
  positions;
- route snapshots preserve selected IDs, rank, original token row, and
  normalized weight;
- direct expert leaves map grouped expert rows back to
  `(batch, position, expert)` exactly;
- changing an unselected expert cannot change a frozen-route output;
- the half-rule forward equals native SwiGLU;
- replay copies one row-0 recurrent state across all edge rows;
- invocation counts map 8/96/8 calls to virtual depths 0–111;
- context-manager exit restores every patched forward after success and error.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/attribution/test_moe_huginn.py \
  -k 'rms or attention or route or expert or replay or restore' -q
```

Expected: import failure for `linearized`.

- [ ] **Step 3: Implement module patches**

Patch by runtime class discovered from the loaded model, not by importing
private xLLM classes at module import time.

RMS:

```python
features = x.float()
rstd = torch.rsqrt(features.pow(2).mean(-1, keepdim=True) + module.eps)
rstd = backend.frozen(module, "rstd", rstd)
return (features * rstd).to(x.dtype) * (module.weight + 1)
```

Attention computes FP32 scores, applies RoPE-provided q/k inputs, repeats KV
heads, masks causally, detaches the pattern, mixes V, then returns in the
native expected layout.

Router calls the native routing calculation once, records and detaches
`indices` and normalized weights, and reuses them on replay.

Expert patch retains native grouped matmuls. At the SwiGLU product, record
group offsets and the inverse permutation so each leaf has an exact original
row and expert. Direct mode detaches activation; total mode uses the half-rule
with a forward-value correction.

- [ ] **Step 4: Implement backend contribution API**

Collect one ordered routed-neuron axis per forward. Reject duplicate execution
IDs and inconsistent replay routes. Implement embedding and state leaves,
completion log-probability, total candidate effects, and batched direct
targets. Parameters remain frozen and are restored to their original
`requires_grad` values on exit.

- [ ] **Step 5: Verify focused, generic, and dense suites**

Run:

```bash
uv run pytest tests/attribution/test_moe_huginn.py -q
uv run pytest tests/attribution/test_moe_core.py -q
uv run pytest tests/attribution/test_huginn.py -q
uv run pytest -q
```

- [ ] **Step 6: Commit**

```bash
git add src/neuron_explorer/attribution/moe/xllm_huginn/linearized.py \
  tests/attribution/test_moe_huginn.py
git commit -m "feat: linearize xLLM Huginn MoE execution"
```

---

### Task 5: MoE Graph Runner, CLI, and Documentation

**Files:**
- Create: `src/neuron_explorer/attribution/moe/xllm_huginn/runner.py`
- Create: `scripts/run_moe_huginn_attribution.py`
- Modify: `src/neuron_explorer/attribution/moe/xllm_huginn/__init__.py`
- Modify: `tests/attribution/test_moe_huginn.py`
- Modify: `tests/scripts/test_cli_help.py`
- Modify: `docs/attribution.md`
- Modify: `docs/development.md`

**Interfaces:**
- Produces:
  - `build_xllm_huginn_graph`
  - `execute(args) -> int`
  - CLI `scripts/run_moe_huginn_attribution.py`

- [ ] **Step 1: Write failing graph and resume tests**

Using the tiny xLLM-shaped backend, assert:

- aggregate completion target covers every continuation token;
- expert node fields and shared IDs match the design;
- route metadata is present but no router node exists;
- version is 3 and runtime/source/schedule identity is required for resume;
- stale dense version-2 JSON is rebuilt;
- missing SVG is regenerated without invoking the backend;
- manifest has one unique record per task/row/condition;
- benchmark configuration records frozen routes, explicit attention, one model
  copy, edge-copy batch, and fidelity reductions.

- [ ] **Step 2: Verify RED**

Run:

```bash
uv run pytest tests/attribution/test_moe_huginn.py \
  -k 'graph or resume or manifest or benchmark' -q
```

Expected: missing runner/build functions.

- [ ] **Step 3: Implement graph wrapper and runner**

Call generic `build_moe_graph` with `XLLMHuginnLinearized`. Reuse
`plan_row`, `iter_task_rows`, `graph_paths`, and `run_batch`, explicitly
passing:

```python
attribution_implementation_version=3
extra_metadata={
    "runtime": "xllm",
    "xllm_source_commit": XLLM_SOURCE_COMMIT,
    "schedule": schedule_metadata,
    "routing": {
        "frozen": True,
        "score": "sigmoid",
        "normalization": "selected_sum",
        "top_k": 2,
        "experts": 25,
        "expert_width": 2432,
    },
}
```

The benchmark first runs batch 1, measures peak memory, and caps direct-edge
copies at 4. OOM reduces copies before reducing `k`; any reduced `k` or
`max_edges` is recorded.

- [ ] **Step 4: Add CLI and documentation**

CLI arguments:

```text
--model --xllm-source --dataset --output-root --commit
--device --k --max-edges --benchmark-first --limit --smoke
```

Document:

- generic MoE backend contract;
- xLLM source and model revisions;
- frozen-route scientific scope;
- expert node identity;
- runtime build and smoke commands;
- output paths and programmatic grouping by
  `(shared_weight_id, position)` or `(physical_layer, expert_id,
  expert_neuron)`.

- [ ] **Step 5: Run all local checks**

Run:

```bash
uv run pytest -q
git diff --check
uv run python scripts/run_moe_huginn_attribution.py --help
uv run python scripts/moe_huginn_mount_check.py --help
```

Expected: all tests pass and both CLIs exit 0.

- [ ] **Step 6: Commit and push the feature branch**

```bash
git add scripts/run_moe_huginn_attribution.py \
  src/neuron_explorer/attribution/moe/xllm_huginn \
  tests docs pyproject.toml uv.lock
git commit -m "feat: run resumable xLLM MoE attribution"
git push -u origin feature/moe-huginn-attribution
```

---

### Task 6: DSW Stage and CPU Runtime Check

**Files:**
- Runtime artifact:
  `/mnt/oss-save/wangjunqi/neuron-explorer/attribution/moe-huginn-500b/runtime/`
- Stage:
  `/mnt/oss-save/wangjunqi/neuron-explorer/attribution/moe-huginn-500b/stage/<commit>/`
- Evidence:
  `runtime-check-01.json`

**Interfaces:**
- Consumes: pushed project commit, xLLM commit, checkpoint, dataset.
- Produces: importable compiled runtime and successful mount/runtime report.

- [ ] **Step 1: Stage exact source trees**

On DSW, fetch the pushed project branch with the one-shot GitHub authorization
header. Export the committed project tree and xLLM commit
`755c7b3a91081fe0f83500eabd53bd2490b8b745` into
separate immutable stage directories, each with a `COMMIT` file.

- [ ] **Step 2: Run DSW CPU tests**

Run the project’s MoE, dense Huginn, and full pytest suites on DSW. Record exact
counts and versions in the stage report.

- [ ] **Step 3: Submit CPU DLC runtime check**

Read the team `launching-dlc-jobs` skill, then create only:

```text
fs0278-xxx-attr-moehuginn500b-01
Priority 1, 0 GPU, 8 CPU, 32Gi, 8Gi shm, hard cap 45 minutes
```

Use the same NeMo 26.06 image and known Zhongwei mounts. Compile
`xllm_extension` against the image, save the build identity, import xLLM, run
the mount checker, and verify 23 tasks/1461 rows, four weight shards, source
commit, project commit, grouped-mm, and output writability.

- [ ] **Step 4: Report JobId immediately and wait to terminal**

Report PAI JobId, full display name, resources, expected 10–35 minute runtime,
and hard cap immediately after acceptance. Do not submit the GPU smoke until
this job succeeds.

- [ ] **Step 5: Commit runtime fixes if needed**

If the check finds a reproducible code/runtime bug, write a failing local
contract test first, fix it on the feature branch, rerun all tests, push a new
commit, and repeat with the next unique sequence. Do not patch only the staged
copy.

---

### Task 7: One-Row GPU Scientific Smoke

**Files:**
- Evidence:
  `/mnt/oss-save/wangjunqi/neuron-explorer/attribution/moe-huginn-500b/smoke-02.json`

- [ ] **Step 1: Submit one bounded GPU smoke**

After CPU success, create:

```text
fs0278-xxx-attr-moehuginn500b-02
Priority 1, 1×L20D, 8 CPU, 32Gi, 8Gi shm, hard cap 90 minutes
```

Run one short real dataset condition at small `k=8`, `max_edges=40`,
direct-edge batch 1.

- [ ] **Step 2: Report JobId immediately**

Report PAI JobId, expected 15–60 minute runtime, hard cap, and resources before
waiting.

- [ ] **Step 3: Enforce scientific gates**

The smoke report must show:

- strict load of 286 tensors;
- tokenizer length 250624 and BOS 0;
- checkpoint and xLLM source fingerprints;
- exactly 112 MoE executions and eight injections;
- two routes per token/depth, valid expert IDs, route weights summing to one;
- stable recurrent shared IDs across eight invocations;
- patched forward values equal the explicit-attention control within measured
  BF16 tolerance;
- native FA control difference separately reported;
- completion log-prob covers all target tokens;
- graph JSON validates as version 3;
- every edge source virtual depth precedes its target;
- peak GPU bytes and seconds per graph.

- [ ] **Step 4: Stop on failure or review smoke**

Any failed gate blocks the full job. Fix through TDD and a new project commit.
After success, perform an independent scientific code review of the branch and
smoke JSON.

---

### Task 8: Full Cognition23 DLC Batch and Completion

**Files:**
- Output:
  `/mnt/oss-save/wangjunqi/neuron-explorer/attribution/moe-huginn-500b/output/`
- Status:
  `launch-status.json`, `gpu-terminal.json`

- [ ] **Step 1: Submit the full batch**

Create:

```text
fs0278-xxx-attr-moehuginn500b-03
Priority 1, 1×L20D, 8 CPU, 32Gi, 8Gi shm, hard cap 1440 minutes
```

Run all 1,461 rows × four conditions at `k=512`,
`max_edges=3000`, with `--benchmark-first`. Use one model copy and the
edge-copy batch proven safe by the benchmark.

- [ ] **Step 2: Report JobId immediately**

Report JobId, resources, expected runtime range based on the benchmark, and
hard cap.

- [ ] **Step 3: Monitor and resume safely**

Monitor progress, failed count, benchmark, and errors. Do not start another
same-content GPU job while this one is running. At terminal state validate
every JSON with version-3 expected metadata. If graphs remain, submit the next
unique sequence only after terminal validation.

- [ ] **Step 4: Validate final corpus**

Require:

- 5,844 valid JSON files;
- 5,844 valid SVG files;
- 5,844 manifest lines and unique task/row/condition keys;
- 23 task directories;
- zero batch failures and zero stale/non-current graphs;
- terminal report recording project commit, xLLM commit, checkpoint
  fingerprint, benchmark, resources, and actual wall time.

- [ ] **Step 5: Merge back to main**

Run the complete suite once more, review the full branch diff, merge
`feature/moe-huginn-attribution` to `main`, rerun tests on merged main, push
`main`, remove the owned worktree, and delete the local feature branch.

