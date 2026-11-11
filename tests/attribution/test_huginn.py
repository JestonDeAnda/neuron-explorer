"""Huginn/Raven attribution: protocol, unrolled depth, and resumable graphs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from neuron_explorer.attribution.huginn.batch import (
    CONDITIONS,
    graph_paths,
    json_is_complete,
    plan_row,
    run_batch,
    task_slug_from_filename,
)
from neuron_explorer.attribution.huginn.protocol import ANSWER_CUE, encode_condition
from neuron_explorer.attribution.huginn.schedule import (
    HUGINN_CODA_BLOCKS,
    HUGINN_PRELUDE_BLOCKS,
    HUGINN_RECURRENT_BLOCKS,
    HUGINN_RECURRENCE,
    execution_schedule,
    shared_weight_id,
)
from neuron_explorer.attribution.huginn.seed import row_seed
from neuron_explorer.attribution.huginn.settings import choose_edge_batch
from neuron_explorer.attribution.schema import load_graph


def _model_path() -> Path:
    configured = os.environ.get("HUGINN_TEST_MODEL")
    if configured:
        return Path(configured).expanduser()
    mounted = Path("/mnt/oss-save/modelscope/tomg-group-umd/huginn-0125")
    if mounted.is_dir():
        return mounted
    snapshots = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--tomg-group-umd--huginn-0125"
        / "snapshots"
    )
    candidates = sorted(snapshots.glob("*")) if snapshots.is_dir() else []
    return candidates[-1] if candidates else mounted


MODEL_PATH = _model_path()


class _Tokenizer:
    bos_token_id = 65504

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        self.seen.append(text)
        return [ord(char) for char in text]


class _BoundaryTokenizer:
    """A BPE-like tokenizer whose continuation token differs in joint text."""

    bos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        return {
            "prompt": [1],
            " answer": [2],
            "prompt answer": [1, 3],
        }[text]


def _row() -> dict:
    return {
        "id": "Lan/anaphor_gender_agreement:412",
        "prompt_variant": "native_bos_answer_cue",
        "clean": "Adam should stun",
        "corrupted": "Ann should stun",
        "correct": " himself",
        "incorrect": " herself",
        "corrupted_correct": " herself",
        "corrupted_incorrect": " himself",
    }


def test_answer_cue_appends_newline_answer_and_keeps_leading_space():
    tokenizer = _Tokenizer()

    encoded = encode_condition(tokenizer, "Adam should stun", " himself", "answer_cue")

    assert tokenizer.seen == ["Adam should stun" + ANSWER_CUE, " himself"]
    assert ANSWER_CUE == "\nAnswer:"
    assert encoded["completion_token_ids"][0] == ord(" ")
    assert encoded["completion_text"] == " himself"
    assert encoded["token_ids"] == encoded["prompt_token_ids"] + encoded["completion_token_ids"]
    assert encoded["completion_start"] == len(encoded["prompt_token_ids"])
    assert encoded["prompt_protocol"]["answer_cue"] is True
    assert encoded["prompt_protocol"]["native_bos"] is False


def test_condition_keeps_fixed_prompt_tokens_when_joint_bpe_would_merge_boundary():
    encoded = encode_condition(_BoundaryTokenizer(), "prompt", " answer", "base")

    assert encoded["prompt_token_ids"] == [1]
    assert encoded["completion_token_ids"] == [2]
    assert encoded["token_ids"] == [1, 2]
    assert encoded["completion_start"] == 1
    assert encoded["prompt_protocol"]["tokenization"] == "separate_context_continuation"


def test_native_bos_is_prepended_after_tokenization_without_answer_cue():
    tokenizer = _Tokenizer()

    encoded = encode_condition(tokenizer, "The men", " were", "native_bos")

    assert tokenizer.seen == ["The men", " were"]
    assert encoded["prompt_token_ids"][0] == tokenizer.bos_token_id
    assert encoded["prompt_token_ids"][1:] == [ord(char) for char in "The men"]
    assert encoded["prompt_protocol"]["native_bos"] is True
    assert encoded["prompt_protocol"]["bos_token_id"] == tokenizer.bos_token_id


def test_base_variant_does_not_add_bos_or_answer_cue():
    tokenizer = _Tokenizer()

    encoded = encode_condition(tokenizer, "The men", " were", "base")

    assert tokenizer.seen == ["The men", " were"]
    assert encoded["prompt_token_ids"][0] == ord("T")
    assert encoded["prompt_protocol"] == {
        "prompt_variant": "base",
        "native_bos": False,
        "answer_cue": False,
        "answer_cue_text": None,
        "bos_token_id": None,
        "tokenization": "separate_context_continuation",
    }


def test_combined_variant_applies_answer_cue_before_native_bos():
    tokenizer = _Tokenizer()

    encoded = encode_condition(
        tokenizer, "Adam should stun", " himself", "native_bos_answer_cue"
    )

    assert tokenizer.seen[0] == "Adam should stun\nAnswer:"
    assert encoded["prompt_token_ids"][0] == tokenizer.bos_token_id
    assert encoded["completion_token_ids"] == [ord(char) for char in " himself"]


def test_row_seed_is_stable_and_shared_by_all_four_conditions():
    jobs = plan_row(_row(), "01_指代性别一致", _Tokenizer())

    assert [job.condition for job in jobs] == [name for name, _prompt, _answer in CONDITIONS]
    assert len({job.seed for job in jobs}) == 1
    assert jobs[0].seed == row_seed("Lan/anaphor_gender_agreement:412")
    assert row_seed("Lan/anaphor_gender_agreement:412") == row_seed(
        "Lan/anaphor_gender_agreement:412"
    )
    assert row_seed("other") != jobs[0].seed
    assert jobs[0].completion_text == " himself"
    assert jobs[2].prompt_text == "Ann should stun"
    assert jobs[2].completion_text == " herself"


def test_huginn_0125_unrolls_virtual_depth_0_through_131():
    slots = execution_schedule(
        HUGINN_PRELUDE_BLOCKS,
        HUGINN_RECURRENT_BLOCKS,
        HUGINN_RECURRENCE,
        HUGINN_CODA_BLOCKS,
    )

    assert [slot.virtual_depth for slot in slots] == list(range(132))
    assert [slot.stage for slot in slots[:2]] == ["prelude", "prelude"]
    assert slots[0].recurrence_iteration is None
    assert slots[0].physical_layer == 0
    recurrent = [slot for slot in slots if slot.stage == "recurrent"]
    assert len(recurrent) == 4 * 32
    assert recurrent[0].virtual_depth == 2
    assert recurrent[0].recurrence_iteration == 0
    assert recurrent[0].physical_layer == 2
    assert recurrent[4].virtual_depth == 6
    assert recurrent[4].recurrence_iteration == 1
    assert recurrent[4].physical_layer == 2
    assert shared_weight_id(recurrent[0], 8109) == shared_weight_id(recurrent[4], 8109)
    assert shared_weight_id(recurrent[0], 8109) == "recurrent:0:n8109"
    assert slots[-2].stage == "coda" and slots[-2].physical_layer == 130
    assert slots[-1].virtual_depth == 131 and slots[-1].physical_layer == 131
    assert slots[-1].recurrence_iteration is None


def test_output_paths_separate_json_and_svg_by_task_and_condition(tmp_path: Path):
    json_path, svg_path = graph_paths(
        tmp_path,
        task_slug_from_filename("01_指代性别一致.jsonl"),
        "Lan/anaphor_gender_agreement:412",
        "clean_correct",
    )

    assert json_path == (
        tmp_path
        / "output"
        / "json"
        / "01_指代性别一致"
        / "Lan__anaphor_gender_agreement:412__clean_correct.json"
    )
    assert svg_path == json_path.with_suffix(".svg").parent.parent.parent / "svg" / json_path.parent.name / (
        json_path.stem + ".svg"
    )
    assert svg_path.name == "Lan__anaphor_gender_agreement:412__clean_correct.svg"


def _graph(token: int = 32) -> dict:
    return {
        "token": token,
        "prob": 0.5,
        "completion_token_ids": [token, token + 1],
        "completion_logprob": -1.25,
        "nodes": [
            {
                "kind": "neuron",
                "layer": 6,
                "pos": 1,
                "n": 3,
                "act": 0.2,
                "influence": 0.4,
                "virtual_depth": 6,
                "stage": "recurrent",
                "recurrence_iteration": 1,
                "physical_layer": 2,
                "physical_neuron": 3,
                "shared_weight_id": "recurrent:0:n3",
            },
            {
                "kind": "logit",
                "layer": 132,
                "pos": 2,
                "n": token + 1,
                "act": -1.25,
                "influence": 1.0,
            },
        ],
        "edges": [[0, 1, -0.2]],
        "stats": {"kept_edges": 1},
        "metadata": {
            "attribution_implementation_version": 2,
            "seed": 7,
            "model_path": "/models/huginn-0125",
            "model_revision": "config-sha256:abc",
            "recurrence_count": 32,
            "prompt_protocol": {"prompt_variant": "base"},
            "code_commit": "deadbeef",
            "graph_config": {"k": 512, "max_edges": 3000, "target": "teacher_forced_completion_logprob"},
        },
    }


def _run_versioned_batch(tmp_path, build=lambda _job: _graph(), **version_options):
    return run_batch(
        [_row()],
        "01_指代性别一致",
        tmp_path,
        _Tokenizer(),
        build,
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

    def fail_if_built(_job):
        raise AssertionError("unchanged dense graph must resume")

    assert _run_versioned_batch(tmp_path, build=fail_if_built)["complete"] == 4


def test_run_batch_rebuilds_dense_version_two_for_explicit_version_three(tmp_path):
    _run_versioned_batch(tmp_path)
    build_calls = 0

    def build(_job):
        nonlocal build_calls
        build_calls += 1
        return _graph()

    _run_versioned_batch(
        tmp_path,
        build=build,
        attribution_implementation_version=3,
    )

    assert build_calls == 4
    graph = load_graph(
        graph_paths(
            tmp_path, "01_指代性别一致", _row()["id"], "clean_correct"
        )[0]
    )
    assert graph["metadata"]["attribution_implementation_version"] == 3


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

    rebuilt: list[str] = []
    _run_versioned_batch(
        tmp_path,
        build=lambda job: rebuilt.append(job.condition) or _graph(),
        attribution_implementation_version=3,
        extra_metadata={"runtime": "xllm", "source_commit": "new-source"},
    )
    assert rebuilt == [name for name, _prompt, _answer in CONDITIONS]


def test_run_batch_rejects_extra_metadata_overriding_reserved_fields(tmp_path):
    with pytest.raises(
        ValueError,
        match=r"extra metadata overrides reserved fields: \['model_path', 'seed'\]",
    ):
        _run_versioned_batch(
            tmp_path,
            extra_metadata={"seed": 1, "model_path": "/other/model"},
        )


def test_edge_serialization_preserves_source_target_and_signed_weight():
    torch = pytest.importorskip("torch")
    from neuron_explorer.attribution.huginn.graph import _serialize_edges

    # Rows are targets, columns are sources: source 0 -> target 2 has weight -0.75.
    adjacency = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-0.75, 0.0, 0.0],
        ]
    )
    edge_mask = adjacency != 0

    assert _serialize_edges(adjacency, edge_mask, {0: 0, 2: 1}) == [[0, 1, -0.75]]


def test_resume_skips_validated_json_and_regenerates_missing_svg(tmp_path: Path):
    calls: list[str] = []

    def build(job):
        calls.append(job.condition)
        graph = _graph()
        graph["metadata"]["condition"] = job.condition
        graph["metadata"]["row_id"] = job.row_id
        return graph

    summary = run_batch(
        [_row()],
        "01_指代性别一致",
        tmp_path,
        _Tokenizer(),
        build,
        code_commit="deadbeef",
        model_path="/models/huginn-0125",
        model_revision="config-sha256:abc",
        recurrence_count=32,
        graph_config={"k": 512, "max_edges": 3000, "target": "teacher_forced_completion_logprob"},
    )
    assert summary["complete"] == 4
    assert len(calls) == 4
    json_path, svg_path = graph_paths(
        tmp_path, "01_指代性别一致", _row()["id"], "clean_correct"
    )
    assert json_is_complete(json_path)
    original = json_path.read_text(encoding="utf-8")
    svg_path.unlink()
    calls.clear()

    again = run_batch(
        [_row()],
        "01_指代性别一致",
        tmp_path,
        _Tokenizer(),
        build,
        code_commit="deadbeef",
        model_path="/models/huginn-0125",
        model_revision="config-sha256:abc",
        recurrence_count=32,
        graph_config={"k": 512, "max_edges": 3000, "target": "teacher_forced_completion_logprob"},
    )

    assert calls == []
    assert again["complete"] == 4
    assert json_path.read_text(encoding="utf-8") == original
    assert svg_path.is_file()
    manifest = (tmp_path / "output" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest) == 4
    progress = json.loads((tmp_path / "output" / "progress.json").read_text(encoding="utf-8"))
    assert progress["tasks"]["01_指代性别一致"]["complete"] == 4
    assert progress["tasks"]["01_指代性别一致"]["expected"] == 4


def test_resume_rebuilds_partial_json_without_clobbering_a_valid_graph(tmp_path: Path):
    valid_job_condition = "clean_correct"
    run_kwargs = {
        "code_commit": "deadbeef",
        "model_path": "/models/huginn-0125",
        "model_revision": "config-sha256:abc",
        "recurrence_count": 32,
        "graph_config": {
            "k": 512,
            "max_edges": 3000,
            "target": "teacher_forced_completion_logprob",
        },
    }

    run_batch(
        [_row()],
        "01_指代性别一致",
        tmp_path,
        _Tokenizer(),
        lambda _job: _graph(token=11),
        **run_kwargs,
    )
    json_path, _svg_path = graph_paths(
        tmp_path, "01_指代性别一致", _row()["id"], valid_job_condition
    )
    partial, _svg = graph_paths(
        tmp_path, "01_指代性别一致", _row()["id"], "clean_incorrect"
    )
    partial.write_text("{", encoding="utf-8")
    built: list[str] = []

    def build(job):
        built.append(job.condition)
        if job.condition == valid_job_condition:
            raise AssertionError("valid graph must not be rebuilt")
        graph = _graph(token=21)
        graph["metadata"]["condition"] = job.condition
        graph["metadata"]["row_id"] = job.row_id
        return graph

    run_batch(
        [_row()],
        "01_指代性别一致",
        tmp_path,
        _Tokenizer(),
        build,
        **run_kwargs,
    )

    assert valid_job_condition not in built
    assert "clean_incorrect" in built
    assert json.loads(json_path.read_text(encoding="utf-8"))["token"] == 11
    assert json_is_complete(partial)


def test_resume_rebuilds_graph_when_tokens_or_run_identity_changed(tmp_path: Path):
    json_path, _svg_path = graph_paths(
        tmp_path, "01_指代性别一致", _row()["id"], "clean_correct"
    )
    stale = _graph()
    stale["prompt_token_ids"] = [999]
    stale["metadata"]["code_commit"] = "old-code"
    json_path.parent.mkdir(parents=True)
    json_path.write_text(json.dumps(stale), encoding="utf-8")
    built: list[str] = []

    def build(job):
        built.append(job.condition)
        return _graph()

    run_batch(
        [_row()],
        "01_指代性别一致",
        tmp_path,
        _Tokenizer(),
        build,
        code_commit="new-code",
        model_path="/models/huginn-0125",
        model_revision="config-sha256:new",
        recurrence_count=32,
        graph_config={"k": 512, "max_edges": 3000, "target": "teacher_forced_completion_logprob"},
    )

    assert "clean_correct" in built
    rebuilt = json.loads(json_path.read_text(encoding="utf-8"))
    assert rebuilt["prompt_token_ids"] != [999]
    assert rebuilt["metadata"]["code_commit"] == "new-code"
    assert rebuilt["metadata"]["model_revision"] == "config-sha256:new"


def test_mount_check_counts_task_files_and_refuses_a_bad_commit(tmp_path: Path):
    from neuron_explorer.attribution.huginn.mount import check_paths

    dataset = tmp_path / "tasks"
    dataset.mkdir()
    (dataset / "01_demo.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "raven_modeling_minimal.py",
        "raven_config_minimal.py",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        (model / name).write_text("{}", encoding="utf-8")
    stage = tmp_path / "stage"
    (stage / "src" / "neuron_explorer" / "attribution" / "huginn").mkdir(parents=True)
    (stage / "COMMIT").write_text("abc\n", encoding="utf-8")

    assert check_paths(
        dataset=dataset,
        model=model,
        output_root=tmp_path / "out",
        stage=stage,
        commit="abc",
        expected_tasks=1,
        expected_rows=2,
    ) == []
    assert check_paths(
        dataset=dataset,
        model=model,
        output_root=tmp_path / "out",
        stage=stage,
        commit="def",
        expected_tasks=1,
        expected_rows=2,
    )


def test_model_revision_fingerprints_checkpoint_weights(tmp_path: Path):
    from neuron_explorer.attribution.huginn.model import model_revision

    (tmp_path / "config.json").write_text('{"model_type":"huginn"}', encoding="utf-8")
    weight = tmp_path / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"first checkpoint")
    first = model_revision(tmp_path)

    weight.write_bytes(b"other checkpoint")
    second = model_revision(tmp_path)

    assert first.startswith("checkpoint-sha256:")
    assert first != second


def test_choose_edge_batch_uses_headroom_and_never_requests_a_second_model_copy():
    tight = choose_edge_batch(peak_bytes=90, total_bytes=100)
    roomy = choose_edge_batch(peak_bytes=10, total_bytes=100)

    assert tight == {"edge_batch": 1, "graph_concurrency": 1, "model_copies": 1}
    assert roomy == {"edge_batch": 2, "graph_concurrency": 2, "model_copies": 1}


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="Huginn checkpoint is not mounted")
def test_real_tokenizer_preserves_leading_space_and_answer_cue():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    spaced = tokenizer.encode(" himself", add_special_tokens=False)
    bare = tokenizer.encode("himself", add_special_tokens=False)
    assert spaced != bare

    encoded = encode_condition(
        tokenizer, "Adam should stun", " himself", "native_bos_answer_cue"
    )

    cue = tokenizer.encode("\nAnswer:", add_special_tokens=False)
    assert encoded["prompt_token_ids"][0] == tokenizer.bos_token_id
    assert encoded["prompt_token_ids"][-len(cue) :] == cue
    assert encoded["completion_token_ids"] == list(spaced)


def test_tiny_raven_forward_neuron_and_completion_contract():
    torch = pytest.importorskip("torch")
    if not MODEL_PATH.is_dir():
        pytest.skip("Huginn checkpoint is not mounted")

    from neuron_explorer.attribution.huginn.linearized import HuginnLinearized
    from neuron_explorer.attribution.huginn.model import tiny_raven

    model = tiny_raven(MODEL_PATH, recurrence=2, seed=0)
    ids = [3, 5, 7, 9, 11, 13]
    completion_start = 4

    def logits_for(seed: int):
        torch.manual_seed(seed)
        with HuginnLinearized(model) as linearized:
            return linearized.forward(ids).detach()

    first = logits_for(11)
    assert torch.allclose(first, logits_for(11), rtol=1e-5, atol=1e-5)
    assert not torch.allclose(first, logits_for(12), rtol=1e-4, atol=1e-4)

    torch.manual_seed(11)
    with torch.no_grad():
        reference = model(torch.tensor([ids]), use_cache=False).logits
    torch.manual_seed(11)
    with HuginnLinearized(model) as linearized:
        matched = linearized.forward(ids)
    assert torch.allclose(matched, reference, rtol=1e-4, atol=1e-4)

    with HuginnLinearized(model) as linearized:
        torch.manual_seed(11)
        replicated = linearized.forward(ids, copies=2)
        recurrent_mlp = model.transformer.core_block[0].mlp
        assert linearized.invocation_counts[id(recurrent_mlp)] == 2
        stored = [
            linearized.cache[key]
            for key in sorted(linearized.cache)
            if key[0] == id(recurrent_mlp) and key[2] == "act"
        ]
        assert len(stored) == 2
        assert not torch.allclose(stored[0], stored[1])

        depths = [1, 6]
        specs = []
        values = []
        for depth in depths:
            activation = linearized.mlp[depth]["a"].detach()[0, -1]
            neuron = int(activation.abs().argmax())
            specs.append((depth, len(ids) - 1, neuron))
            values.append(activation[neuron])
        values = torch.stack(values)
        target = linearized.neuron_target(specs, rule="half")
        contributions = linearized.contributions(target)
        total = (
            contributions["mlp"].sum((0, 2, 3))
            + contributions["embed"].sum(-1)
            + contributions["state"]
        )
        assert torch.allclose(total, values, rtol=1e-3, atol=1e-4)
        assert (contributions["mlp"] < 0).any()

        logprob = linearized.completion_logprob(0, completion_start, ids)
        manual = torch.zeros(())
        for position in range(completion_start - 1, len(ids) - 1):
            manual = manual + torch.log_softmax(replicated[0, position].float(), dim=-1)[
                ids[position + 1]
            ]
        first_only = torch.log_softmax(replicated[0, completion_start - 1].float(), dim=-1)[
            ids[completion_start]
        ]
        assert torch.allclose(logprob, manual, rtol=1e-5, atol=1e-5)
        assert not torch.allclose(logprob, first_only, rtol=1e-3, atol=1e-3)

    torch.manual_seed(11)
    with HuginnLinearized(model) as linearized:
        linearized.forward(ids, copies=1)
        logprob = linearized.completion_logprob(0, completion_start, ids)
        logit_contributions = linearized.contributions(
            linearized.completion_logprob_target([(completion_start, ids)])
        )
        edge = logit_contributions["mlp"][1, 0]
        flat = int(edge.abs().view(-1).argmax())
        position, neuron = divmod(flat, edge.shape[-1])
        local = float(edge[position, neuron])
        activation = float(linearized.mlp[1]["a"][0, position, neuron].detach())
        assert abs(activation) > 1e-6
        direction = 1e-3
        linearized.inject = {1: [(position, neuron, direction)]}
        with torch.no_grad():
            linearized.forward(ids, copies=1, replay=True, freeze_mlp=True)
            changed = float(linearized.completion_logprob(0, completion_start, ids).detach())
        delta = direction * (local / activation)
        assert abs(changed - float(logprob.detach()) - delta) / max(abs(delta), 1e-8) < 0.05


def test_built_graph_records_execution_identity_and_full_completion():
    torch = pytest.importorskip("torch")
    if not MODEL_PATH.is_dir():
        pytest.skip("Huginn checkpoint is not mounted")

    from neuron_explorer.attribution.huginn.graph import build_huginn_graph
    from neuron_explorer.attribution.huginn.model import tiny_raven

    model = tiny_raven(MODEL_PATH, recurrence=2, seed=1)
    ids = [2, 4, 6, 8, 10, 12, 14]
    metadata = {
        "seed": 19,
        "model_path": str(MODEL_PATH),
        "model_revision": "config-sha256:test",
        "recurrence_count": 2,
        "prompt_protocol": {"prompt_variant": "base", "native_bos": False, "answer_cue": False},
        "code_commit": "abc123",
        "graph_config": {
            "k": 6,
            "max_edges": 30,
            "batch": 2,
            "node_threshold": 0.8,
            "edge_threshold": 0.98,
            "target": "teacher_forced_completion_logprob",
            "edge_semantics": "half_rule_direct",
        },
    }
    torch.manual_seed(19)
    graph = build_huginn_graph(
        model,
        ids,
        completion_start=4,
        k=6,
        batch=2,
        max_edges=30,
        metadata=metadata,
    )

    assert graph["completion_token_ids"] == ids[4:]
    assert graph["target"] == {
        "kind": "completion_logprob",
        "token_ids": ids[4:],
        "prediction_positions": [3, 4, 5],
    }
    assert graph["prob"] == pytest.approx(
        math.exp(graph["completion_logprob"]), rel=1e-5, abs=1e-8
    )
    assert graph["metadata"]["seed"] == 19
    assert graph["metadata"]["code_commit"] == "abc123"
    assert graph["metadata"]["graph_config"]["k"] == 6
    assert graph["metadata"]["graph_config"]["max_edges"] == 30
    neurons = [node for node in graph["nodes"] if node["kind"] == "neuron"]
    targets = [node for node in graph["nodes"] if node["kind"] == "logit"]
    assert neurons
    assert len(targets) == 1
    assert targets[0]["target_kind"] == "completion_logprob"
    assert targets[0]["target_token_ids"] == ids[4:]
    assert targets[0]["prediction_positions"] == [3, 4, 5]
    assert graph["edges"]
    for node in neurons:
        assert node["virtual_depth"] == node["layer"]
        assert node["physical_neuron"] == node["n"]
        stage, block, neuron_name = node["shared_weight_id"].split(":")
        assert stage == node["stage"]
        assert block.isdigit()
        assert neuron_name == f"n{node['n']}"
        assert "recurrence_iteration" in node
        assert isinstance(node["influence"], float)
        if node["stage"] == "recurrent":
            assert node["recurrence_iteration"] is not None
        else:
            assert node["recurrence_iteration"] is None
    assert graph["completion_logprob"] != pytest.approx(
        graph["stats"]["first_token_logprob"], rel=1e-3, abs=1e-3
    )
