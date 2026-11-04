"""Recurrence-aware steering sweep on toy looped models.

These tests do not load model weights.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep.looped_models import adapt
from sweep.rank import compute_units
from sweep.test_looped_models import ToyHuginn


HIDDEN = 4
PROMPTS = {
    "who": "Who are you?",
    "snake": "Write a snake game in Python.",
    "orange": "Describe the color orange.",
    "free": "Write whatever you like.",
}
MAIN_REL = [(prompt, dose) for prompt in ("who", "snake", "orange") for dose in (-0.7, 0.5, 0.8)] + [("free", 3.0)]
PROMPT_IDS = {
    "Who are you?": [1, 2],
    "Write a snake game in Python.": [3, 4],
    "Describe the color orange.": [5, 6],
    "Write whatever you like.": [7, 8],
}


class Tok:
    eos_token_id = None
    chat_template = "toy"

    def apply_chat_template(self, messages, add_generation_prompt=True, enable_thinking=False, tokenize=True):
        del add_generation_prompt, enable_thinking, tokenize
        return list(PROMPT_IDS[messages[0]["content"]])

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(token)) for token in ids)


class OuroLM(nn.Module):
    def __init__(self, loops: int) -> None:
        super().__init__()
        self.forwards = 0
        self.seen = []
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, HIDDEN)
        self.model.layers = nn.ModuleList([_block() for _ in range(1)])
        self.model.total_ut_steps = loops
        self.lm_head = nn.Linear(HIDDEN, 16, bias=False)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.arange(16 * HIDDEN, dtype=torch.float32).reshape(16, HIDDEN))
            self.model.layers[0].mlp.down_proj.weight.copy_(torch.eye(HIDDEN))
            self.lm_head.weight.copy_(torch.eye(16, HIDDEN))

    def forward(self, input_ids, use_cache=False, **kwargs):
        del use_cache, kwargs
        self.forwards += 1
        hidden = self.model.embed_tokens(input_ids)
        for _ in range(self.model.total_ut_steps):
            for layer in self.model.layers:
                hidden = layer(hidden)
        self.seen.append((input_ids.detach().clone(), hidden.detach().clone()))
        return type("Out", (), {"logits": self.lm_head(hidden)})()


class CausalHuginn(ToyHuginn):
    def __init__(self) -> None:
        super().__init__()
        self.transformer.wte = nn.Embedding(16, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, 16, bias=False)
        self.flags = []
        self.steps = []
        self.noises = []

    def forward(self, input_ids, num_steps, use_cache=False, **kwargs):
        self.flags.append(float(self.config.test_time_noise))
        hidden = super().forward(input_ids, num_steps, use_cache=use_cache, **kwargs)
        self.steps.append(tuple(self.seen_num_steps))
        self.noises.append(self.noise.detach().clone())
        self.config.test_time_noise = 0.25
        return type("Out", (), {"logits": self.lm_head(hidden)})()


def _block():
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.down_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
    block.forward = lambda hidden, layer=block: layer.mlp.down_proj(hidden)
    return block


def _hook_count(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        total += len(getattr(module, "_forward_hooks", {}))
        total += len(getattr(module, "_forward_pre_hooks", {}))
    return total


def _write_stats(path, sites, p50, family, recurrences=None):
    meta = [
        {
            "kind": site.kind,
            "physical_index": site.physical_index,
            "recurrence": site.recurrence,
            "virtual_index": site.virtual_index,
        }
        for site in sites
    ]
    resid = np.zeros((len(sites), 3), dtype=np.float64)
    for index, value in enumerate(p50):
        resid[index, 1] = value
    config = {
        "family": family,
        "model": "toy",
        "test_time_noise": 0.0,
        "seed": 0,
        "recurrences": recurrences,
    }
    np.savez(
        path,
        pp_max=np.ones((len(sites), HIDDEN)),
        nn_max=np.ones((len(sites), HIDDEN)),
        pn_min=-np.ones((len(sites), HIDDEN)),
        np_min=-np.ones((len(sites), HIDDEN)),
        abs_max=np.ones((len(sites), HIDDEN)),
        resid_norm_q10_50_90=resid,
        sites_json=np.array(json.dumps(meta)),
        config_json=np.array(json.dumps(config)),
    )


def _records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


class LoopedSteerSweepTests(unittest.TestCase):
    def test_site_run_records_rank_fields_and_one_recurrence_push(self):
        # Identity columns have norm 1. Virtual site 1 has residual p50 10, so a
        # dose of 0.5 adds 5 along neuron 0. Also editing recurrence 0 (p50 4)
        # would add 2 more and land on 7.
        from sweep.looped_steer_sweep import MAIN_REL as design
        from sweep.looped_steer_sweep import PROMPTS as prompts
        from sweep.looped_steer_sweep import parse_args, run

        self.assertEqual(prompts, PROMPTS)
        self.assertEqual(design, MAIN_REL)
        model = OuroLM(loops=2)
        adapter = adapt(model, family="ouro")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.npz"
            _write_stats(stats, adapter.sites, [4.0, 10.0], "ouro")
            args = parse_args([
                "--model", "missing-weights",
                "--family", "ouro",
                "--stats", str(stats),
                "--site", "1",
                "--smoke-neurons", "0",
                "--run", "site1",
                "--out-dir", str(root),
                "--max-tokens", "1",
                "--top-k", "3",
                "--batch-size", "8",
                "--random-controls", "0",
            ])
            path = run(args, model, Tok())
            self.assertEqual(model.forwards, 8)
            self.assertEqual(max(ids.shape[0] for ids, _hidden in model.seen), 3)
            rows = _records(path)
            self.assertEqual(len(rows), 14)
            self.assertEqual(sum(row["kind"] == "baseline" for row in rows), 4)
            neuron_rows = [row for row in rows if row["kind"] == "neuron"]
            self.assertEqual(
                [(row["prompt"], row["dose"]) for row in neuron_rows],
                MAIN_REL,
            )
            sample = next(row for row in neuron_rows if row["prompt"] == "who" and row["dose"] == 0.5)
            self.assertEqual(sample["schema_version"], 2)
            self.assertEqual(sample["family"], "ouro")
            self.assertEqual(sample["selector"], "site")
            self.assertEqual(sample["virtual_index"], 1)
            self.assertEqual(sample["physical_index"], 0)
            self.assertEqual(sample["recurrence"], 1)
            self.assertEqual(sample["block_kind"], "layer")
            self.assertEqual(sample["virtual_indices"], [1])
            self.assertEqual(sample["layer"], 1)
            self.assertEqual(sample["neuron"], 0)
            self.assertEqual(sample["ref_neuron"], 0)
            self.assertEqual(sample["add"], 5.0)
            self.assertEqual(sample["site_add"], [5.0])
            self.assertEqual(sample["rel_push"], 0.5)
            self.assertEqual(sample["finish"], "length")
            self.assertEqual(sample["n_gen"], 1)
            self.assertEqual(len(sample["token_ids"]), 1)
            self.assertEqual(len(sample["logprob"]), 1)
            self.assertEqual(len(sample["top_ids"]), 1)
            self.assertEqual(len(sample["top_ids"][0]), 3)
            self.assertEqual(len(sample["top_lps"][0]), 3)
            self.assertEqual(sample["text"], Tok().decode(sample["token_ids"]))
            for field in ("text", "token_ids", "finish", "logprob", "rel_push", "prompt", "dose", "layer", "ref_neuron"):
                self.assertIn(field, sample)
            units = compute_units(rows)
            self.assertEqual(units[0]["neuron"], 0)
            self.assertEqual(units[0]["kind"], "neuron")
            who = model.seen[4]
            self.assertEqual(who[0].shape[0], 3)
            embed = model.model.embed_tokens(who[0])
            diff = who[1] - embed
            self.assertTrue(torch.allclose(diff[1, :, 0], torch.full((2,), 5.0)))
            self.assertTrue(torch.count_nonzero(diff[1, :, 1:]) == 0)
            base_ids, base_hidden = model.seen[0]
            self.assertTrue(torch.equal(base_hidden, model.model.embed_tokens(base_ids)))
            self.assertEqual(_hook_count(model), 0)

    def test_all_recurrences_add_every_selected_firing(self):
        from sweep.looped_steer_sweep import parse_args, run

        model = OuroLM(loops=2)
        adapter = adapt(model, family="ouro")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.npz"
            _write_stats(stats, adapter.sites, [4.0, 10.0], "ouro")
            args = parse_args([
                "--model", "missing-weights",
                "--family", "ouro",
                "--stats", str(stats),
                "--all-recurrences", "layer:0",
                "--smoke-neurons", "0",
                "--run", "all",
                "--out-dir", str(root),
                "--max-tokens", "1",
                "--batch-size", "8",
                "--random-controls", "0",
            ])
            run(args, model, Tok())
            sample = next(row for row in _records(root / "all.jsonl") if row["dose"] == 0.5 and row["kind"] == "neuron")
            self.assertEqual(sample["selector"], "all_recurrences")
            self.assertIsNone(sample["virtual_index"])
            self.assertIsNone(sample["recurrence"])
            self.assertEqual(sample["physical_index"], 0)
            self.assertEqual(sample["block_kind"], "layer")
            self.assertEqual(sample["virtual_indices"], [0, 1])
            self.assertEqual(sample["recurrences"], [0, 1])
            self.assertEqual(sample["sites"][0]["virtual_index"], 0)
            self.assertEqual(sample["sites"][1]["recurrence"], 1)
            self.assertEqual(sample["site_add"], [2.0, 5.0])
            self.assertEqual(sample["rel_push"], 0.5)
            _ids, hidden = model.seen[4]
            embed = model.model.embed_tokens(_ids)
            diff = hidden - embed
            self.assertTrue(torch.allclose(diff[1, :, 0], torch.full((2,), 7.0)))
            self.assertTrue(torch.count_nonzero(diff[1, :, 1:]) == 0)

    def test_row_specific_batch_keeps_each_neurons_dose(self):
        from sweep.looped_steer_sweep import parse_args, run

        model = OuroLM(loops=1)
        adapter = adapt(model, family="ouro")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.npz"
            _write_stats(stats, adapter.sites, [4.0], "ouro")
            args = parse_args([
                "--model", "missing-weights",
                "--family", "ouro",
                "--stats", str(stats),
                "--site", "0",
                "--smoke-neurons", "0,1",
                "--run", "rows",
                "--out-dir", str(root),
                "--max-tokens", "1",
                "--batch-size", "4",
                "--random-controls", "0",
            ])
            run(args, model, Tok())
            ids, hidden = model.seen[4]
            self.assertEqual(tuple(ids.shape), (4, 2))
            embed = model.model.embed_tokens(ids)
            diff = hidden - embed
            # who jobs, batch 4: neuron 0 at -0.7, 0.5, 0.8, then neuron 1 at -0.7.
            # p50 4 and column norm 1, so the activation add is dose * 4.
            expected = torch.zeros_like(diff)
            expected[0, :, 0] = -2.8
            expected[1, :, 0] = 2.0
            expected[2, :, 0] = 3.2
            expected[3, :, 1] = -2.8
            self.assertTrue(torch.allclose(diff, expected, atol=1e-5))

    def test_random_control_matches_borrowed_column_and_rel_push(self):
        from sweep.looped_steer_sweep import edits_for_rows, load_compact_stats, parse_args, run

        model = OuroLM(loops=1)
        with torch.no_grad():
            weight = torch.zeros(HIDDEN, HIDDEN)
            weight[0, 0] = 2
            weight[1, 1] = 6
            weight[2, 2] = 3
            weight[3, 3] = 5
            model.model.layers[0].mlp.down_proj.weight.copy_(weight)
        adapter = adapt(model, family="ouro")
        matched = torch.tensor([-0.77545846, 3.7592952, 0.615766, -3.1443913])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats_path = root / "stats.npz"
            _write_stats(stats_path, adapter.sites, [4.0], "ouro")
            args = parse_args([
                "--model", "missing-weights",
                "--family", "ouro",
                "--stats", str(stats_path),
                "--site", "0",
                "--smoke-neurons", "0",
                "--run", "rand",
                "--out-dir", str(root),
                "--max-tokens", "1",
                "--batch-size", "4",
                "--random-controls", "1",
                "--seed", "0",
            ])
            run(args, model, Tok())
            sample = next(
                row for row in _records(root / "rand.jsonl")
                if row["kind"] == "random" and row["prompt"] == "who" and row["dose"] == 0.5
            )
            self.assertEqual(sample["ref_neuron"], 3)
            self.assertAlmostEqual(sample["add"], 0.4)
            self.assertAlmostEqual(sample["rel_push"], 0.5)
            stats = load_compact_stats(str(stats_path))
            column = torch.linalg.vector_norm(weight.float(), dim=0)
            edits = edits_for_rows(
                [{"kind": "random", "neuron": 0, "ref_neuron": 3, "dose": 0.5, "prompt": "who"}],
                (adapter.sites[0],),
                stats,
                column,
                matched.unsqueeze(0),
                HIDDEN,
            )
            plain = OuroLM(loops=1)
            with torch.no_grad():
                plain.model.layers[0].mlp.down_proj.weight.copy_(weight)
                plain.model.embed_tokens.weight.copy_(model.model.embed_tokens.weight)
            plain_adapter = adapt(plain, family="ouro")
            ids = torch.tensor([PROMPT_IDS["Who are you?"]])
            base = plain_adapter.forward(ids)
            with plain_adapter.intervene(edits):
                steered = plain_adapter.forward(ids)
            # The diagonal maps embed[..., i] to column_norm[i] * embed[..., i].
            # The control is added after that map, so the residual is the whole difference.
            diff = steered.logits[:, :, :HIDDEN] - base.logits[:, :, :HIDDEN]
            # lm_head is eye(16, 4), so the first 4 logits are the hidden state.
            self.assertTrue(torch.allclose(diff, (0.4 * matched).view(1, 1, 4), atol=1e-5))

    def test_resume_skips_finished_keys_and_keeps_the_other_site(self):
        from sweep.looped_steer_sweep import parse_args, record_key, run

        model = OuroLM(loops=1)
        adapter = adapt(model, family="ouro")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.npz"
            _write_stats(stats, adapter.sites, [4.0], "ouro")
            args = parse_args([
                "--model", "missing-weights",
                "--family", "ouro",
                "--stats", str(stats),
                "--site", "0",
                "--smoke-neurons", "0",
                "--run", "resume",
                "--out-dir", str(root),
                "--max-tokens", "1",
                "--batch-size", "2",
                "--random-controls", "0",
            ])
            path = Path(run(args, model, Tok()))
            first = path.read_text(encoding="utf-8")
            done = model.forwards
            again = run(args, model, Tok())
            self.assertEqual(again, str(path))
            self.assertEqual(model.forwards, done)
            self.assertEqual(path.read_text(encoding="utf-8"), first)
            lines = first.splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            run(args, model, Tok())
            restored = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(restored[:-1], lines[:-1])
            self.assertEqual(json.loads(restored[-1])["prompt"], json.loads(lines[-1])["prompt"])
            self.assertEqual(json.loads(restored[-1])["dose"], json.loads(lines[-1])["dose"])
            self.assertEqual(len(restored), len(lines))
            self.assertEqual(model.forwards, done + 1)
            keys = [record_key(json.loads(line)) for line in restored]
            self.assertEqual(len(keys), len(set(keys)))

            other = parse_args([
                "--model", "missing-weights",
                "--family", "ouro",
                "--stats", str(stats),
                "--site", "0",
                "--smoke-neurons", "0",
                "--run", "resume",
                "--out-dir", str(root),
                "--max-tokens", "1",
                "--batch-size", "2",
                "--random-controls", "0",
                "--all-recurrences", "layer:0",
            ])
            with self.assertRaises(SystemExit):
                parse_args([
                    "--model", "missing-weights",
                    "--family", "ouro",
                    "--stats", str(stats),
                    "--smoke-neurons", "0",
                    "--run", "x",
                    "--out-dir", str(root),
                ])
            self.assertIsNotNone(other)

    def test_smoke_file_limits_neurons(self):
        from sweep.looped_steer_sweep import parse_args, run

        model = OuroLM(loops=1)
        adapter = adapt(model, family="ouro")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.npz"
            listing = root / "neurons.txt"
            listing.write_text("0\n1\n", encoding="utf-8")
            _write_stats(stats, adapter.sites, [4.0], "ouro")
            args = parse_args([
                "--model", "missing-weights",
                "--family", "ouro",
                "--stats", str(stats),
                "--site", "0",
                "--smoke-neurons", str(listing),
                "--run", "smoke",
                "--out-dir", str(root),
                "--max-tokens", "1",
                "--batch-size", "4",
                "--random-controls", "0",
            ])
            run(args, model, Tok())
            neurons = {row["neuron"] for row in _records(root / "smoke.jsonl") if row["kind"] == "neuron"}
            self.assertEqual(neurons, {0, 1})

    def test_huginn_generation_pins_recurrences_seed_and_zero_noise(self):
        from sweep.looped_steer_sweep import parse_args, run

        model = CausalHuginn()
        adapter = adapt(model, family="huginn", recurrences=2, seed=11)
        model.config.test_time_noise = 0.25
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.npz"
            _write_stats(stats, adapter.sites, [8.0] * len(adapter.sites), "huginn", recurrences=2)
            torch.manual_seed(5)
            sentinel = torch.randn(4).clone()
            args = parse_args([
                "--model", "missing-weights",
                "--family", "huginn",
                "--stats", str(stats),
                "--site", "2",
                "--smoke-neurons", "0",
                "--run", "huginn",
                "--out-dir", str(root),
                "--max-tokens", "2",
                "--batch-size", "4",
                "--random-controls", "0",
                "--recurrences", "2",
                "--seed", "11",
            ])
            run(args, model, Tok())
            self.assertGreaterEqual(len(model.flags), 2)
            self.assertTrue(all(flag == 0.0 for flag in model.flags))
            self.assertTrue(all(step == (0, 2) for step in model.steps))
            torch.manual_seed(11)
            expected = torch.randn(4)
            for noise in model.noises:
                self.assertTrue(torch.equal(noise, expected))
            torch.manual_seed(5)
            self.assertTrue(torch.equal(sentinel, torch.randn(4)))
            self.assertEqual(model.config.test_time_noise, 0.0)
            self.assertEqual(_hook_count(model), 0)
            sample = next(row for row in _records(root / "huginn.jsonl") if row["kind"] == "neuron")
            self.assertEqual(sample["family"], "huginn")
            self.assertEqual(sample["schema_version"], 2)
            self.assertEqual(sample["n_gen"], 2)
            self.assertEqual(len(sample["logprob"]), 2)


if __name__ == "__main__":
    unittest.main()
