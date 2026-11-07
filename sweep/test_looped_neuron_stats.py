"""Recurrence-aware activation statistics on toy looped models.

These tests do not load model weights.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep.looped_models import adapt
from sweep.looped_neuron_stats import (
    collect_site_stats,
    iter_parquet_batches,
    iter_text_batches,
    load_compact_arrays,
    parse_args,
    resolve_run,
    save_stats,
)


HIDDEN = 4
NEURONS = 2


def _rows(*positions: list[float]) -> torch.Tensor:
    return torch.tensor(positions, dtype=torch.float32).unsqueeze(0)


def _residual(*norms: float) -> torch.Tensor:
    hidden = torch.zeros(1, len(norms), HIDDEN)
    for index, norm in enumerate(norms):
        hidden[0, index, 0] = norm
    return hidden


class _Scripted(nn.Module):
    def __init__(self, values: list[torch.Tensor]) -> None:
        super().__init__()
        self.values = values
        self.calls = 0
        self.in_features = values[0].shape[-1]

    def forward(self, _hidden):
        value = self.values[self.calls]
        self.calls += 1
        return value


class _SwiGLU(nn.Module):
    def __init__(self, gates: list[torch.Tensor], ups: list[torch.Tensor]) -> None:
        super().__init__()
        self.gate_proj = _Scripted(gates)
        self.up_proj = _Scripted(ups)
        self.down_proj = _Scripted([torch.zeros_like(gate) for gate in gates])

    def forward(self, hidden):
        gate = self.gate_proj(hidden)
        up = self.up_proj(hidden)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class _ChunkedSwiGLU(nn.Module):
    """Huginn's MLP: one fc, split into gate and up, then proj."""

    def __init__(self, gates: list[torch.Tensor], ups: list[torch.Tensor]) -> None:
        super().__init__()
        self.fc = _Scripted([torch.cat([gate, up], dim=-1) for gate, up in zip(gates, ups)])
        self.proj = _Scripted([torch.zeros_like(gate) for gate in gates])

    def forward(self, hidden):
        gate, up = self.fc(hidden).chunk(2, dim=-1)
        return self.proj(torch.nn.functional.silu(gate) * up)


class _Block(nn.Module):
    def __init__(self, mlp: nn.Module, residuals: list[torch.Tensor]) -> None:
        super().__init__()
        self.mlp = mlp
        self.residuals = residuals
        self.calls = 0

    def forward(self, hidden):
        self.mlp(hidden)
        residual = self.residuals[self.calls]
        self.calls += 1
        return residual


class _Ouro(nn.Module):
    def __init__(self, layers: list[nn.Module], loops: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, HIDDEN)
        self.model.layers = nn.ModuleList(layers)
        self.model.total_ut_steps = loops

    def forward(self, input_ids, use_cache=False, **kwargs):
        del use_cache, kwargs
        hidden = self.model.embed_tokens(input_ids)
        for _ in range(self.model.total_ut_steps):
            for layer in self.model.layers:
                hidden = layer(hidden)
        return hidden


class _Huginn(nn.Module):
    def __init__(self, prelude, core, coda) -> None:
        super().__init__()
        self.config = type("Config", (), {"test_time_noise": 0.4, "n_embd": HIDDEN})()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(8, HIDDEN)
        self.transformer.prelude = nn.ModuleList([prelude])
        self.transformer.core_block = nn.ModuleList([core])
        self.transformer.coda = nn.ModuleList([coda])
        self.seen_num_steps = None

    def forward(self, input_ids, num_steps, use_cache=False, **kwargs):
        del use_cache, kwargs
        self.seen_num_steps = num_steps
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.prelude:
            hidden = block(hidden)
        for _ in range(num_steps[0] + num_steps[1]):
            for block in self.transformer.core_block:
                hidden = block(hidden)
        for block in self.transformer.coda:
            hidden = block(hidden)
        return hidden


class _CharTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(char) - 96 for char in text]}


def _hook_count(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        total += len(getattr(module, "_forward_hooks", {}))
        total += len(getattr(module, "_forward_pre_hooks", {}))
    return total


def _ouro_fixture():
    # Position 0 is an attention-sink trap (activation 1e6). Body positions split the four sign regimes.
    # Two forwards of one shared layer: recurrence 0 then recurrence 1, repeated.
    trap = [1000.0, -1000.0]
    gates = [
        _rows(trap, [1.0, -2.0], [-1.0, 2.0]),
        _rows([1000.0, 1000.0], [2.0, 1.0], [-3.0, -0.5]),
        _rows(trap, [4.0, -0.25], [-0.5, 0.5]),
        _rows([1000.0, 1000.0], [2.0, 1.0], [-3.0, -0.5]),
    ]
    ups = [
        _rows([1000.0, -1000.0], [3.0, -4.0], [4.0, -5.0]),
        _rows([1000.0, -1000.0], [0.5, -6.0], [-1.5, 2.0]),
        _rows([1000.0, -1000.0], [1.0, -0.5], [1.0, -0.5]),
        _rows([1000.0, -1000.0], [0.5, -6.0], [-1.5, 2.0]),
    ]
    residuals = [
        _residual(1000.0, 10.0, 30.0),
        _residual(1000.0, 40.0, 80.0),
        _residual(1000.0, 50.0, 70.0),
        _residual(1000.0, 40.0, 80.0),
    ]
    model = _Ouro([_Block(_SwiGLU(gates, ups), residuals)], loops=2)
    return model, adapt(model, family="ouro")


class LoopedNeuronStatsTests(unittest.TestCase):
    def test_extrema_are_separated_by_site_and_sign_regime(self):
        # Including position 0 would make abs_max 1e6. Recurrence 0 and 1 must not share extremes.
        _model, adapter = _ouro_fixture()
        ids = torch.tensor([[1, 2, 3]])
        stats = collect_site_stats(adapter, [ids, ids])
        pp = stats.pp_max
        nn = stats.nn_max
        pn = stats.pn_min
        np_ = stats.np_min
        self.assertEqual(tuple(pp.shape), (2, NEURONS))
        np.testing.assert_allclose(pp[0], [3.92805505, 0.0], rtol=1e-5)
        np.testing.assert_allclose(nn[0], [0.0, 0.95362335], rtol=1e-5)
        np.testing.assert_allclose(pn[0], [0.0, -8.80797005], rtol=1e-5)
        np.testing.assert_allclose(np_[0], [-1.07576573, 0.0], rtol=1e-5)
        np.testing.assert_allclose(pp[1], [0.88079703, 0.0], rtol=1e-5)
        np.testing.assert_allclose(nn[1], [0.21341643, 0.0], rtol=1e-5)
        np.testing.assert_allclose(pn[1], [0.0, -4.38635159], rtol=1e-5)
        np.testing.assert_allclose(np_[1], [0.0, -0.37754068], rtol=1e-5)
        np.testing.assert_allclose(stats.abs_max[0], [3.92805505, 8.80797005], rtol=1e-5)
        np.testing.assert_allclose(stats.abs_max[1], [0.88079703, 4.38635159], rtol=1e-5)
        self.assertLess(float(stats.abs_max.max()), 100.0)

    def test_residual_quantiles_follow_the_execution_site(self):
        _model, adapter = _ouro_fixture()
        ids = torch.tensor([[1, 2, 3]])
        stats = collect_site_stats(adapter, [ids, ids])
        self.assertEqual(tuple(stats.resid_norm_q10_50_90.shape), (2, 3))
        np.testing.assert_allclose(stats.resid_norm_q10_50_90[0], [16.0, 40.0, 64.0])
        np.testing.assert_allclose(stats.resid_norm_q10_50_90[1], [40.0, 60.0, 80.0])
        self.assertEqual(
            [(site.kind, site.physical_index, site.recurrence, site.virtual_index) for site in stats.sites],
            [("layer", 0, 0, 0), ("layer", 0, 1, 1)],
        )

    def test_huginn_chunked_fc_keeps_gate_and_up_signs(self):
        # Swapping the fc halves would put neuron 1's negative activation on np instead of pn.
        gates = [_rows([1000.0, 1000.0], [1.0, 1.0], [1.0, 1.0])]
        ups = [_rows([1000.0, -1000.0], [3.0, -4.0], [3.0, -4.0])]
        residual = [_residual(1000.0, 10.0, 30.0)]
        filler_g = [_rows([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])]
        filler_u = [_rows([1.0, 1.0], [1.0, 1.0], [1.0, 1.0])]
        filler_r = [_residual(0.0, 100.0, 300.0)]
        model = _Huginn(
            _Block(_ChunkedSwiGLU(gates, ups), residual),
            _Block(_ChunkedSwiGLU(filler_g, filler_u), filler_r),
            _Block(_ChunkedSwiGLU(filler_g, filler_u), filler_r),
        )
        adapter = adapt(model, family="huginn", recurrences=1, seed=5)
        base = _hook_count(model)
        stats = collect_site_stats(adapter, [torch.tensor([[1, 2, 3]])])
        self.assertEqual(model.seen_num_steps, (0, 1))
        self.assertEqual(model.config.test_time_noise, 0)
        prelude = stats.sites[0]
        self.assertEqual((prelude.kind, prelude.recurrence), ("prelude", 0))
        np.testing.assert_allclose(stats.pp_max[0], [2.19317579, 0.0], rtol=1e-5)
        np.testing.assert_allclose(stats.pn_min[0], [0.0, -2.92423439], rtol=1e-5)
        np.testing.assert_allclose(stats.np_min[0], [0.0, 0.0])
        np.testing.assert_allclose(stats.resid_norm_q10_50_90[0], [12.0, 20.0, 28.0])
        np.testing.assert_allclose(stats.resid_norm_q10_50_90[1], [120.0, 200.0, 280.0])
        self.assertEqual(_hook_count(model), base)

    def test_selected_site_and_incompatible_width_fail_clearly(self):
        _model, adapter = _ouro_fixture()
        ids = torch.tensor([[1, 2, 3]])
        only = collect_site_stats(adapter, [ids, ids], sites=[adapter.sites[1]])
        self.assertEqual(len(only.sites), 1)
        self.assertEqual(only.sites[0].recurrence, 1)
        np.testing.assert_allclose(only.pp_max[0], [0.88079703, 0.0], rtol=1e-5)

        wide = nn.Linear(3, HIDDEN, bias=False)
        narrow = nn.Linear(NEURONS, HIDDEN, bias=False)

        class Plain(nn.Module):
            def __init__(self, proj):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.down_proj = proj

            def forward(self, hidden):
                raise AssertionError("width check runs before the forward")

        class Mixed(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(8, HIDDEN)
                self.model.total_ut_steps = 1
                self.model.layers = nn.ModuleList([Plain(narrow), Plain(wide)])

            def forward(self, input_ids, use_cache=False, **kwargs):
                del input_ids, use_cache, kwargs
                raise AssertionError("width check runs before the forward")

        mixed = Mixed()
        mixed_adapter = adapt(mixed, family="ouro")
        base = _hook_count(mixed)
        with self.assertRaises(ValueError) as caught:
            collect_site_stats(mixed_adapter, [ids])
        self.assertIn("neuron", str(caught.exception).lower())
        self.assertIn("width", str(caught.exception).lower())
        self.assertEqual(_hook_count(mixed), base)

    def test_missing_firing_and_hook_cleanup(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.gate_proj = nn.Linear(HIDDEN, NEURONS, bias=False)
                self.mlp.up_proj = nn.Linear(HIDDEN, NEURONS, bias=False)
                self.mlp.down_proj = nn.Linear(NEURONS, HIDDEN, bias=False)

            def forward(self, hidden):
                gate = self.mlp.gate_proj(hidden)
                up = self.mlp.up_proj(hidden)
                return self.mlp.down_proj(torch.nn.functional.silu(gate) * up)

        class Short(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(8, HIDDEN)
                self.model.total_ut_steps = 2
                self.model.layers = nn.ModuleList([Block()])

            def forward(self, input_ids, use_cache=False, **kwargs):
                del use_cache, kwargs
                return self.model.layers[0](self.model.embed_tokens(input_ids))

        model = Short()
        adapter = adapt(model, family="ouro")
        base = _hook_count(model)
        with self.assertRaises(RuntimeError) as caught:
            collect_site_stats(adapter, [torch.tensor([[1, 2, 3]])])
        message = str(caught.exception)
        self.assertIn("fired", message)
        self.assertIn("expected", message)
        self.assertEqual(_hook_count(model), base)

        _again, again = _ouro_fixture()
        before = _hook_count(_again)
        collect_site_stats(again, [torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 3]])])
        self.assertEqual(_hook_count(_again), before)

    def test_compact_npz_keeps_site_metadata_and_controls_without_down_matrices(self):
        _model, adapter = _ouro_fixture()
        ids = torch.tensor([[1, 2, 3]])
        stats = collect_site_stats(adapter, [ids, ids])
        config = {
            "model": "/models/toy-ouro",
            "family": "ouro",
            "recurrences": None,
            "seed": 0,
            "test_time_noise": 0.0,
            "texts": "/data/texts.jsonl",
            "text_field": "passage",
            "corpus_parquet": None,
            "tokens": None,
            "seq_len": None,
            "batch": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "stats.npz")
            save_stats(path, stats, config)
            saved = np.load(path)
            self.assertNotIn("w_down", saved.files)
            self.assertFalse(any("weight" in name or "down" in name for name in saved.files))
            for name in saved.files:
                array = saved[name]
                if array.ndim >= 2:
                    self.assertNotIn(HIDDEN, array.shape)
            np.testing.assert_allclose(saved["pp_max"], stats.pp_max)
            np.testing.assert_allclose(saved["abs_max"], stats.abs_max)
            np.testing.assert_allclose(saved["resid_norm_q10_50_90"], stats.resid_norm_q10_50_90)
            meta = json.loads(saved["sites_json"].item())
            self.assertEqual(meta, [
                {"kind": "layer", "physical_index": 0, "recurrence": 0, "virtual_index": 0},
                {"kind": "layer", "physical_index": 0, "recurrence": 1, "virtual_index": 1},
            ])
            self.assertEqual(json.loads(saved["config_json"].item()), config)
            seen = []
            original_load = np.load

            def local_only(file, *args, **kwargs):
                seen.append(str(file))
                if Path(str(file)).resolve() == Path(path).resolve():
                    raise OSError(22, "Invalid argument")
                return original_load(file, *args, **kwargs)

            np.load = local_only
            try:
                arrays = load_compact_arrays(path)
            finally:
                np.load = original_load
            self.assertTrue(seen)
            self.assertNotEqual(Path(seen[0]).resolve(), Path(path).resolve())
            np.testing.assert_allclose(arrays["pp_max"], stats.pp_max)

    def test_text_and_parquet_batches_and_run_controls(self):
        tokenizer = _CharTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "texts.jsonl"
            jsonl.write_text(
                '{"passage": "ab"}\n{"passage": "c"}\n',
                encoding="utf-8",
            )
            batches = list(iter_text_batches(tokenizer, jsonl, "passage"))
            self.assertEqual([batch.tolist() for batch in batches], [[[1, 2]], [[3]]])

            broken = root / "broken.jsonl"
            broken.write_text('{"passage": "ab"}\n{"text": "c"}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                list(iter_text_batches(tokenizer, broken, "passage"))
            self.assertIn("passage", str(caught.exception))
            self.assertIn("2", str(caught.exception))

            table = pa.table({"passage": ["ab", "cd"]})
            parquet = root / "corpus.parquet"
            pq.write_table(table, parquet)
            windows = list(iter_parquet_batches(tokenizer, parquet, tokens=6, seq_len=3, batch=2, text_field="passage"))
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0].tolist(), [[1, 2, 9], [3, 4, 9]])
            with self.assertRaises(ValueError) as short:
                list(iter_parquet_batches(tokenizer, parquet, tokens=9, seq_len=3, batch=1, text_field="passage"))
            self.assertIn("token", str(short.exception).lower())

        ouro = resolve_run(parse_args([
            "--model", "/models/ouro",
            "--family", "ouro",
            "--out", "/tmp/ouro.npz",
            "--texts", "/tmp/t.jsonl",
            "--text-field", "passage",
        ]))
        self.assertEqual(ouro.family, "ouro")
        self.assertIsNone(ouro.recurrences)
        self.assertEqual(ouro.seed, 0)
        self.assertEqual(ouro.test_time_noise, 0.0)
        self.assertEqual(ouro.text_field, "passage")

        huginn = resolve_run(parse_args([
            "--model", "/models/huginn",
            "--family", "huginn",
            "--out", "/tmp/huginn.npz",
            "--recurrences", "8",
            "--seed", "3",
            "--corpus-parquet", "/tmp/corpus.parquet",
            "--tokens", "128",
            "--seq-len", "16",
            "--batch", "2",
        ]))
        self.assertEqual(huginn.recurrences, 8)
        self.assertEqual(huginn.seed, 3)
        self.assertEqual(huginn.test_time_noise, 0.0)
        self.assertEqual(huginn.tokens, 128)
        self.assertEqual(huginn.seq_len, 16)
        self.assertEqual(huginn.batch, 2)

        with self.assertRaises(SystemExit):
            parse_args([
                "--model", "/models/huginn",
                "--family", "huginn",
                "--out", "/tmp/h.npz",
                "--texts", "/tmp/t.jsonl",
                "--test-time-noise", "0.5",
            ])
        with self.assertRaises(ValueError):
            resolve_run(parse_args([
                "--model", "/models/huginn",
                "--family", "huginn",
                "--out", "/tmp/h.npz",
                "--texts", "/tmp/t.jsonl",
            ]))
        with self.assertRaises(ValueError):
            resolve_run(parse_args([
                "--model", "/models/ouro",
                "--family", "ouro",
                "--out", "/tmp/o.npz",
                "--texts", "/tmp/t.jsonl",
                "--recurrences", "4",
            ]))
        with self.assertRaises(ValueError):
            resolve_run(parse_args([
                "--model", "/models/ouro",
                "--family", "ouro",
                "--out", "/tmp/o.npz",
                "--texts", "/tmp/t.jsonl",
                "--corpus-parquet", "/tmp/c.parquet",
                "--tokens", "8",
                "--seq-len", "4",
                "--batch", "1",
            ]))


if __name__ == "__main__":
    unittest.main()
