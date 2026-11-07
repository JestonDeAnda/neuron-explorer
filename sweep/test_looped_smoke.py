"""Production smoke orchestration tests; no checkpoints or GPUs are used."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sweep.looped_models import adapt
from sweep.looped_smoke import (
    _legacy_rope_compat,
    _legacy_tied_weight_mapping,
    aggregate_nvidia_csv,
    build_summary,
    prepare_checkpoint_config,
    run_smoke,
    stabilize_low_precision_attention,
    validate_full_occupancy,
    validate_observer_equivalence,
)


class TinyTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return {"input_ids": [1, 2, 3]}

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [1, 2]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(item)) for item in ids)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden):
        gate = self.gate_proj(hidden)
        up = self.up_proj(hidden)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, hidden):
        return self.mlp(hidden)


class TinyLM(nn.Module):
    def __init__(self, changing=False):
        super().__init__()
        torch.manual_seed(4)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 4)
        self.model.layers = nn.ModuleList([TinyBlock()])
        self.model.total_ut_steps = 2
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.changing = changing
        self.calls = 0

    def forward(self, input_ids, use_cache=False, **kwargs):
        del use_cache, kwargs
        hidden = self.model.embed_tokens(input_ids)
        for _ in range(self.model.total_ut_steps):
            for layer in self.model.layers:
                hidden = layer(hidden)
        logits = self.lm_head(hidden)
        if self.changing:
            logits = logits + self.calls * 0.01
        self.calls += 1
        return SimpleNamespace(logits=logits)


class NoCuda:
    @staticmethod
    def is_available():
        return False


class LoopedSmokeTests(unittest.TestCase):
    def test_legacy_default_rope_uses_unit_proportional_fallback_temporarily(self):
        proportional = object()
        registry = {"proportional": proportional}
        with _legacy_rope_compat(registry):
            self.assertIs(registry["default"], proportional)
        self.assertNotIn("default", registry)

    def test_checkpoint_config_fills_missing_pad_token_from_eos(self):
        config = SimpleNamespace(eos_token_id=2)
        tokenizer = SimpleNamespace(pad_token_id=None, eos_token_id=3)
        self.assertIs(prepare_checkpoint_config(config, tokenizer), config)
        self.assertEqual(config.pad_token_id, 3)

    def test_checkpoint_config_preserves_explicit_pad_token(self):
        config = SimpleNamespace(pad_token_id=17, eos_token_id=2)
        tokenizer = SimpleNamespace(pad_token_id=9, eos_token_id=3)
        prepare_checkpoint_config(config, tokenizer)
        self.assertEqual(config.pad_token_id, 17)

    def test_ouro_uses_equivalent_unit_proportional_rope_on_transformers_5(self):
        config = SimpleNamespace(
            model_type="ouro", pad_token_id=None, eos_token_id=2, rope_scaling=None
        )
        prepare_checkpoint_config(
            config, SimpleNamespace(pad_token_id=None, eos_token_id=2),
            rope_functions={"proportional": object()},
        )
        self.assertEqual(
            config.rope_scaling, {"rope_type": "proportional", "factor": 1.0}
        )

    def test_legacy_tied_weight_list_is_converted_to_alias_mapping(self):
        class LegacyTiedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 4)
                self.head = nn.Linear(4, 8, bias=False)
                self.head.weight = self.embed.weight
                self._tied_weights_keys = ["head.weight"]

        mapping = _legacy_tied_weight_mapping(LegacyTiedModel())
        self.assertEqual(mapping, {"head.weight": "embed.weight"})

    def test_observer_equivalence_accepts_identical_nan_mask_with_finite_logits(self):
        class StableNanLM(TinyLM):
            def forward(self, *args, **kwargs):
                output = super().forward(*args, **kwargs)
                output.logits[..., 0] = torch.nan
                return output

        result = validate_observer_equivalence(
            adapt(StableNanLM(), family="ouro"), torch.tensor([[1, 2]])
        )
        self.assertTrue(result["logits_equal"])
        self.assertTrue(result["greedy_equal"])
        self.assertGreater(result["finite_logits"], 0)
        self.assertGreater(result["nonfinite_logits"], 0)

    def test_observer_equivalence_rejects_all_nonfinite_logits(self):
        class AllNanLM(TinyLM):
            def forward(self, *args, **kwargs):
                output = super().forward(*args, **kwargs)
                output.logits.fill_(torch.nan)
                return output

        with self.assertRaisesRegex(AssertionError, "no finite logits"):
            validate_observer_equivalence(
                adapt(AllNanLM(), family="ouro"), torch.tensor([[1, 2]])
            )

    def test_observer_equivalence_rejects_changed_nan_mask(self):
        class ChangingNanLM(TinyLM):
            def forward(self, *args, **kwargs):
                output = super().forward(*args, **kwargs)
                output.logits[..., self.calls % 2] = torch.nan
                return output

        with self.assertRaisesRegex(AssertionError, "nonfinite_mask_equal=False"):
            validate_observer_equivalence(
                adapt(ChangingNanLM(), family="ouro"), torch.tensor([[1, 2]])
            )

    def test_bf16_occupancy_accepts_gemm_rounding_when_activation_delta_is_exact(self):
        class WideMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(32, 8192, bias=False)
                self.up_proj = nn.Linear(32, 8192, bias=False)
                self.down_proj = nn.Linear(8192, 32, bias=False)

            def forward(self, hidden):
                gate = torch.nn.functional.silu(self.gate_proj(hidden))
                return self.down_proj(gate * self.up_proj(hidden))

        class WideBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = WideMLP()

            def forward(self, hidden):
                return self.mlp(hidden)

        class WideLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.embed_tokens = nn.Embedding(16, 32)
                self.model.layers = nn.ModuleList([WideBlock()])
                self.model.total_ut_steps = 2
                self.lm_head = nn.Linear(32, 16, bias=False)

            def forward(self, input_ids, use_cache=False, **kwargs):
                del use_cache, kwargs
                hidden = self.model.embed_tokens(input_ids)
                for _ in range(self.model.total_ut_steps):
                    hidden = self.model.layers[0](hidden)
                return SimpleNamespace(logits=self.lm_head(hidden))

        torch.manual_seed(0)
        model = WideLM().to(torch.bfloat16)
        with torch.no_grad():
            model.model.layers[0].mlp.down_proj.weight.normal_(0, 0.02)
        adapter = adapt(model, family="ouro")
        result = validate_full_occupancy(
            adapter, torch.randint(0, 16, (4, 2)), adapter.sites[0], adapter.sites[1],
            cuda=NoCuda(),
        )
        self.assertEqual(result["rows_verified"], 4)
        self.assertTrue(result["selected_delta_matches_write_column"])

    def test_low_precision_causal_mask_keeps_zeros_and_replaces_dtype_minimum(self):
        class Masked(nn.Module):
            def _update_causal_mask(self, dtype):
                mask = torch.zeros(2, 2, dtype=dtype)
                mask[0, 1] = torch.finfo(dtype).min
                return mask

        model = Masked()
        stabilize_low_precision_attention(model)
        mask = model._update_causal_mask(torch.bfloat16)
        self.assertEqual(float(mask[0, 0]), 0)
        self.assertTrue(torch.isfinite(mask).all())
        self.assertLess(float(mask[0, 1]), -100)

    def test_requested_batch_is_fully_occupied_and_summary_is_structured(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                model="tiny", family="ouro", out=directory, batch_size=7,
                recurrences=None, max_tokens=1, seed=3,
            )
            result = run_smoke(args, TinyLM(), TinyTokenizer(), cuda=NoCuda())
            probe = result["occupancy_probe"]
            self.assertEqual(probe["requested_batch"], 7)
            self.assertEqual(probe["actual_batch"], 7)
            self.assertEqual(probe["rows_verified"], 7)
            self.assertEqual(probe["sequence_length"], 2)
            self.assertTrue(probe["selected_delta_matches_write_column"])
            self.assertTrue(probe["unselected_recurrence_unchanged"])
            self.assertGreater(probe["throughput_tokens_per_second"], 0)
            self.assertEqual(result["steering"]["neurons"], [0, 1, 2])
            self.assertGreater(result["steering"]["rows"], 0)
            self.assertTrue(Path(result["stats"]["path"]).is_file())

            summary = build_summary([{
                "family": "ouro", "status": result["status"],
                "requested_batch": 7, "actual_batch": probe["actual_batch"],
                "throughput_tokens_per_second": probe["throughput_tokens_per_second"],
            }])
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["models_total"], 1)
            self.assertEqual(summary["models_passed"], 1)
            self.assertIn("models", summary)

    def test_observer_equivalence_failure_is_loud(self):
        model = TinyLM(changing=True)
        adapter = adapt(model, family="ouro")
        with self.assertRaisesRegex(AssertionError, "no-op observer changed output"):
            validate_observer_equivalence(adapter, torch.tensor([[1, 2]]))

    def test_nvidia_csv_aggregation(self):
        rows = """2026/11/03 12:00:00.000, 0, 100, 0
2026/11/03 12:00:01.000, 0, 400, 20
2026/11/03 12:00:02.000, 0, 300, 80
2026/11/03 12:00:03.000, 0, 200, 100
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvidia.csv"
            path.write_text(rows, encoding="utf-8")
            metrics = aggregate_nvidia_csv(path)
        self.assertEqual(metrics["samples"], 4)
        self.assertEqual(metrics["peak_memory_mib"], 400)
        util = metrics["utilization_percent"]
        self.assertEqual(util["mean"], 50)
        self.assertEqual(util["p50"], 50)
        self.assertEqual(util["max"], 100)
        self.assertAlmostEqual(util["p95"], 97)

    def test_driver_has_exact_production_batches_and_summary(self):
        source = (Path(__file__).with_name("run_looped_smoke.sh"))
        text = source.read_text(encoding="utf-8")
        self.assertIn('"ouro|$OURO_NAS|256|"', text)
        self.assertIn('"nanbeige|$NANBEIGE_NAS|256|"', text)
        self.assertIn('"huginn|$HUGINN_NAS|64|$HUGINN_RECURRENCES"', text)
        self.assertIn("ouro/models/Ouro-2.6B", text)
        self.assertIn("j-lens-on-loop-nanbeige/models/Nanbeige4.2-3B", text)
        self.assertIn("j-lens-on-loop/models/huginn-0125", text)
        self.assertIn("CHECK_ONLY", text)
        self.assertIn("summary.json", text)
        self.assertIn("sleep 1", text)
        failed = build_summary([
            {"family": "ouro", "status": "passed"},
            {"family": "huginn", "status": "failed"},
        ])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["models_passed"], 1)


if __name__ == "__main__":
    unittest.main()
