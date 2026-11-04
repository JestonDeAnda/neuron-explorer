"""Recurrence-aware sites for Ouro, Nanbeige, and Huginn.

These tests build toy module trees. They do not load model weights.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sweep.looped_models import ExecutionSite, adapt, all_recurrences, targets_site


WIDTH = 4


class ToyMLP(nn.Module):
    def __init__(self, activation_name: str) -> None:
        super().__init__()
        setattr(self, activation_name, nn.Linear(WIDTH, WIDTH, bias=False))
        self.activation_name = activation_name

    @property
    def activation(self) -> nn.Module:
        return getattr(self, self.activation_name)


class SharedBlock(nn.Module):
    def __init__(self, activation_name: str, trace: list, kind: str, index: int) -> None:
        super().__init__()
        self.mlp = ToyMLP(activation_name)
        self.trace = trace
        self.kind = kind
        self.index = index

    def forward(self, hidden):
        module = self.mlp.activation
        self.trace.append((self.kind, self.index, module))
        return module(hidden)


def _blocks(count: int, activation_name: str, trace: list, kind: str) -> nn.ModuleList:
    return nn.ModuleList(
        SharedBlock(activation_name, trace, kind, index) for index in range(count)
    )


class ToyOuro(nn.Module):
    """48 shared decoder layers, each fired total_ut_steps times."""

    def __init__(self, physical: int = 48, loops: int = 4) -> None:
        super().__init__()
        self.trace: list = []
        self.config = type("Config", (), {"hidden_size": WIDTH, "total_ut_steps": loops})()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, WIDTH)
        self.model.layers = _blocks(physical, "down_proj", self.trace, "layer")
        self.model.norm = nn.LayerNorm(WIDTH)
        self.model.total_ut_steps = loops
        self.model.gradient_checkpointing = False
        self.lm_head = nn.Linear(WIDTH, 8, bias=False)

    def forward(self, input_ids, use_cache=False, **kwargs):
        del use_cache, kwargs
        hidden = self.model.embed_tokens(input_ids)
        for _ in range(self.model.total_ut_steps):
            for layer in self.model.layers:
                hidden = layer(hidden)
        return self.model.norm(hidden)


class ToyNanbeige(nn.Module):
    """22 shared decoder layers, each fired config.num_loops times."""

    def __init__(self, physical: int = 22, loops: int = 2) -> None:
        super().__init__()
        self.trace: list = []
        self.config = type("Config", (), {
            "hidden_size": WIDTH,
            "model_type": "nanbeige",
            "num_loops": loops,
        })()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, WIDTH)
        self.model.layers = _blocks(physical, "down_proj", self.trace, "layer")
        self.model.norm = nn.LayerNorm(WIDTH)
        self.model.gradient_checkpointing = False
        self.lm_head = nn.Linear(WIDTH, 8, bias=False)

    def forward(self, input_ids, use_cache=False, **kwargs):
        del use_cache, kwargs
        hidden = self.model.embed_tokens(input_ids)
        for _ in range(self.config.num_loops):
            for layer in self.model.layers:
                hidden = layer(hidden)
        return hidden


class ToyHuginn(nn.Module):
    """Prelude 2, core 4 repeated R times, coda 2. R comes from the adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.trace: list = []
        self.config = type("Config", (), {
            "n_embd": WIDTH,
            "test_time_noise": 0.25,
            "mean_recurrence": 32,
        })()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(8, WIDTH)
        self.transformer.prelude = _blocks(2, "proj", self.trace, "prelude")
        self.transformer.core_block = _blocks(4, "proj", self.trace, "core")
        self.transformer.coda = _blocks(2, "proj", self.trace, "coda")
        self.transformer.ln_f = nn.LayerNorm(WIDTH)
        self.lm_head = nn.Linear(WIDTH, 8, bias=False)
        self.gradient_checkpointing = False
        self.seen_num_steps = None
        self.noise = None

    def forward(self, input_ids, num_steps, use_cache=False, **kwargs):
        del use_cache, kwargs
        self.seen_num_steps = num_steps
        self.noise = torch.randn(4)
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.prelude:
            hidden = block(hidden)
        no_grad_steps, grad_steps = num_steps
        state = hidden
        for _ in range(no_grad_steps + grad_steps):
            for block in self.transformer.core_block:
                state = block(state)
        for block in self.transformer.coda:
            state = block(state)
        return self.transformer.ln_f(state)


def _ids():
    return torch.tensor([[1, 2, 3]])


def _uniform_sites(family: str, physical: int, loops: int) -> list[ExecutionSite]:
    sites = []
    virtual = 0
    for recurrence in range(loops):
        for index in range(physical):
            sites.append(ExecutionSite("layer", index, recurrence, virtual))
            virtual += 1
    return sites


def _huginn_sites(recurrences: int) -> list[ExecutionSite]:
    sites = []
    virtual = 0
    for index in range(2):
        sites.append(ExecutionSite("prelude", index, 0, virtual))
        virtual += 1
    for recurrence in range(recurrences):
        for index in range(4):
            sites.append(ExecutionSite("core", index, recurrence, virtual))
            virtual += 1
    for index in range(2):
        sites.append(ExecutionSite("coda", index, 0, virtual))
        virtual += 1
    return sites


class LoopedAdapterTests(unittest.TestCase):
    def test_execution_site_identity_is_immutable(self):
        site = ExecutionSite("layer", 3, 1, 51)
        twin = ExecutionSite("layer", 3, 1, 51)
        other = ExecutionSite("layer", 3, 2, 99)
        self.assertEqual(site, twin)
        self.assertNotEqual(site, other)
        self.assertEqual(hash(site), hash(twin))
        with self.assertRaises(AttributeError):
            site.recurrence = 0

    def test_ouro_48x4_site_order(self):
        first = adapt(ToyOuro(), family="ouro").sites
        second = adapt(ToyOuro(), family="ouro").sites
        expected = _uniform_sites("ouro", 48, 4)
        self.assertEqual(list(first), expected)
        self.assertEqual(list(second), expected)
        self.assertEqual(first[0], ExecutionSite("layer", 0, 0, 0))
        self.assertEqual(first[47], ExecutionSite("layer", 47, 0, 47))
        self.assertEqual(first[48], ExecutionSite("layer", 0, 1, 48))
        self.assertEqual(first[191], ExecutionSite("layer", 47, 3, 191))
        self.assertEqual(len(first), 192)

    def test_nanbeige_22x2_site_order(self):
        sites = adapt(ToyNanbeige(), family="nanbeige").sites
        expected = _uniform_sites("nanbeige", 22, 2)
        self.assertEqual(list(sites), expected)
        self.assertEqual(sites[0], ExecutionSite("layer", 0, 0, 0))
        self.assertEqual(sites[21], ExecutionSite("layer", 21, 0, 21))
        self.assertEqual(sites[22], ExecutionSite("layer", 0, 1, 22))
        self.assertEqual(sites[43], ExecutionSite("layer", 21, 1, 43))
        self.assertEqual(len(sites), 44)

    def test_huginn_prelude2_core4_coda2_site_order(self):
        for recurrences, length, coda0 in ((4, 20, 18), (2, 12, 10)):
            with self.subTest(recurrences=recurrences):
                sites = adapt(
                    ToyHuginn(), family="huginn", recurrences=recurrences, seed=0
                ).sites
                expected = _huginn_sites(recurrences)
                self.assertEqual(list(sites), expected)
                self.assertEqual(len(sites), length)
                self.assertEqual(sites[0], ExecutionSite("prelude", 0, 0, 0))
                self.assertEqual(sites[1], ExecutionSite("prelude", 1, 0, 1))
                self.assertEqual(sites[2], ExecutionSite("core", 0, 0, 2))
                self.assertEqual(sites[5], ExecutionSite("core", 3, 0, 5))
                self.assertEqual(sites[6], ExecutionSite("core", 0, 1, 6))
                self.assertEqual(sites[coda0], ExecutionSite("coda", 0, 0, coda0))
                self.assertEqual(
                    sites[coda0 + 1], ExecutionSite("coda", 1, 0, coda0 + 1)
                )
                self.assertEqual(sites[-1].virtual_index, length - 1)

    def test_one_call_per_site_matches_real_invocations(self):
        cases = (
            ("ouro", ToyOuro(), {}, _uniform_sites("ouro", 48, 4)),
            ("nanbeige", ToyNanbeige(), {}, _uniform_sites("nanbeige", 22, 2)),
            ("huginn", ToyHuginn(), {"recurrences": 4, "seed": 1}, _huginn_sites(4)),
        )
        for family, model, kwargs, expected in cases:
            with self.subTest(family=family):
                adapter = adapt(model, family=family, **kwargs)
                with adapter.route() as calls:
                    adapter.forward(_ids())
                self.assertEqual(calls, expected)
                self.assertEqual(len(calls), len(model.trace))
                seen = {}
                for position, (site, (kind, index, module)) in enumerate(zip(calls, model.trace)):
                    recurrence = seen.get((kind, index), 0)
                    self.assertEqual((site.kind, site.physical_index, site.recurrence), (kind, index, recurrence))
                    self.assertEqual(site.virtual_index, position)
                    self.assertIs(adapter.activation_module(site), module)
                    seen[(kind, index)] = recurrence + 1

    def test_targeted_site_predicate_selects_one_firing(self):
        cases = (
            ("ouro", ToyOuro(), {}, 48),
            ("nanbeige", ToyNanbeige(), {}, 27),
            ("huginn", ToyHuginn(), {"recurrences": 4, "seed": 2}, 8),
        )
        for family, model, kwargs, virtual in cases:
            with self.subTest(family=family):
                adapter = adapt(model, family=family, **kwargs)
                site = adapter.sites[virtual]
                same_block_other_loop = next(
                    other for other in adapter.sites
                    if other.kind == site.kind
                    and other.physical_index == site.physical_index
                    and other.recurrence != site.recurrence
                )
                predicate = targets_site(site)
                self.assertTrue(predicate(site))
                self.assertTrue(predicate(ExecutionSite(
                    site.kind, site.physical_index, site.recurrence, site.virtual_index
                )))
                self.assertFalse(predicate(same_block_other_loop))
                with adapter.route(predicate) as calls:
                    adapter.forward(_ids())
                self.assertEqual(calls, [site])

    def test_all_recurrences_predicate_selects_every_firing_of_one_block(self):
        expectations = {
            "ouro": [0, 48, 96, 144],
            "nanbeige": [5, 27],
            "huginn": [4, 8, 12, 16],
        }
        cases = (
            ("ouro", ToyOuro(), {}, 0),
            ("nanbeige", ToyNanbeige(), {}, 5),
            ("huginn", ToyHuginn(), {"recurrences": 4, "seed": 3}, 4),
        )
        for family, model, kwargs, virtual in cases:
            with self.subTest(family=family):
                adapter = adapt(model, family=family, **kwargs)
                site = adapter.sites[virtual]
                neighbor = next(
                    other for other in adapter.sites
                    if other.virtual_index != site.virtual_index
                    and (other.kind != site.kind or other.physical_index != site.physical_index)
                )
                predicate = all_recurrences(site.kind, site.physical_index)
                self.assertFalse(predicate(neighbor))
                with adapter.route(predicate) as calls:
                    adapter.forward(_ids())
                self.assertEqual([call.virtual_index for call in calls], expectations[family])
                self.assertTrue(all(
                    call.kind == site.kind and call.physical_index == site.physical_index
                    for call in calls
                ))
                self.assertEqual([call.recurrence for call in calls], list(range(len(calls))))

    def test_routing_resets_at_each_top_level_forward(self):
        cases = (
            ("ouro", ToyOuro(), {}, _uniform_sites("ouro", 48, 4)),
            ("nanbeige", ToyNanbeige(), {}, _uniform_sites("nanbeige", 22, 2)),
            ("huginn", ToyHuginn(), {"recurrences": 4, "seed": 4}, _huginn_sites(4)),
        )
        for family, model, kwargs, expected in cases:
            with self.subTest(family=family):
                adapter = adapt(model, family=family, **kwargs)
                with adapter.route() as calls:
                    adapter.forward(_ids())
                    adapter.forward(_ids())
                self.assertEqual(calls, expected + expected)
                self.assertEqual(calls[len(expected)].recurrence, 0)
                self.assertEqual(calls[len(expected)].virtual_index, 0)

    def test_activation_input_and_write_matrix_follow_the_physical_mlp(self):
        ouro = ToyOuro()
        ouro_adapter = adapt(ouro, family="ouro")
        ouro_site = ouro_adapter.sites[48 + 3]
        ouro_repeat = ouro_adapter.sites[3]
        ouro_module = ouro.model.layers[3].mlp.down_proj
        self.assertIs(ouro_adapter.activation_module(ouro_site), ouro_module)
        self.assertIs(ouro_adapter.activation_module(ouro_repeat), ouro_module)
        self.assertIs(ouro_adapter.write_matrix(ouro_site), ouro_module.weight)
        self.assertIsNot(
            ouro_adapter.activation_module(ouro_site),
            ouro_adapter.activation_module(ouro_adapter.sites[4]),
        )

        nanbeige = ToyNanbeige()
        nanbeige_adapter = adapt(nanbeige, family="nanbeige")
        nanbeige_module = nanbeige.model.layers[21].mlp.down_proj
        self.assertIs(nanbeige_adapter.activation_module(nanbeige_adapter.sites[21]), nanbeige_module)
        self.assertIs(nanbeige_adapter.write_matrix(nanbeige_adapter.sites[43]), nanbeige_module.weight)

        huginn = ToyHuginn()
        huginn_adapter = adapt(huginn, family="huginn", recurrences=4, seed=5)
        core = huginn_adapter.sites[6]
        self.assertEqual((core.kind, core.physical_index, core.recurrence), ("core", 0, 1))
        core_module = huginn.transformer.core_block[0].mlp.proj
        prelude_module = huginn.transformer.prelude[0].mlp.proj
        self.assertIs(huginn_adapter.activation_module(core), core_module)
        self.assertIs(huginn_adapter.activation_module(huginn_adapter.sites[2]), core_module)
        self.assertIs(huginn_adapter.write_matrix(huginn_adapter.sites[0]), prelude_module.weight)
        self.assertIsNot(core_module, prelude_module)
        matrix = huginn_adapter.write_matrix(core)
        self.assertEqual(tuple(matrix.shape), (core_module.out_features, core_module.in_features))

    def test_huginn_fixes_num_steps_seed_and_test_time_noise(self):
        model = ToyHuginn()
        self.assertEqual(model.config.test_time_noise, 0.25)
        adapter = adapt(model, family="huginn", recurrences=4, seed=123)
        self.assertEqual(adapter.num_steps, (0, 4))
        self.assertEqual(adapter.seed, 123)
        self.assertEqual(model.config.test_time_noise, 0)
        self.assertEqual(adapter.test_time_noise, 0)

        torch.manual_seed(5)
        sentinel = torch.randn(4).clone()
        adapter.forward(_ids(), num_steps=(9, 9))
        self.assertEqual(model.seen_num_steps, (0, 4))
        torch.manual_seed(123)
        self.assertTrue(torch.equal(model.noise, torch.randn(4)))
        torch.manual_seed(5)
        self.assertTrue(torch.equal(sentinel, torch.randn(4)))

    def test_observe_callback_targets_one_firing_and_reads_activation_input(self):
        # Layer 0 doubles, layer 1 triples. Virtual 2 is layer 0 on its second firing:
        # activation input is 6x the embedding, and the block output is 12x.
        # A callback that sees every recurrence, or the down-projection output, gets a different tensor.
        model = ToyOuro(physical=2, loops=2)
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(
                torch.arange(8 * WIDTH, dtype=torch.float32).reshape(8, WIDTH)
            )
            model.model.layers[0].mlp.down_proj.weight.copy_(2 * torch.eye(WIDTH))
            model.model.layers[1].mlp.down_proj.weight.copy_(3 * torch.eye(WIDTH))
        adapter = adapt(model, family="ouro")
        site = adapter.sites[2]
        ids = torch.tensor([[1, 2, 3]])
        expected_input = 6 * model.model.embed_tokens(ids)
        expected_output = 12 * model.model.embed_tokens(ids)
        activations = []
        residuals = []
        with adapter.observe(
            lambda seen, activation: activations.append((seen, activation.detach().clone())),
            targets_site(site),
            on_block_output=lambda seen, output: residuals.append((seen, output.detach().clone())),
        ):
            adapter.forward(ids)
        self.assertEqual([seen for seen, _ in activations], [site])
        self.assertEqual([seen for seen, _ in residuals], [site])
        self.assertTrue(torch.equal(activations[0][1], expected_input))
        self.assertTrue(torch.equal(residuals[0][1], expected_output))

    def test_block_output_follows_each_execution_site(self):
        model = ToyOuro(physical=2, loops=2)
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(
                torch.arange(8 * WIDTH, dtype=torch.float32).reshape(8, WIDTH)
            )
            model.model.layers[0].mlp.down_proj.weight.copy_(2 * torch.eye(WIDTH))
            model.model.layers[1].mlp.down_proj.weight.copy_(3 * torch.eye(WIDTH))
        adapter = adapt(model, family="ouro")
        ids = torch.tensor([[1, 2, 3]])
        residuals = []
        with adapter.observe(
            on_block_output=lambda site, output: residuals.append((site, output.detach().clone())),
        ):
            adapter.forward(ids)
        embed = model.model.embed_tokens(ids)
        expected = (
            (adapter.sites[0], 2 * embed),
            (adapter.sites[1], 6 * embed),
            (adapter.sites[2], 12 * embed),
            (adapter.sites[3], 36 * embed),
        )
        self.assertEqual([site for site, _ in residuals], [site for site, _ in expected])
        for (_site, got), (_expected_site, want) in zip(residuals, expected):
            self.assertTrue(torch.equal(got, want))

    def test_physical_block_is_the_enclosing_module_shared_across_recurrences(self):
        ouro = ToyOuro(physical=2, loops=2)
        ouro_adapter = adapt(ouro, family="ouro")
        self.assertIs(ouro_adapter.physical_block(ouro_adapter.sites[0]), ouro.model.layers[0])
        self.assertIs(ouro_adapter.physical_block(ouro_adapter.sites[2]), ouro.model.layers[0])
        self.assertIs(ouro_adapter.physical_block(ouro_adapter.sites[1]), ouro.model.layers[1])

        nanbeige = ToyNanbeige(physical=2, loops=2)
        nanbeige_adapter = adapt(nanbeige, family="nanbeige")
        self.assertIs(
            nanbeige_adapter.physical_block(nanbeige_adapter.sites[2]),
            nanbeige.model.layers[0],
        )

        huginn = ToyHuginn()
        huginn_adapter = adapt(huginn, family="huginn", recurrences=2, seed=0)
        self.assertIs(huginn_adapter.physical_block(huginn_adapter.sites[0]), huginn.transformer.prelude[0])
        core = next(
            site for site in huginn_adapter.sites
            if site.kind == "core" and site.physical_index == 0 and site.recurrence == 1
        )
        self.assertIs(huginn_adapter.physical_block(core), huginn.transformer.core_block[0])
        self.assertIs(huginn_adapter.physical_block(huginn_adapter.sites[-1]), huginn.transformer.coda[1])
        self.assertIsNot(
            huginn_adapter.physical_block(core),
            huginn_adapter.activation_module(core),
        )

    def test_nested_and_repeated_observe_resets_counts_and_removes_hooks(self):
        model = ToyOuro(physical=1, loops=2)
        adapter = adapt(model, family="ouro")
        site = adapter.sites[1]
        base = _hook_count(model)
        outer = []
        inner = []
        with adapter.observe(lambda seen, _activation: outer.append(seen)):
            outer_hooks = _hook_count(model)
            self.assertGreater(outer_hooks, base)
            with adapter.observe(
                lambda seen, _activation: inner.append(seen),
                targets_site(site),
            ):
                self.assertGreater(_hook_count(model), outer_hooks)
                adapter.forward(_ids())
                adapter.forward(_ids())
            self.assertEqual(_hook_count(model), outer_hooks)
        self.assertEqual(outer, list(adapter.sites) + list(adapter.sites))
        self.assertEqual(inner, [site, site])
        self.assertEqual(outer[len(adapter.sites)].recurrence, 0)
        self.assertEqual(_hook_count(model), base)

        with self.assertRaises(RuntimeError):
            with adapter.observe(lambda _site, _activation: None):
                raise RuntimeError("stop")
        self.assertEqual(_hook_count(model), base)

        def explode(_site, _activation):
            raise RuntimeError("from callback")

        with self.assertRaises(RuntimeError):
            with adapter.observe(explode):
                adapter.forward(_ids())
        self.assertEqual(_hook_count(model), base)

        repeated = []
        with adapter.observe(lambda seen, _activation: repeated.append(seen.recurrence)):
            adapter.forward(_ids())
        self.assertEqual(_hook_count(model), base)
        with adapter.observe(lambda seen, _activation: repeated.append(seen.recurrence)):
            adapter.forward(_ids())
        self.assertEqual(repeated, [0, 1, 0, 1])
        self.assertEqual(_hook_count(model), base)

    def test_huginn_observe_pairs_each_block_output_with_its_site(self):
        model = ToyHuginn()
        adapter = adapt(model, family="huginn", recurrences=2, seed=7)
        seen = []
        blocks = []
        with adapter.observe(
            lambda site, _activation: seen.append(site),
            on_block_output=lambda site, _output: blocks.append(site),
        ):
            adapter.forward(_ids())
            adapter.forward(_ids())
        self.assertEqual(seen, list(adapter.sites) + list(adapter.sites))
        self.assertEqual(blocks, seen)
        self.assertEqual(seen[len(adapter.sites)].recurrence, 0)
        self.assertEqual(_hook_count(model), 0)

    def test_block_output_callback_receives_the_residual_tensor(self):
        # Nanbeige layers return (hidden, ...). The residual is that hidden tensor.
        model = ToyNanbeige(physical=1, loops=1)
        layer = model.model.layers[0]
        real_forward = layer.forward

        def forward(hidden):
            return (real_forward(hidden), None)

        layer.forward = forward
        adapter = adapt(model, family="nanbeige")
        seen = []
        with adapter.observe(on_block_output=lambda _site, output: seen.append(output)):
            adapter.forward(_ids())
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], torch.Tensor)
        self.assertEqual(tuple(seen[0].shape), (1, 3, WIDTH))

    def test_observe_reports_a_forward_that_skips_a_scheduled_firing(self):
        class ShortOuro(ToyOuro):
            def forward(self, input_ids, use_cache=False, **kwargs):
                del use_cache, kwargs
                hidden = self.model.embed_tokens(input_ids)
                for layer in self.model.layers:
                    hidden = layer(hidden)
                return hidden

        model = ShortOuro(physical=1, loops=2)
        adapter = adapt(model, family="ouro")
        with self.assertRaises(RuntimeError) as caught:
            with adapter.observe(lambda _site, _activation: None):
                adapter.forward(_ids())
        self.assertIn("fired 1 times", str(caught.exception))
        self.assertIn("expected 2", str(caught.exception))
        self.assertEqual(_hook_count(model), 0)


def _hook_count(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        total += len(getattr(module, "_forward_hooks", {}))
        total += len(getattr(module, "_forward_pre_hooks", {}))
    return total


def _batch_edits(neurons, deltas, residuals=None):
    from sweep.looped_models import BatchEdits

    return BatchEdits(
        torch.tensor(neurons, dtype=torch.long),
        torch.tensor(deltas, dtype=torch.float32),
        None if residuals is None else torch.tensor(residuals, dtype=torch.float32),
    )


def _spy_activation(module):
    """Clones seen by a pre-hook registered before and after intervention."""
    before, before_ptr, after, live = [], [], [], []

    def earlier(_module, inputs):
        live.append(inputs[0])
        before_ptr.append(inputs[0].data_ptr())
        before.append(inputs[0].detach().clone())
        return None

    def later(_module, inputs):
        after.append((inputs[0].detach().clone(), inputs[0].data_ptr()))
        return None

    return before, before_ptr, after, live, earlier, later


class InterventionTests(unittest.TestCase):
    def _identity(self, physical, loops):
        model = ToyOuro(physical=physical, loops=loops)
        with torch.no_grad():
            for layer in model.model.layers:
                layer.mlp.down_proj.weight.copy_(torch.eye(WIDTH))
        return model, adapt(model, family="ouro")

    def test_row_specific_deltas_change_only_the_addressed_activation(self):
        # A replacement that edited every row, or wrote the delta through a
        # shared column, would move the other row or the other neurons.
        model, adapter = self._identity(physical=1, loops=1)
        site = adapter.sites[0]
        edits = {site: _batch_edits([1, 3], [5.0, -2.0])}
        module = adapter.activation_module(site)
        before, _before_ptr, after, live, earlier, later = _spy_activation(module)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        outer = module.register_forward_pre_hook(earlier)
        try:
            with adapter.intervene(edits):
                inner = module.register_forward_pre_hook(later)
                try:
                    adapter.forward(ids)
                finally:
                    inner.remove()
        finally:
            outer.remove()
        self.assertEqual(len(before), 1)
        delta = after[0][0] - before[0]
        expected = torch.zeros_like(delta)
        expected[0, :, 1] = 5
        expected[1, :, 3] = -2
        self.assertTrue(torch.equal(delta, expected))
        self.assertNotEqual(after[0][1], live[0].data_ptr())
        self.assertTrue(torch.equal(live[0], before[0]))

    def test_unselected_recurrence_keeps_the_incoming_activation(self):
        # Editing every firing of the shared down projection would change
        # recurrence 1 even though only virtual index 0 is selected.
        model, adapter = self._identity(physical=1, loops=2)
        selected = adapter.sites[0]
        other = adapter.sites[1]
        edits = {selected: _batch_edits([0], [4.0])}
        module = adapter.activation_module(selected)
        self.assertIs(module, adapter.activation_module(other))
        before, before_ptr, after, _live, earlier, later = _spy_activation(module)
        outer = module.register_forward_pre_hook(earlier)
        try:
            with adapter.intervene(edits):
                inner = module.register_forward_pre_hook(later)
                try:
                    adapter.forward(_ids())
                    adapter.forward(_ids())
                finally:
                    inner.remove()
        finally:
            outer.remove()
        self.assertEqual(len(before), 4)
        for firing, (edited, ptr) in enumerate(after):
            recurrence = firing % 2
            if recurrence == 0:
                delta = edited - before[firing]
                expected = torch.zeros_like(delta)
                expected[:, :, 0] = 4
                self.assertTrue(torch.equal(delta, expected), firing)
                self.assertNotEqual(ptr, before_ptr[firing])
            else:
                self.assertEqual(ptr, before_ptr[firing], firing)
                self.assertTrue(torch.equal(edited, before[firing]))

    def test_all_recurrences_use_each_sites_own_delta(self):
        model, adapter = self._identity(physical=2, loops=2)
        first, second = adapter.sites[0], adapter.sites[2]
        self.assertEqual((first.physical_index, first.recurrence), (0, 0))
        self.assertEqual((second.physical_index, second.recurrence), (0, 1))
        untouched = adapter.sites[1]
        edits = {
            first: _batch_edits([2], [1.5]),
            second: _batch_edits([2], [-4.0]),
        }
        module = adapter.activation_module(first)
        neighbor = adapter.activation_module(untouched)
        before, _before_ptr, after, _live, earlier, later = _spy_activation(module)
        neighbor_before, neighbor_after = [], []
        outer = module.register_forward_pre_hook(earlier)
        neighbor_outer = neighbor.register_forward_pre_hook(
            lambda _m, inputs: neighbor_before.append(inputs[0].data_ptr()) or None
        )
        try:
            with adapter.intervene(edits):
                inner = module.register_forward_pre_hook(later)
                neighbor_inner = neighbor.register_forward_pre_hook(
                    lambda _m, inputs: neighbor_after.append(inputs[0].data_ptr()) or None
                )
                try:
                    adapter.forward(_ids())
                finally:
                    inner.remove()
                    neighbor_inner.remove()
        finally:
            outer.remove()
            neighbor_outer.remove()
        # Layer 0 fires at calls 0 and 2 (layer 1 sits between them).
        self.assertEqual(len(before), 2)
        first_delta = after[0][0] - before[0]
        second_delta = after[1][0] - before[1]
        torch.testing.assert_close(first_delta[:, :, 2], torch.full((1, 3), 1.5), rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(first_delta[:, :, :2], torch.zeros(1, 3, 2)))
        self.assertTrue(torch.equal(first_delta[:, :, 3], torch.zeros(1, 3)))
        torch.testing.assert_close(second_delta[:, :, 2], torch.full((1, 3), -4.0), rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.count_nonzero(second_delta[:, :, :2]) == 0)
        self.assertEqual(neighbor_before, neighbor_after)

    def test_random_residual_is_added_at_the_projection_output(self):
        # Adding the control on the activation, or in place on the linear
        # output, would change the saved input or the caller's output tensor.
        model, adapter = self._identity(physical=1, loops=1)
        site = adapter.sites[0]
        vector = [0.5, -1.0, 2.0, 0.0]
        edits = {site: _batch_edits([-1, -1], [0.0, 0.0], residuals=[[0.0] * WIDTH, vector])}
        module = adapter.activation_module(site)
        raw, edited, inputs = [], [], []

        def earlier_out(_module, _inputs, output):
            raw.append(output.detach().clone())
            return None

        def later_out(_module, _inputs, output):
            edited.append(output.detach().clone())
            return None

        def earlier_in(_module, hook_inputs):
            inputs.append(hook_inputs[0].data_ptr())
            return None

        def later_in(_module, hook_inputs):
            inputs.append(hook_inputs[0].data_ptr())
            return None

        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        pre = module.register_forward_pre_hook(earlier_in)
        post = module.register_forward_hook(earlier_out)
        try:
            with adapter.intervene(edits):
                pre2 = module.register_forward_pre_hook(later_in)
                post2 = module.register_forward_hook(later_out)
                try:
                    adapter.forward(ids)
                finally:
                    pre2.remove()
                    post2.remove()
        finally:
            pre.remove()
            post.remove()
        self.assertEqual(inputs[0], inputs[1])
        delta = edited[0] - raw[0]
        self.assertTrue(torch.equal(delta[0], torch.zeros_like(delta[0])))
        expected = torch.tensor(vector, dtype=delta.dtype)
        torch.testing.assert_close(delta[1], expected.expand_as(delta[1]), rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(raw[0], module(model.model.embed_tokens(ids))))

    def test_empty_and_zero_edits_match_the_unhooked_forward(self):
        model = ToyOuro(physical=2, loops=2)
        adapter = adapt(model, family="ouro")
        ids = _ids()
        plain = adapter.forward(ids).clone()
        with adapter.intervene({}):
            empty = adapter.forward(ids)
        zeros = {
            adapter.sites[1]: _batch_edits(
                [-1],
                [0.0],
                residuals=[[0.0] * WIDTH],
            )
        }
        with adapter.intervene(zeros):
            zeroed = adapter.forward(ids)
        self.assertTrue(torch.equal(plain, empty))
        self.assertTrue(torch.equal(plain, zeroed))
        self.assertEqual(_hook_count(model), 0)

    def test_intervene_drops_hooks_after_success_and_exceptions(self):
        model, adapter = self._identity(physical=1, loops=1)
        site = adapter.sites[0]
        edits = {site: _batch_edits([0], [1.0])}
        base = _hook_count(model)
        with adapter.intervene(edits):
            self.assertGreater(_hook_count(model), base)
            adapter.forward(_ids())
        self.assertEqual(_hook_count(model), base)

        with self.assertRaises(RuntimeError):
            with adapter.intervene(edits):
                raise RuntimeError("stop")
        self.assertEqual(_hook_count(model), base)

        mismatch = {site: _batch_edits([0, 1], [1.0, 1.0])}
        with self.assertRaises(ValueError):
            with adapter.intervene(mismatch):
                adapter.forward(_ids())
        self.assertEqual(_hook_count(model), base)
        adapter.forward(_ids())
        self.assertEqual(_hook_count(model), base)

    def test_huginn_intervention_keeps_num_steps_seed_and_zero_noise(self):
        model = ToyHuginn()
        adapter = adapt(model, family="huginn", recurrences=2, seed=11)
        self.assertEqual(model.config.test_time_noise, 0.0)
        site = next(
            item for item in adapter.sites
            if item.kind == "core" and item.physical_index == 0 and item.recurrence == 0
        )
        edits = {site: _batch_edits([0], [0.25])}
        torch.manual_seed(5)
        sentinel = torch.randn(4).clone()
        with adapter.intervene(edits):
            adapter.forward(_ids())
            first_noise = model.noise.clone()
            first_steps = model.seen_num_steps
            model.config.test_time_noise = 0.9
            adapter.forward(_ids())
            second_noise = model.noise.clone()
            second_steps = model.seen_num_steps
        self.assertEqual(first_steps, (0, 2))
        self.assertEqual(second_steps, (0, 2))
        self.assertEqual(model.config.test_time_noise, 0.0)
        torch.manual_seed(11)
        expected = torch.randn(4)
        self.assertTrue(torch.equal(first_noise, expected))
        self.assertTrue(torch.equal(second_noise, expected))
        torch.manual_seed(5)
        self.assertTrue(torch.equal(sentinel, torch.randn(4)))
        self.assertEqual(_hook_count(model), 0)


if __name__ == "__main__":
    unittest.main()
