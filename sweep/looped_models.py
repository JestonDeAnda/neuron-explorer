"""Execution sites for recurrent sweeps.

Qwen sweeps hook ``language_model.layers[i].mlp.down_proj`` once: that module
is one layer. Ouro, Nanbeige, and Huginn reuse each physical MLP across
recurrences, so the same module fires several times in one forward. An
``ExecutionSite`` names one of those firings. Routing counts firings of each
physical activation module and resets that count at the start of every
top-level forward.

The activation input is the tensor existing sweeps read as the neuron
activation: ``down_proj`` for Ouro and Nanbeige, ``mlp.proj`` for Huginn
(not ``attn.proj``). The write matrix is that module's weight; column ``n``
is the residual direction neuron ``n`` adds, and every recurrence of a
physical block shares the same matrix.

``observe`` reports the current ``ExecutionSite`` at each of those
invocations and can hand back the enclosing physical block's output, which
is the residual stream leaving that firing. Each ``observe`` or ``route``
owns its hooks and its firing counts. Leaving the context, including by an
exception, removes the hooks. A nested context counts the same forward
again from recurrence 0; it does not advance the outer count.

``intervene`` applies a :class:`BatchEdits` at chosen firings. A real-neuron
row adds its delta to that activation column at every sequence position.
The pre-hook returns a cloned input, so earlier hooks keep the original
tensor. A residual row is added on the projection output and returned as a
new tensor. Firings that are not in the edit map are left untouched.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ExecutionSite:
    """One firing of one shared block.

    ``kind`` is the physical block role (``layer``, ``prelude``, ``core``, or
    ``coda``). ``physical_index`` selects the module inside that role.
    ``recurrence`` is the 0-based firing of that module during the current
    top-level forward. ``virtual_index`` is the dense schedule position of
    that firing: loop-major for Ouro and Nanbeige, and prelude, then each
    core pass, then coda for Huginn.
    """

    kind: str
    physical_index: int
    recurrence: int
    virtual_index: int


def targets_site(site: ExecutionSite) -> Callable[[ExecutionSite], bool]:
    """True only for this firing, not for other recurrences of the same block."""
    if not isinstance(site, ExecutionSite):
        raise TypeError("target must be an ExecutionSite")
    return lambda candidate: candidate == site


def all_recurrences(kind: str, physical_index: int) -> Callable[[ExecutionSite], bool]:
    """True for every firing of one physical block."""
    return lambda site: site.kind == kind and site.physical_index == physical_index


@dataclass(frozen=True, slots=True)
class BatchEdits:
    """Per-row edits for one execution site.

    ``neurons[row]`` selects the activation column that receives
    ``deltas[row]`` at every sequence position. A negative index skips that
    row. ``residuals`` is an optional ``[batch, hidden]`` tensor added to the
    projection output after the write; ``None`` adds nothing there.
    """

    neurons: torch.Tensor
    deltas: torch.Tensor
    residuals: torch.Tensor | None = None


def enumerate_sites(family: str, **spec: int) -> tuple[ExecutionSite, ...]:
    """Sites in forward order for one of the three families.

    Ouro and Nanbeige take ``n_layers`` and ``n_loops``. Huginn takes
    ``n_prelude``, ``n_core``, ``n_coda``, and ``recurrences``.
    """
    family = family.lower()
    if family in ("ouro", "nanbeige"):
        n_layers = spec.pop("n_layers")
        n_loops = spec.pop("n_loops")
        _reject_unknown(spec)
        return _layer_sites(n_layers, n_loops)
    if family == "huginn":
        n_prelude = spec.pop("n_prelude")
        n_core = spec.pop("n_core")
        n_coda = spec.pop("n_coda")
        recurrences = spec.pop("recurrences")
        _reject_unknown(spec)
        return _huginn_sites(n_prelude, n_core, n_coda, recurrences)
    raise ValueError("family must be one of: ouro, huginn, nanbeige")


def _reject_unknown(spec: dict) -> None:
    if spec:
        names = ", ".join(sorted(spec))
        raise TypeError(f"unexpected enumerate_sites argument(s): {names}")


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _layer_sites(n_layers: int, n_loops: int) -> tuple[ExecutionSite, ...]:
    n_layers = _positive("n_layers", n_layers)
    n_loops = _positive("n_loops", n_loops)
    return tuple(
        ExecutionSite("layer", virtual % n_layers, virtual // n_layers, virtual)
        for virtual in range(n_layers * n_loops)
    )


def _huginn_sites(n_prelude: int, n_core: int, n_coda: int, recurrences: int) -> tuple[ExecutionSite, ...]:
    n_prelude = _positive("n_prelude", n_prelude)
    n_core = _positive("n_core", n_core)
    n_coda = _positive("n_coda", n_coda)
    recurrences = _positive("recurrences", recurrences)
    sites = []
    virtual = 0
    for index in range(n_prelude):
        sites.append(ExecutionSite("prelude", index, 0, virtual))
        virtual += 1
    for recurrence in range(recurrences):
        for index in range(n_core):
            sites.append(ExecutionSite("core", index, recurrence, virtual))
            virtual += 1
    for index in range(n_coda):
        sites.append(ExecutionSite("coda", index, 0, virtual))
        virtual += 1
    return tuple(sites)


class LoopedAdapter:
    """Physical MLP locations plus a per-forward firing count."""

    def __init__(
        self,
        model: nn.Module,
        family: str,
        sites: tuple[ExecutionSite, ...],
        modules: dict[tuple[str, int], nn.Module],
        blocks: dict[tuple[str, int], nn.Module],
        firings: dict[tuple[str, int], int],
    ) -> None:
        self.model = model
        self.family = family
        self.sites = sites
        self._modules = modules
        self._blocks = blocks
        self._firings = firings
        grouped: dict[tuple[str, int], list[ExecutionSite]] = {}
        for site in sites:
            grouped.setdefault((site.kind, site.physical_index), []).append(site)
        self._by_firing = grouped

    def activation_module(self, site: ExecutionSite) -> nn.Module:
        """Module whose forward input is the neuron activation at ``site``."""
        self._check(site)
        return self._modules[(site.kind, site.physical_index)]

    def write_matrix(self, site: ExecutionSite) -> torch.Tensor:
        """Shared output weight. Column ``n`` is neuron ``n``'s residual write."""
        return self.activation_module(site).weight

    def physical_block(self, site: ExecutionSite) -> nn.Module:
        """Enclosing block. Its forward output is the residual after this firing."""
        self._check(site)
        return self._blocks[(site.kind, site.physical_index)]

    @contextmanager
    def observe(
        self,
        callback: Callable[[ExecutionSite, torch.Tensor], None] | None = None,
        predicate: Callable[[ExecutionSite], bool] | None = None,
        *,
        on_block_output: Callable[[ExecutionSite, torch.Tensor], None] | None = None,
    ) -> Iterator[list[ExecutionSite]]:
        """Call ``callback(site, activation)`` at each activation-module firing.

        ``activation`` is that module's forward input, the post-SwiGLU tensor
        read by the down projection. ``on_block_output(site, residual)`` runs
        when the enclosing physical block returns the same firing; ``residual``
        is the hidden state, unwrapped when the block returns a tuple.
        ``predicate`` chooses which firings invoke the callbacks. Counts still
        advance for every firing and reset when the top-level module starts a
        forward. Hooks from this context are removed on the way out.
        """
        seen: list[ExecutionSite] = []
        counters = {key: 0 for key in self._firings}
        pending: dict[tuple[str, int], ExecutionSite | None] = {key: None for key in self._firings}

        def reset(_module, _inputs) -> None:
            for key in counters:
                counters[key] = 0
                pending[key] = None

        def finish(_module, _inputs, _output) -> None:
            for key, expected in self._firings.items():
                got = counters[key]
                kind, index = key
                if got != expected:
                    raise RuntimeError(
                        f"{self.family} {kind}[{index}] fired {got} times in one forward, expected {expected}"
                    )
                if pending[key] is not None:
                    raise RuntimeError(
                        f"{self.family} {kind}[{index}] activation had no enclosing block output"
                    )

        def watch(key: tuple[str, int]):
            def hook(_module, inputs) -> None:
                firing = counters[key]
                expected = self._firings[key]
                kind, index = key
                if firing >= expected:
                    raise RuntimeError(
                        f"{self.family} {kind}[{index}] fired too many times (unexpected firing {firing})"
                    )
                if pending[key] is not None:
                    raise RuntimeError(
                        f"{self.family} {kind}[{index}] fired again before its block output was observed"
                    )
                counters[key] = firing + 1
                site = self._by_firing[key][firing]
                pending[key] = site
                if predicate is None or predicate(site):
                    seen.append(site)
                    if callback is not None:
                        callback(site, _first_tensor(inputs, "activation input"))

            return hook

        def watch_block(key: tuple[str, int]):
            def hook(_module, _inputs, output) -> None:
                site = pending[key]
                kind, index = key
                if site is None:
                    raise RuntimeError(
                        f"{self.family} {kind}[{index}] returned before its activation module fired"
                    )
                pending[key] = None
                if on_block_output is not None and (predicate is None or predicate(site)):
                    on_block_output(site, _first_tensor(output, "block output"))

            return hook

        handles = [
            self.model.register_forward_pre_hook(reset),
            self.model.register_forward_hook(finish),
        ]
        for key, module in self._modules.items():
            handles.append(module.register_forward_pre_hook(watch(key)))
            handles.append(self._blocks[key].register_forward_hook(watch_block(key)))
        try:
            yield seen
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def route(self, predicate: Callable[[ExecutionSite], bool] | None = None) -> Iterator[list[ExecutionSite]]:
        """Record firings whose activation module runs while this context is open.

        The count for each physical block resets when the top-level module
        starts a forward, so a second call repeats recurrence 0 instead of
        continuing the previous call. ``predicate`` selects which of those
        firings are recorded; it does not change how calls are counted.
        """
        with self.observe(predicate=predicate) as seen:
            yield seen

    @contextmanager
    def intervene(self, edits: Mapping[ExecutionSite, BatchEdits]) -> Iterator[None]:
        """Apply ``edits`` until this context exits, then remove the hooks.

        Keys are execution sites. Each value is one batch of row edits for
        that firing. The same physical module can carry a different
        :class:`BatchEdits` on each recurrence; a firing with no entry keeps
        the tensor it was given. The batch length is checked against the
        activation when that firing runs. An empty map does not install hooks.
        """
        grouped: dict[tuple[str, int], dict[ExecutionSite, BatchEdits]] = {}
        for site, edit in edits.items():
            self._check(site)
            if not isinstance(edit, BatchEdits):
                raise TypeError("intervene values must be BatchEdits")
            grouped.setdefault((site.kind, site.physical_index), {})[site] = edit
        if not grouped:
            yield
            return

        counters = {key: 0 for key in grouped}
        current: dict[tuple[str, int], ExecutionSite | None] = {key: None for key in grouped}

        def reset(_module, _inputs) -> None:
            for key in counters:
                counters[key] = 0
                current[key] = None

        def watch(key: tuple[str, int]):
            def hook(_module, inputs):
                firing = counters[key]
                expected = self._firings[key]
                kind, index = key
                if firing >= expected:
                    raise RuntimeError(
                        f"{self.family} {kind}[{index}] fired too many times (unexpected firing {firing})"
                    )
                site = self._by_firing[key][firing]
                counters[key] = firing + 1
                current[key] = site
                edit = grouped[key].get(site)
                if edit is None:
                    return None
                activation = _first_tensor(inputs, "activation input")
                _check_edit_batch(edit, int(activation.shape[0]), int(activation.shape[-1]))
                updated = _with_activation_deltas(activation, edit)
                if updated is None:
                    return None
                return (updated, *tuple(inputs)[1:])

            return hook

        def watch_output(key: tuple[str, int]):
            def hook(_module, _inputs, output):
                site = current[key]
                if site is None:
                    return None
                edit = grouped[key].get(site)
                if edit is None or edit.residuals is None:
                    return None
                return _with_residual(output, edit.residuals)

            return hook

        handles = []
        try:
            handles.append(self.model.register_forward_pre_hook(reset))
            for key in grouped:
                module = self._modules[key]
                handles.append(module.register_forward_pre_hook(watch(key)))
                handles.append(module.register_forward_hook(watch_output(key)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _check(self, site: ExecutionSite) -> None:
        if site not in self._by_firing.get((site.kind, site.physical_index), ()):
            raise KeyError(site)


class HuginnAdapter(LoopedAdapter):
    """Huginn sites plus the runtime knobs that change how many times the core runs.

    A scalar ``num_steps`` puts every recurrence in Huginn's no-grad phase.
    ``(0, recurrences)`` runs each of the ``recurrences`` core passes with
    grad enabled, which is the fixed schedule a sweep has to name. Seed and
    ``test_time_noise`` are applied on every forward: the core draws its
    latent and its optional noise from the global RNG, and a non-zero
    ``test_time_noise`` makes the same prompt a different function each call.
    """

    def __init__(
        self,
        model: nn.Module,
        family: str,
        sites: tuple[ExecutionSite, ...],
        modules: dict[tuple[str, int], nn.Module],
        blocks: dict[tuple[str, int], nn.Module],
        firings: dict[tuple[str, int], int],
        *,
        recurrences: int,
        seed: int,
        test_time_noise: float,
    ) -> None:
        super().__init__(model, family, sites, modules, blocks, firings)
        self.recurrences = recurrences
        self.seed = seed
        self.test_time_noise = float(test_time_noise)
        self._apply_noise()

    @property
    def num_steps(self) -> tuple[int, int]:
        return (0, self.recurrences)

    def forward(self, *args, **kwargs):
        kwargs["num_steps"] = self.num_steps
        self._apply_noise()
        try:
            with _seeded_rng(self.model, self.seed):
                return self.model(*args, **kwargs)
        finally:
            # The module may write test_time_noise during the call. Pin it
            # again so the next read, including after the last token, is 0.
            self._apply_noise()

    def _apply_noise(self) -> None:
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "test_time_noise"):
            config.test_time_noise = self.test_time_noise


def adapt(
    model: nn.Module,
    *,
    family: str = "auto",
    recurrences: int | None = None,
    seed: int = 0,
    test_time_noise: float = 0.0,
) -> LoopedAdapter:
    """Build an adapter from a loaded module. Does not read a checkpoint."""
    if family == "auto":
        family = _detect_family(model)
    else:
        family = family.lower()
    if family == "ouro":
        _reject_huginn_controls(recurrences, seed, test_time_noise)
        return _from_layers(model, "ouro", int(model.model.total_ut_steps), "Ouro total_ut_steps")
    if family == "nanbeige":
        _reject_huginn_controls(recurrences, seed, test_time_noise)
        return _from_layers(model, "nanbeige", int(model.config.num_loops), "Nanbeige num_loops")
    if family == "huginn":
        if recurrences is None:
            raise ValueError("Huginn recurrences must be set")
        return _from_huginn(model, _positive("recurrences", recurrences), seed, test_time_noise)
    raise ValueError("family must be one of: auto, ouro, huginn, nanbeige")


def _reject_huginn_controls(recurrences: int | None, seed: int, test_time_noise: float) -> None:
    if recurrences is not None or seed != 0 or test_time_noise != 0:
        raise ValueError("recurrences, seed, and test_time_noise are Huginn controls")


def _detect_family(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "nanbeige":
        return "nanbeige"
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "total_ut_steps"):
        return "ouro"
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "core_block"):
        return "huginn"
    raise ValueError(f"cannot identify looped architecture {type(model).__name__}")


def _from_layers(model: nn.Module, family: str, n_loops: int, loop_name: str) -> LoopedAdapter:
    if n_loops < 1:
        raise ValueError(f"{loop_name} must be positive")
    # The modules the forward calls. Nanbeige's vLLM path rewrites
    # config.num_hidden_layers to physical depth times num_loops.
    layers = model.model.layers
    sites = enumerate_sites(family, n_layers=len(layers), n_loops=n_loops)
    modules = {}
    blocks = {}
    firings = {}
    for index, layer in enumerate(layers):
        key = ("layer", index)
        modules[key] = layer.mlp.down_proj
        blocks[key] = layer
        firings[key] = n_loops
    return LoopedAdapter(model, family, sites, modules, blocks, firings)


def _from_huginn(model: nn.Module, recurrences: int, seed: int, test_time_noise: float) -> HuginnAdapter:
    transformer = model.transformer
    groups = (
        ("prelude", list(transformer.prelude), 1),
        ("core", list(transformer.core_block), recurrences),
        ("coda", list(transformer.coda), 1),
    )
    sites = enumerate_sites(
        "huginn",
        n_prelude=len(groups[0][1]),
        n_core=len(groups[1][1]),
        n_coda=len(groups[2][1]),
        recurrences=recurrences,
    )
    modules: dict[tuple[str, int], nn.Module] = {}
    block_modules: dict[tuple[str, int], nn.Module] = {}
    firings: dict[tuple[str, int], int] = {}
    for kind, blocks, expected in groups:
        for index, block in enumerate(blocks):
            key = (kind, index)
            modules[key] = block.mlp.proj
            block_modules[key] = block
            firings[key] = expected
    return HuginnAdapter(
        model,
        "huginn",
        sites,
        modules,
        block_modules,
        firings,
        recurrences=recurrences,
        seed=seed,
        test_time_noise=test_time_noise,
    )


def _check_edit_batch(edit: BatchEdits, batch: int, hidden: int) -> None:
    neurons, deltas = edit.neurons, edit.deltas
    if neurons.ndim != 1 or deltas.ndim != 1 or int(neurons.shape[0]) != batch or int(deltas.shape[0]) != batch:
        raise ValueError(f"edit length {tuple(neurons.shape)} does not match activation batch {batch}")
    residual = edit.residuals
    if residual is None:
        return
    if residual.ndim != 2 or int(residual.shape[0]) != batch or int(residual.shape[1]) != hidden:
        raise ValueError(
            f"residuals shape {tuple(residual.shape)} does not match batch {batch} and hidden {hidden}"
        )


def _with_activation_deltas(activation: torch.Tensor, edit: BatchEdits) -> torch.Tensor | None:
    neurons = edit.neurons
    if not bool(torch.any(neurons >= 0)):
        return None
    updated = activation.clone()
    deltas = edit.deltas.to(device=updated.device, dtype=updated.dtype)
    for row in range(int(updated.shape[0])):
        neuron = int(neurons[row])
        if neuron < 0:
            continue
        updated[row, ..., neuron] = updated[row, ..., neuron] + deltas[row]
    return updated


def _with_residual(output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor | None:
    if not torch.is_tensor(output):
        raise TypeError(f"projection output has unsupported type {type(output)!r}")
    if not bool(torch.any(residual)):
        return None
    residual = residual.to(device=output.device, dtype=output.dtype)
    if output.ndim == 2:
        return output + residual
    ones = (1,) * (output.ndim - 2)
    return output + residual.view(residual.shape[0], *ones, residual.shape[1])


def _first_tensor(value, what: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"{what} has unsupported type {type(value)!r}")


@contextmanager
def _seeded_rng(model: nn.Module, seed: int) -> Iterator[None]:
    devices = sorted({
        parameter.device.index
        for parameter in model.parameters()
        if parameter.device.type == "cuda" and parameter.device.index is not None
    })
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield
