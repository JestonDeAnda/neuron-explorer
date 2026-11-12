"""Linearized execution of a natively loaded xLLM Huginn MoE model.

The plain xLLM modules are patched in place by runtime class, discovered from
the supplied model object, so importing this module never imports xLLM.
RMSNorm denominators, explicit causal attention patterns, and top-k routes are
frozen at their forward values; router IDs and normalized weights are metadata
rather than graph nodes. Each routed ``SiLU(gate) * up`` expert activation is a
leaf. Direct mode stops gradient at the leaf; total mode uses the half-rule
product with a forward-value correction. Every edge row replays one row-0
recurrent state, so edge-batch rows are replicas of one prompt.
"""

from __future__ import annotations

import operator
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from neuron_explorer.attribution.moe.types import (
    MoEContributions,
    MoEExecutionSlot,
    MoEModelIdentity,
    MoESourceLeaf,
    RoutedNeuron,
    RoutingSnapshot,
)
from neuron_explorer.attribution.moe.xllm_huginn.schedule import execution_schedule

_SCHEDULE_KEYS = (
    "num_layers",
    "huginn_prelude_layers",
    "huginn_recurrent_layers",
    "huginn_coda_layers",
    "loop_times",
    "huginn_hierarchical_state",
)
_RUNTIME_CLASSES = {
    "norm": "GroupRMSNorm",
    "attention": "CausalSoftmaxAttention",
    "router": "TopKRouter",
    "expert": "Expert",
}
_DETERMINISTIC_STATE_INITS = ("zero", "normal")
_OWNER = "_xllm_linearized"


class _Shapley(torch.autograd.Function):
    """Half-rule product: each factor receives half of the downstream gradient."""

    @staticmethod
    def forward(ctx, left, right):
        ctx.save_for_backward(left, right)
        return left * right

    @staticmethod
    def backward(ctx, grad):
        left, right = ctx.saved_tensors
        return 0.5 * grad * right, 0.5 * grad * left


@dataclass
class _ExpertRecord:
    slot: MoEExecutionSlot
    leaf: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    gate_linear: torch.Tensor
    up_linear: torch.Tensor
    order: torch.Tensor
    experts: torch.Tensor
    weights: torch.Tensor


class _RoutedAxis(Sequence):
    """Row-0 routed neurons ordered by depth, position, router rank, and neuron.

    Neurons are materialized on access; the real checkpoint has tens of
    millions of executed expert neurons per prompt.
    """

    def __init__(
        self,
        slots: Sequence[MoEExecutionSlot],
        experts: torch.Tensor,
        weights: torch.Tensor,
        width: int,
    ) -> None:
        self._slots = tuple(slots)
        self._experts = experts.tolist()
        self._weights = weights.tolist()
        self._length = int(experts.shape[1])
        self._top_k = int(experts.shape[2])
        self._width = int(width)

    def __len__(self) -> int:
        return len(self._slots) * self._length * self._top_k * self._width

    def coordinates(self, index: int) -> tuple[int, int, int, int]:
        rest, neuron = divmod(index, self._width)
        rest, rank = divmod(rest, self._top_k)
        depth, position = divmod(rest, self._length)
        return depth, position, rank, neuron

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("routed-neuron index out of range")
        depth, position, rank, neuron = self.coordinates(index)
        return RoutedNeuron(
            slot=self._slots[depth],
            position=position,
            expert_id=int(self._experts[depth][position][rank]),
            expert_neuron=neuron,
            expert_width=self._width,
            router_rank=rank,
            routing_weight=float(self._weights[depth][position][rank]),
        )

    def index(self, neuron, start=0, stop=None) -> int:
        del start, stop
        depth = neuron.slot.virtual_depth
        valid = (
            0 <= depth < len(self._slots)
            and self._slots[depth] == neuron.slot
            and 0 <= neuron.position < self._length
            and neuron.router_rank is not None
            and 0 <= neuron.router_rank < self._top_k
            and neuron.expert_width == self._width
            and 0 <= neuron.expert_neuron < self._width
        )
        if valid:
            index = (
                (depth * self._length + neuron.position) * self._top_k
                + neuron.router_rank
            ) * self._width + neuron.expert_neuron
            if self[index] == neuron:
                return index
        raise ValueError(f"{neuron!r} is not on the routed axis")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _RoutedAxis):
            return (
                self._slots == other._slots
                and self._experts == other._experts
                and self._weights == other._weights
                and self._width == other._width
            )
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return len(self) == len(other) and all(
                left == right for left, right in zip(self, other)
            )
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]


def _schedule_config(config: object) -> Mapping[str, object]:
    if isinstance(config, Mapping):
        return config
    return {"model": {key: getattr(config, key, None) for key in _SCHEDULE_KEYS}}


def _module_function(module: torch.nn.Module, name: str):
    function = getattr(sys.modules.get(type(module).__module__), name, None)
    if function is None:
        raise RuntimeError(
            f"{type(module).__module__} does not define native {name!r}"
        )
    return function


class XLLMHuginnLinearized:
    """Frozen-route linearization of the non-hierarchical xLLM ``Huginn``."""

    def __init__(
        self,
        model,
        config,
        *,
        model_name: str = "IFM/LoopedLM-P1-moe-huginn-500b",
    ) -> None:
        self.model = model
        self.schedule = execution_schedule(_schedule_config(config))
        if type(model).__name__ != "Huginn":
            raise RuntimeError(f"expected xLLM Huginn, got {type(model).__name__}")
        if getattr(model, "huginn_hierarchical_state", None) != "none":
            raise RuntimeError("only non-hierarchical Huginn execution is supported")
        state_init = getattr(model, "huginn_state_init", None)
        if state_init not in _DETERMINISTIC_STATE_INITS:
            raise RuntimeError(
                f"unsupported Huginn state init {state_init!r}; the recurrent "
                f"state leaf requires one of {_DETERMINISTIC_STATE_INITS}"
            )
        self._state_init = str(state_init)

        injection = getattr(model, "input_injection", None)
        if type(injection).__name__ != "InputInjection":
            raise RuntimeError(
                f"expected diagonal InputInjection, got {type(injection).__name__}"
            )

        layers = list(model.layers)
        physical_layers = 1 + max(slot.physical_layer for slot in self.schedule)
        if len(layers) != physical_layers:
            raise RuntimeError(
                f"model has {len(layers)} physical layers, schedule needs {physical_layers}"
            )
        first = layers[0]
        self._classes = {
            "norm": type(model.output.final_norm),
            "attention": type(first.attention.causal_attention),
            "router": type(first.moe.router),
            "expert": type(first.moe.experts),
        }
        for role, expected in _RUNTIME_CLASSES.items():
            actual = self._classes[role].__name__
            if actual != expected:
                raise RuntimeError(f"expected xLLM {expected} for {role}, got {actual}")

        moe = first.moe
        self._top_k = int(moe.topk)
        self._width = int(moe.inter_dim)
        self._expert_layer: dict[int, int] = {}
        self._expert_router: dict[int, torch.nn.Module] = {}
        for index, layer in enumerate(layers):
            attention = layer.attention
            if getattr(attention, "apply_attn_gate", False) or (
                getattr(attention, "wr", None) is not None
            ):
                raise RuntimeError(f"layer {index} enables an attention output gate")
            for name, owner in (("attention", attention), ("MoE", layer.moe)):
                residual = type(getattr(owner, "residual", None)).__name__
                if residual != "BaseResidual":
                    raise RuntimeError(
                        f"layer {index} {name} residual is {residual}, "
                        "expected BaseResidual"
                    )
            block = layer.moe
            if getattr(block, "fc1", None) is not None or getattr(
                block, "n_shared_experts", 0
            ):
                raise RuntimeError(f"layer {index} has shared experts, which are unsupported")
            if block.n_local_experts != block.n_experts:
                raise RuntimeError(f"layer {index} is expert-parallel, which is unsupported")
            if (int(block.topk), int(block.inter_dim), int(block.n_experts)) != (
                self._top_k,
                self._width,
                int(moe.n_experts),
            ):
                raise RuntimeError(f"layer {index} MoE shape differs from layer 0")
            for role, module in (
                ("attention", layer.attention.causal_attention),
                ("router", block.router),
                ("expert", block.experts),
            ):
                if type(module) is not self._classes[role]:
                    raise RuntimeError(f"layer {index} {role} class differs from layer 0")
            self._expert_layer[id(block.experts)] = index
            self._expert_router[id(block.experts)] = block.router
        self._slots = {
            (slot.physical_layer, slot.recurrence_iteration or 0): slot
            for slot in self.schedule
        }
        self._executions = {
            layer: sum(1 for slot in self.schedule if slot.physical_layer == layer)
            for layer in range(physical_layers)
        }

        router = moe.router
        scaling = getattr(router, "scaling_factor", None)
        normalization = "selected_renormalized" if self._top_k > 1 else "unnormalized"
        if scaling:
            normalization += "_scaled"
        self.identity = MoEModelIdentity(
            model_family="xllm-huginn",
            model_name=model_name,
            num_experts=int(moe.n_experts),
            expert_width=self._width,
            top_k=self._top_k,
            router_activation=str(router.score_func),
            router_normalization=normalization,
            has_shared_experts=False,
            device_placement="single_device",
            expert_parallel=False,
            expert_parallel_capable=False,
            direct_edge_semantics="local_linear_half_rule_direct",
            auxiliary=(
                ("implementation", "xllm-native-patched"),
                ("attention", "explicit_causal_fp32_frozen_pattern"),
                ("normalization", "frozen_rms_rstd"),
                ("routing", "frozen_topk_ids_and_weights"),
                ("recurrent_state", f"row0_{self._state_init}"),
                ("router_scaling_factor", str(scaling)),
            ),
        )

        self.mode = "direct"
        self.replay = False
        self.invocation_counts: dict[int, int] = {}
        self.logits: torch.Tensor | None = None
        self.embedding_leaf: torch.Tensor | None = None
        self.state_leaf: torch.Tensor | None = None
        self._current: dict[int, int] = {}
        self._cache: dict[object, object] = {}
        self._cache_tokens: tuple[int, ...] | None = None
        self._records: dict[int, _ExpertRecord] = {}
        self._pending_routes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._state_calls = 0
        self._token_ids: tuple[int, ...] | None = None
        self._copies = 0
        self._axis: _RoutedAxis | None = None
        self._routing_snapshot: RoutingSnapshot | None = None
        self._saved: list[tuple[torch.nn.Module, bool, object]] = []
        self._saved_init: tuple[bool, object] | None = None
        self._native_init = None
        self._saved_requires_grad: list[tuple[torch.nn.Parameter, bool]] = []
        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("the linearized backend context is already active")
        try:
            for module in self.model.modules():
                if isinstance(module, self._classes["norm"]):
                    self._patch(module, _norm_forward)
                elif isinstance(module, self._classes["attention"]):
                    self._patch(module, _attention_forward)
                elif isinstance(module, self._classes["router"]):
                    self._patch(module, _router_forward)
                elif isinstance(module, self._classes["expert"]):
                    self._patch(module, _expert_forward)
            self._patch(self.model.embed, _embedding_forward)
            self._native_init = self.model.initialize_recurrent_state
            self._saved_init = (
                "initialize_recurrent_state" in self.model.__dict__,
                self.model.__dict__.get("initialize_recurrent_state"),
            )
            self.model.initialize_recurrent_state = self._initialize_state
            for parameter in self.model.parameters():
                self._saved_requires_grad.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
        except BaseException:
            self._restore()
            raise
        self._entered = True
        return self

    def __exit__(self, *exc) -> None:
        self._restore()

    def _restore(self) -> None:
        for module, had_forward, old in reversed(self._saved):
            if had_forward:
                module.forward = old
            else:
                module.__dict__.pop("forward", None)
            module.__dict__.pop(_OWNER, None)
        self._saved.clear()
        if self._saved_init is not None:
            had_init, old_init = self._saved_init
            if had_init:
                self.model.initialize_recurrent_state = old_init
            else:
                self.model.__dict__.pop("initialize_recurrent_state", None)
            self._saved_init = None
        self._native_init = None
        for parameter, requires_grad in self._saved_requires_grad:
            parameter.requires_grad_(requires_grad)
        self._saved_requires_grad.clear()
        self._entered = False

    def _patch(self, module: torch.nn.Module, function) -> None:
        self._saved.append(
            (module, "forward" in module.__dict__, module.__dict__.get("forward"))
        )
        setattr(module, _OWNER, self)
        module.forward = types.MethodType(function, module)

    def _reset(self, *, replay: bool) -> None:
        self.replay = replay
        self.invocation_counts.clear()
        self._current.clear()
        self._records = {}
        self._pending_routes = {}
        self._state_calls = 0
        if not replay:
            self._cache = {}
            self._cache_tokens = None

    def _begin(self, module: torch.nn.Module) -> int:
        key = id(module)
        invocation = self.invocation_counts.get(key, 0)
        self.invocation_counts[key] = invocation + 1
        self._current[key] = invocation
        return invocation

    def frozen(self, module: torch.nn.Module, name: str, value: torch.Tensor) -> torch.Tensor:
        key = (id(module), self._current[id(module)], name)
        if self.replay:
            if key not in self._cache:
                raise RuntimeError(f"replay cache has no frozen {name} for this execution")
            return self._cache[key]  # type: ignore[return-value]
        stored = value.detach()
        self._cache[key] = stored[:1].clone()
        return stored

    def _slot_for(self, experts: torch.nn.Module, invocation: int) -> MoEExecutionSlot:
        layer = self._expert_layer[id(experts)]
        slot = self._slots.get((layer, invocation))
        if slot is None:
            raise RuntimeError(
                f"physical layer {layer} expert ran {invocation + 1} times; "
                f"schedule allows {self._executions[layer]}"
            )
        return slot

    def _record(self, record: _ExpertRecord) -> None:
        depth = record.slot.virtual_depth
        if depth in self._records:
            raise RuntimeError(f"duplicate MoE execution ID {depth}")
        self._records[depth] = record

    def _initialize_state(self, input_embeds: torch.Tensor, cache_len: int = 0):
        """Draw one row-0 recurrent state and repeat it across edge rows."""
        self._state_calls += 1
        if self._state_calls > 1:
            raise RuntimeError("recurrent state was initialized more than once")
        if self.replay:
            if "state" not in self._cache:
                raise RuntimeError("replay cache has no recurrent state")
            base = self._cache["state"]
        else:
            base = self._native_init(input_embeds[:1], cache_len).detach().clone()
            self._cache["state"] = base
        state = (
            base.to(device=input_embeds.device, dtype=input_embeds.dtype)
            .expand(input_embeds.shape[0], -1, -1)
            .clone()
            .requires_grad_(True)
        )
        self.state_leaf = state
        return state

    def forward(
        self,
        token_ids: Sequence[int],
        *,
        copies: int = 1,
        replay: bool = False,
        mode: str = "direct",
    ) -> torch.Tensor:
        if not self._entered:
            raise RuntimeError("enter the linearized backend context before forward")
        if mode not in {"direct", "total"}:
            raise ValueError(f"unknown linearization mode {mode!r}")
        if copies < 1:
            raise ValueError("copies must be positive")
        if self.model.training:
            raise RuntimeError("the xLLM model must be in eval mode")
        ids = tuple(int(token) for token in token_ids)
        if not ids:
            raise ValueError("token_ids must not be empty")
        if replay and (self._cache_tokens != ids or "state" not in self._cache):
            raise RuntimeError("replay requires a completed forward of the same tokens")

        cached_tokens = self._cache_tokens
        self.logits = None
        self._axis = None
        self.mode = mode
        self._reset(replay=replay)
        device = next(self.model.parameters()).device
        tokens = torch.tensor([list(ids)] * copies, dtype=torch.long, device=device)
        with torch.enable_grad():
            output = self.model(tokens, multi_segments=False)
        logits = output[0] if isinstance(output, tuple) else output

        expected = len(self.schedule)
        if len(self._records) != expected:
            missing = sorted(set(range(expected)) - set(self._records))
            raise RuntimeError(
                f"MoE executions {len(self._records)} != {expected}; missing {missing}"
            )
        if self._state_calls != 1:
            raise RuntimeError("recurrent state was not initialized exactly once")
        if self._pending_routes:
            raise RuntimeError("a router ran without its routed experts")

        records = [self._records[depth] for depth in range(expected)]
        experts = torch.stack([record.experts[0] for record in records]).cpu()
        weights = torch.stack([record.weights[0] for record in records]).float().cpu()
        self._axis = _RoutedAxis(self.schedule, experts, weights, self._width)
        self._routing_snapshot = RoutingSnapshot(
            selected_experts=experts,
            routing_weights=weights,
            selected_expert_counts=torch.full(
                experts.shape[:2], self._top_k, dtype=torch.long
            ),
            router_activation=self.identity.router_activation,
            router_normalization=self.identity.router_normalization,
        )
        self._cache_tokens = cached_tokens if replay else ids
        self._token_ids = ids
        self._copies = copies
        self.logits = logits
        return logits

    @property
    def routing_snapshot(self) -> RoutingSnapshot:
        if self._routing_snapshot is None:
            raise RuntimeError("forward must be called before reading routes")
        return self._routing_snapshot

    def routed_neurons(self) -> Sequence[RoutedNeuron]:
        return self._require_forward()

    def _require_forward(
        self, token_ids: Sequence[int] | None = None, mode: str | None = None
    ) -> _RoutedAxis:
        if self._axis is None or self.logits is None:
            raise RuntimeError("forward must complete before reading attribution values")
        if token_ids is not None and tuple(int(t) for t in token_ids) != self._token_ids:
            raise RuntimeError("the last forward used different token IDs")
        if mode is not None and self.mode != mode:
            raise RuntimeError(f"this operation requires a {mode}-mode forward")
        return self._axis

    def _ordered_records(self) -> list[_ExpertRecord]:
        return [self._records[depth] for depth in range(len(self.schedule))]

    @staticmethod
    def _routed_rows(
        records: Sequence[_ExpertRecord],
        values: Sequence[torch.Tensor],
        rows: int,
    ) -> torch.Tensor:
        return torch.cat(
            [
                value[record.order[:rows]].reshape(rows, -1)
                for record, value in zip(records, values, strict=True)
            ],
            dim=1,
        )

    def candidate_effects(
        self, completion_start: int, token_ids: Sequence[int]
    ) -> torch.Tensor:
        self._require_forward(token_ids, mode="total")
        target = self.completion_logprob(0, completion_start, token_ids)
        records = self._ordered_records()
        leaves = [record.leaf for record in records]
        gradients = torch.autograd.grad(
            target, leaves, retain_graph=True, allow_unused=True
        )
        effects = [
            (torch.zeros_like(leaf) if grad is None else grad).float()
            * leaf.detach().float()
            for grad, leaf in zip(gradients, leaves, strict=True)
        ]
        return self._routed_rows(records, effects, 1)[0].detach()

    def _neuron_target(self, row: int, index: int) -> torch.Tensor:
        assert self._axis is not None
        depth, position, rank, neuron = self._axis.coordinates(index)
        record = self._records[depth]
        grouped_row = record.order[row, position, rank]
        gate = record.gate[grouped_row, neuron].float()
        up = record.up[grouped_row, neuron].float()
        gate_linear = record.gate_linear[grouped_row, neuron].float()
        up_linear = record.up_linear[grouped_row, neuron].float()
        return 0.5 * torch.sigmoid(gate) * (gate_linear * up + gate * up_linear)

    def direct_targets(
        self,
        targets: Sequence[int],
        completion_start: int,
        token_ids: Sequence[int],
    ) -> MoEContributions:
        """Attribute targets whose nonnegative indices address ``routed_neurons``."""
        axis = self._require_forward(token_ids, mode="direct")
        requested = [int(target) for target in targets]
        if not requested:
            raise ValueError("at least one target is required")
        if len(requested) > self._copies:
            raise ValueError(
                f"{len(requested)} targets exceed the {self._copies} forward copies"
            )
        scalars = []
        for row, target in enumerate(requested):
            if target == -1:
                scalars.append(self.completion_logprob(row, completion_start, token_ids))
            elif 0 <= target < len(axis):
                scalars.append(self._neuron_target(row, target))
            else:
                raise ValueError(f"target {target} is outside the routed axis")
        assert self.embedding_leaf is not None and self.state_leaf is not None
        records = self._ordered_records()
        leaves = [record.leaf for record in records]
        sources = [*leaves, self.embedding_leaf, self.state_leaf]
        gradients = torch.autograd.grad(
            torch.stack(scalars).sum(), sources, retain_graph=True, allow_unused=True
        )
        gradients = [
            torch.zeros_like(source) if grad is None else grad
            for grad, source in zip(gradients, sources, strict=True)
        ]
        rows = len(requested)
        routed = self._routed_rows(
            records,
            [
                grad.float() * leaf.detach().float()
                for grad, leaf in zip(gradients[:-2], leaves, strict=True)
            ],
            rows,
        )
        embedding = (
            gradients[-2].float() * self.embedding_leaf.detach().float()
        ).sum(-1)[:rows]
        state = (
            (gradients[-1].float() * self.state_leaf.detach().float())
            .flatten(1)
            .sum(-1)[:rows]
        )
        return MoEContributions(
            routed=routed.detach(),
            token_embeddings=embedding.detach(),
            source_contributions=state.detach().unsqueeze(-1),
            targets=torch.stack([scalar.detach().float() for scalar in scalars]),
            routed_neurons=axis,  # type: ignore[arg-type]
            source_leaves=(
                MoESourceLeaf(
                    "recurrent_state",
                    kind="huginn_state",
                    metadata=(("state_init", self._state_init),),
                ),
            ),
        )

    def activations(self, neurons: Sequence[RoutedNeuron]) -> torch.Tensor:
        axis = self._require_forward()
        values = []
        for neuron in neurons:
            depth, position, rank, expert_neuron = axis.coordinates(axis.index(neuron))
            record = self._records[depth]
            values.append(
                record.leaf[record.order[0, position, rank], expert_neuron].detach()
            )
        return torch.stack(values)

    def completion_logprob(
        self, row: int, completion_start: int, token_ids: Sequence[int]
    ) -> torch.Tensor:
        """Sum of teacher-forced log-probabilities of every completion token."""
        self._require_forward(token_ids)
        assert self.logits is not None
        if completion_start < 1 or completion_start >= len(token_ids):
            raise ValueError("completion must be non-empty and follow a non-empty prompt")
        total = self.logits.new_zeros((), dtype=torch.float32)
        for position in range(completion_start - 1, len(token_ids) - 1):
            log_probabilities = torch.log_softmax(self.logits[row, position].float(), dim=-1)
            total = total + log_probabilities[int(token_ids[position + 1])]
        return total


def _embedding_forward(self, tokens):
    backend = getattr(self, _OWNER)
    native = type(self).forward(self, tokens)
    leaf = native.detach().requires_grad_(True)
    backend.embedding_leaf = leaf
    return leaf


def _norm_forward(self, x):
    backend = getattr(self, _OWNER)
    backend._begin(self)
    if getattr(self, "gather_input", False):
        raise RuntimeError("model-parallel RMSNorm is unsupported during attribution")
    groups = int(getattr(self, "groups_per_partition", 1))
    features = x.float().reshape(*x.shape[:-1], groups, -1)
    rstd = torch.rsqrt(features.pow(2).mean(-1, keepdim=True) + self.eps)
    rstd = backend.frozen(self, "rstd", rstd)
    output = (features * rstd).reshape(x.shape).to(x.dtype)
    return output if self.weight is None else output * (self.weight + 1)


def _attention_forward(
    self,
    xq,
    xk,
    xv,
    g,
    freqs_cis,
    segments,
    fp32_attn_output=False,
    deterministic=True,
    cache=None,
):
    del g, fp32_attn_output, deterministic
    backend = getattr(self, _OWNER)
    backend._begin(self)
    if segments is not None or cache is not None:
        raise RuntimeError("explicit attention requires one segment and no KV cache")
    batch, length, _ = xq.shape
    query = xq.reshape(batch, length, self.n_heads, self.head_dim)
    key = xk.reshape(batch, length, self.n_kv_heads, self.head_dim)
    value = xv.reshape(batch, length, self.n_kv_heads, self.head_dim)
    rope = int(self.rope_head_dim)
    if rope > 0:
        rotary = _module_function(self, "apply_rotary_embeddings")
        if rope == self.head_dim:
            query, key = rotary(query, key, freqs_cis=freqs_cis, backward=False)
        else:
            sizes = [rope, self.head_dim - rope]
            query_rope, query_pass = torch.split(query, sizes, dim=-1)
            key_rope, key_pass = torch.split(key, sizes, dim=-1)
            query_rope, key_rope = rotary(
                query_rope, key_rope, freqs_cis=freqs_cis, backward=False
            )
            query = torch.cat([query_rope, query_pass], dim=-1)
            key = torch.cat([key_rope, key_pass], dim=-1)
    repeats = self.n_heads // self.n_kv_heads
    key = key.repeat_interleave(repeats, dim=2).transpose(1, 2)
    value = value.repeat_interleave(repeats, dim=2).transpose(1, 2)
    query = query.transpose(1, 2)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * self.scale
    future = torch.ones(length, length, device=query.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(future, float("-inf"))
    pattern = torch.softmax(scores, dim=-1).to(query.dtype)
    pattern = backend.frozen(self, "pattern", pattern)
    mixed = torch.matmul(pattern, value).transpose(1, 2)
    return mixed.reshape(batch, length, -1), None


def _router_forward(self, x, load_balancing_type=None):
    backend = getattr(self, _OWNER)
    invocation = backend._begin(self)
    if load_balancing_type is not None:
        raise RuntimeError("router load balancing is unsupported during attribution")
    scores, indices, _tokens_per_expert, _aux = type(self).forward(self, x.detach(), None)
    indices = indices.detach()
    scores = scores.detach()
    key = (id(self), invocation, "route")
    if backend.replay:
        if key not in backend._cache:
            raise RuntimeError("replay cache has no route for this execution")
        cached_indices, cached_scores = backend._cache[key]
        if cached_indices.shape[1:] != indices.shape[1:] or not torch.equal(
            indices, cached_indices.expand_as(indices)
        ):
            raise RuntimeError("replay route differs from the cached frozen route")
        indices = cached_indices.expand_as(indices).contiguous()
        scores = cached_scores.expand_as(scores).contiguous()
    else:
        if not torch.equal(indices, indices[:1].expand_as(indices)):
            raise RuntimeError("edge rows selected different routes")
        scores = scores[:1].expand_as(scores).contiguous()
        backend._cache[key] = (indices[:1].clone(), scores[:1].clone())
    if id(self) in backend._pending_routes:
        raise RuntimeError("router ran twice before its routed experts")
    backend._pending_routes[id(self)] = (indices, scores)
    tokens_per_expert = torch.bincount(indices.flatten(), minlength=self.num_experts)
    return scores, indices, tokens_per_expert, None


def _expert_forward(self, x, group_sizes):
    backend = getattr(self, _OWNER)
    invocation = backend._begin(self)
    slot = backend._slot_for(self, invocation)
    router = backend._expert_router[id(self)]
    route = backend._pending_routes.pop(id(router), None)
    if route is None:
        raise RuntimeError("routed experts ran without a recorded route")
    indices, scores = route
    batch, length, top_k = indices.shape
    rows = batch * length * top_k
    sizes = [int(size) for size in group_sizes]
    if len(sizes) != self.n_local_experts or sum(sizes) != rows or x.shape[0] != rows:
        raise RuntimeError("grouped expert rows do not match the recorded route")
    flat = indices.flatten()
    sorted_indices = torch.argsort(flat, stable=True)
    grouped_experts = torch.repeat_interleave(
        torch.arange(len(sizes), device=flat.device),
        torch.tensor(sizes, device=flat.device),
    )
    if not torch.equal(flat[sorted_indices], grouped_experts):
        raise RuntimeError("grouped expert offsets do not match the recorded route")
    order = torch.empty_like(sorted_indices)
    order[sorted_indices] = torch.arange(rows, device=flat.device)

    mgmm = _module_function(self, "mgmm")
    w1 = self.weight1.reshape(self.n_local_experts, self.inter_dim, self.model_dim)
    w2 = self.weight2.reshape(self.n_local_experts, self.model_dim, self.inter_dim)
    w3 = self.weight3.reshape(self.n_local_experts, self.inter_dim, self.model_dim)
    gate_linear = mgmm(x, w1, group_sizes, True, self.backend)
    up_linear = mgmm(x, w3, group_sizes, True, self.backend)
    gate = gate_linear.detach()
    up = up_linear.detach()
    activation = F.silu(gate) * up
    if backend.mode == "total":
        linear = _Shapley.apply(gate_linear * torch.sigmoid(gate), up_linear)
        leaf = linear + (activation - linear).detach()
    else:
        leaf = activation.requires_grad_(True)
    backend._record(
        _ExpertRecord(
            slot=slot,
            leaf=leaf,
            gate=gate,
            up=up,
            gate_linear=gate_linear,
            up_linear=up_linear,
            order=order.view(batch, length, top_k),
            experts=indices,
            weights=scores,
        )
    )
    return mgmm(leaf, w2, group_sizes, True, self.backend)
