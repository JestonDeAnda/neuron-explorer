"""Linearized Raven forward.

RMSNorm denominators and attention patterns are frozen at their forward
values. Each SiLU(gate) * up neuron is a leaf. Shared blocks are keyed by an
invocation counter, because prelude, the recurrent core, and ``ln_f`` are
executed more than once. Direct mode stops gradient at the neuron leaf; total
mode uses the half-rule SiLU linearization so earlier neurons receive a total
effect.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F

from neuron_explorer.attribution.huginn.schedule import ExecutionSlot, execution_schedule


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


class HuginnLinearized:
    """Patched Raven execution that exposes one leaf per executed MLP neuron."""

    def __init__(self, model, mode: str = "direct"):
        self.model = model
        self.mode = mode
        self.cache: dict[tuple, torch.Tensor] = {}
        self.invocation_counts: dict[int, int] = {}
        self.mlp: dict[int, dict] = {}
        self.inject: dict[int, list[tuple[int, int, float]]] = {}
        self.replay = False
        self.freeze_mlp = False
        self.embed: torch.Tensor | None = None
        self.state: torch.Tensor | None = None
        self.logits: torch.Tensor | None = None
        self._current: dict[int, int] = {}
        self._saved: list[tuple] = []
        self._saved_init = None
        config = model.config
        self.slots = execution_schedule(
            config.n_layers_in_prelude,
            config.n_layers_in_recurrent_block,
            config.mean_recurrence,
            config.n_layers_in_coda,
        )
        self._mlp_slot: dict[int, ExecutionSlot] = {}
        self._mlp_block: dict[int, torch.nn.Module] = {}
        slots_for = {
            "prelude": [slot for slot in self.slots if slot.stage == "prelude"],
            "recurrent": [
                slot
                for slot in self.slots
                if slot.stage == "recurrent" and slot.recurrence_iteration == 0
            ],
            "coda": [slot for slot in self.slots if slot.stage == "coda"],
        }
        groups = (
            ("prelude", model.transformer.prelude),
            ("recurrent", model.transformer.core_block),
            ("coda", model.transformer.coda),
        )
        for stage, blocks in groups:
            for block, slot in zip(blocks, slots_for[stage], strict=True):
                if int(block.layer_id) != slot.physical_layer:
                    raise RuntimeError(
                        f"{stage} layer_id {block.layer_id} != {slot.physical_layer}"
                    )
                self._mlp_slot[id(block.mlp)] = slot
                self._mlp_block[id(block.mlp)] = block

    def __enter__(self):
        norm_type = type(self.model.transformer.ln_f)
        attention_type = type(self.model.transformer.prelude[0].attn)
        mlp_type = type(self.model.transformer.prelude[0].mlp)
        for module in self.model.modules():
            if isinstance(module, norm_type):
                self._patch(module, _norm_forward)
            elif isinstance(module, attention_type):
                self._patch(module, _attn_forward)
            elif isinstance(module, mlp_type):
                self._patch(module, _mlp_forward)
        self._saved_init = self.model.initialize_state
        self.model.initialize_state = self._initialize_state
        return self

    def __exit__(self, *exc):
        for module, had_forward, old in self._saved:
            if had_forward:
                module.forward = old
            else:
                del module.forward
            del module._huginn
        self._saved.clear()
        if self._saved_init is not None:
            self.model.initialize_state = self._saved_init
            self._saved_init = None

    def _patch(self, module, fn) -> None:
        self._saved.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
        module._huginn = self
        module.forward = types.MethodType(fn, module)

    def _begin(self, module) -> int:
        module_id = id(module)
        invocation = self.invocation_counts.get(module_id, 0)
        self.invocation_counts[module_id] = invocation + 1
        self._current[module_id] = invocation
        return invocation

    def frozen(self, module, name: str, value: torch.Tensor) -> torch.Tensor:
        key = (id(module), self._current[id(module)], name)
        if self.replay:
            return self.cache[key]
        stored = value.detach()
        self.cache[key] = stored
        return stored

    def slot_for(self, mlp, invocation: int) -> ExecutionSlot:
        template = self._mlp_slot[id(mlp)]
        if template.stage != "recurrent":
            if invocation != 0:
                raise RuntimeError(f"{template.stage} MLP ran {invocation + 1} times")
            return template
        config = self.model.config
        return ExecutionSlot(
            virtual_depth=(
                config.n_layers_in_prelude
                + invocation * config.n_layers_in_recurrent_block
                + template.physical_block
            ),
            stage="recurrent",
            recurrence_iteration=invocation,
            physical_layer=template.physical_layer,
            physical_block=template.physical_block,
        )

    def _initialize_state(self, input_embeds, scale: float = 1.0):
        """Draw one recurrent state and repeat it across attribution copies.

        Edge-batch rows are replicas of one prompt, matching the Qwen builder.
        Independent noise per copy would make those rows different graphs.
        """
        key = (0, 0, "state")
        if self.replay and key in self.cache:
            cached = self.cache[key]
            if cached.shape[0] != input_embeds.shape[0]:
                raise RuntimeError("replay batch does not match the cached recurrent state")
            return cached.to(device=input_embeds.device, dtype=input_embeds.dtype).clone()
        state = self._saved_init(input_embeds[:1], scale).detach()
        if input_embeds.shape[0] > 1:
            state = state.repeat(input_embeds.shape[0], 1, 1)
        state = state.requires_grad_(True)
        self.state = state
        if not self.replay:
            self.cache[key] = state.detach().clone()
        return state

    def forward(self, token_ids, copies: int = 1, replay: bool = False, freeze_mlp: bool = False):
        """Run ``copies`` identical rows. Replay reuses frozen norms, patterns, and state."""
        self.replay = replay
        self.freeze_mlp = freeze_mlp
        self.invocation_counts.clear()
        self._current.clear()
        self.mlp = {}
        if not replay:
            self.cache = {}
        device = self.model.transformer.wte.weight.device
        ids = torch.tensor([list(token_ids)] * copies, dtype=torch.long, device=device)
        self.embed = self.model.transformer.wte(ids).detach().requires_grad_(True)
        output = self.model(
            input_ids=ids,
            input_embeds=self.embed,
            num_steps=(0, int(self.model.config.mean_recurrence)),
            use_cache=False,
        )
        self.logits = output.logits
        if len(self.mlp) != len(self.slots):
            missing = sorted(set(range(len(self.slots))) - set(self.mlp))
            raise RuntimeError(f"MLP executions {len(self.mlp)} != {len(self.slots)} missing {missing}")
        return self.logits

    def neuron_target(self, specs, rule: str = "half"):
        """Half-rule (or gate/up) value of one neuron per batch row.

        ``specs[row] = (virtual_depth, position, neuron)``. At the forward
        point the half rule equals SiLU(gate) * up.
        """
        outputs = []
        for row, (depth, pos, neuron) in enumerate(specs):
            state = self.mlp[depth]
            weight = state["block"].mlp.fc.weight
            hidden = weight.shape[0] // 2
            features = state["x"][row, pos]
            gate_linear = features @ weight[neuron]
            up_linear = features @ weight[hidden + neuron]
            gate = state["g"][row, pos, neuron].float()
            up = state["u"][row, pos, neuron].float()
            sigmoid = torch.sigmoid(gate)
            if rule == "half":
                outputs.append(
                    0.5 * sigmoid * (gate_linear.float() * up + gate * up_linear.float())
                )
            elif rule == "up":
                outputs.append(sigmoid * gate * up_linear.float())
            elif rule == "gate":
                outputs.append(sigmoid * up * gate_linear.float())
            else:
                raise ValueError(f"unknown neuron rule {rule}")
        return torch.stack(outputs)

    def completion_logprob(self, row: int, completion_start: int, token_ids) -> torch.Tensor:
        """Sum of teacher-forced log-probabilities of every completion token."""
        if completion_start < 1 or completion_start >= len(token_ids):
            raise ValueError("completion must be non-empty and follow a non-empty prompt")
        total = self.logits.new_zeros((), dtype=torch.float32)
        for position in range(completion_start - 1, len(token_ids) - 1):
            log_probabilities = torch.log_softmax(self.logits[row, position].float(), dim=-1)
            total = total + log_probabilities[int(token_ids[position + 1])]
        return total

    def completion_logprob_target(self, specs):
        """``specs[row] = (completion_start, token_ids)``."""
        return torch.stack(
            [
                self.completion_logprob(row, start, token_ids)
                for row, (start, token_ids) in enumerate(specs)
            ]
        )

    def contributions(self, targets, keep: bool = True):
        """Activation times gradient for every executed neuron and token embedding."""
        leaves = [self.mlp[depth]["a"] for depth in range(len(self.slots))]
        sources = [*leaves, self.embed, self.state]
        gradients = torch.autograd.grad(
            targets.sum(), sources, allow_unused=True, retain_graph=keep
        )
        mlp = torch.stack(
            [
                (grad if grad is not None else torch.zeros_like(leaf)).float()
                * leaf.detach().float()
                for grad, leaf in zip(gradients[:-2], leaves)
            ]
        )
        embed_grad = gradients[-2]
        embed = (embed_grad.float() * self.embed.detach().float()).sum(-1)
        state_grad = gradients[-1]
        if state_grad is None:
            state = torch.zeros(self.state.shape[0], device=self.state.device)
        else:
            state = (state_grad.float() * self.state.detach().float()).flatten(1).sum(-1)
        return {"mlp": mlp, "embed": embed, "state": state}


def _norm_forward(self, x):
    linearized = self._huginn
    linearized._begin(self)
    features = x.float()
    inverse = torch.rsqrt(features.pow(2).mean(-1, keepdim=True) + self.eps)
    inverse = linearized.frozen(self, "inv", inverse)
    return (features * inverse).type_as(x) * self.weight


def _attn_forward(self, x, freqs_cis, block_idx, mask=None, past_key_values=None):
    del block_idx, mask, past_key_values
    linearized = self._huginn
    linearized._begin(self)
    batch, length, _channels = x.shape
    query, key, value = self.Wqkv(x).split(self.chunks, dim=2)
    query = query.view(batch, length, self.n_head, self.head_dim)
    key = key.view(batch, length, self.n_kv_heads, self.head_dim)
    value = value.view(batch, length, self.n_kv_heads, self.head_dim)
    if self.config.qk_bias:
        query_bias, key_bias = self.qk_bias.split(1, dim=0)
        query = (query + query_bias).to(query.dtype)
        key = (key + key_bias).to(key.dtype)
    rotary = sys.modules[type(self).__module__].apply_rotary_emb_complex_like
    query, key = rotary(query, key, freqs_cis=freqs_cis)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    scale = self.head_dim ** -0.5
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    causal = torch.ones(length, length, device=query.device, dtype=torch.bool).triu(1)
    scores = scores.masked_fill(causal, float("-inf"))
    pattern = torch.softmax(scores, dim=-1).to(dtype=query.dtype)
    pattern = linearized.frozen(self, "pattern", pattern)
    mixed = torch.matmul(pattern, value)
    mixed = mixed.transpose(1, 2).reshape(batch, length, -1).contiguous()
    return self.proj(mixed)


def _mlp_forward(self, x):
    linearized = self._huginn
    invocation = linearized._begin(self)
    slot = linearized.slot_for(self, invocation)
    gate, up = self.fc(x).chunk(2, dim=-1)
    cache_key = (id(self), invocation, "act")
    if linearized.replay and linearized.freeze_mlp:
        activation = linearized.cache[cache_key].clone()
    else:
        activation = (F.silu(gate) * up).detach()
        if not linearized.replay:
            linearized.cache[cache_key] = activation.clone()
    injected = linearized.inject.get(slot.virtual_depth)
    if injected:
        activation = activation.clone()
        for pos, neuron, delta in injected:
            activation[:, pos, neuron] = activation[:, pos, neuron] + delta
    if linearized.mode == "total":
        scale = torch.sigmoid(gate).detach()
        linear = _Shapley.apply(gate * scale, up)
        leaf = linear + (activation - linear).detach()
    else:
        leaf = activation.requires_grad_(True)
    linearized.mlp[slot.virtual_depth] = {
        "x": x,
        "g": gate.detach(),
        "u": up.detach(),
        "a": leaf,
        "slot": slot,
        "block": linearized._mlp_block[id(self)],
    }
    return self.proj(leaf)
