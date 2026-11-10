# Neuron explorer — source notes

Three inputs, collected 2026-10-18.

## 1. "Neural Networks Are Already Clean" (Cammarata, Goodfire)

Mirror: `source/` (site + assets), plain text: `source/essay.md`.
Gallery data: `source/data/neuron-gallery.json` (1000 Qwen3-32B neuron papers, summary form),
`source/data/neuron-gallery-more.json` (per-neuron long text), `source/neuron-scientist/papers/shelf.json`.

Gallery record schema (inferred): `id` "L0_N15647", `k` kind, `L` layer, `N` neuron, `t` title,
`l` lede, `st` status ("tested"), `i`/`c`/`f` counts, `v` version, `d` date,
`m` = [[probe-category, activation], ...] — positive controls down to no-cue controls.
`more.json`: `l` full explanation, `wi` "why interesting", `wc` "why confident".

Methodology the explorer should make cheap:
- **Unit = neuron pole.** Positive and negative poles are separate objects (ID `L22 N2742+`). Negative pole carries
  gating meaning (V vs O; Santa lives on a negative pole).
- **SwiGLU sign regimes.** Split activations by sign(gate) × sign(up). The −×− "double negative echo" fills the noisy
  middle of the positive range; for ~4% of neurons it's the whole top of the list.
- **Raw activations, not percentiles.** Show the activation-vs-frequency shape; frame rarity in pages
  (p99 ≈ 4×/page, p99.9 ≈ every 2–3 pages, p99.99 ≈ every 25–30).
- **Long context.** Long-context neurons look polysemantic on 2-paragraph windows.
- **Look forward too.** Many neurons fire on the token *before* the thing ("boat" next); logit effects matter.
- **Summary motif.** Context neurons fire on commas/punctuation where attention summarizes — show surrounding text,
  not just the max token.
- **Steering.** Turn it up, read what the model writes (Colorado neuron → plans a hike).
- **Upstream / downstream.** What feeds it, what reads it; cluster input attribution into bundles.
- **Blind maximum.** Agent A writes explanation; agent B (sees only explanation) writes a max-activating example;
  compare to corpus max. Plus positive controls, near misses, and "most distant concept" texts.
- **Last layer = "say or just think"** single-token suppression; **"don't say a dot"** motif.
- **Recursive generalization**: offshoots are local branches (iy → diy → do it yourself). Explorer should make
  following a weak activation back through its circuit easy.

## 2. Sauers/Qwen3-8B-tawny-owl-128-neurons

Local: `tawny-owl/` (README, neurons.json, neurons.safetensors deltas, training code, persona_128.json).

128 MLP neurons in Qwen3-8B (36 layers × 12,288) found by causal search: copy neuron values from a character
system-prompt run into an AI-assistant run, find the fewest that flip "As an AI…" → "As a nurse…". Training only
those neurons' gate row / up row / down column makes the model "Hoot", a tawny owl. Zeroing the 128 in the base model
drops "I am an AI" replies 80% → 30% (random-128 control: 77–82%).

Layer spread: heavy at L0 (21) and L24–35 (68), sparse in the middle. Largest deltas: L11 N2050, L0 N847, L13 N9272,
L0 N1919, L23 N5088.

Open tension with the essay: in a pilot, 128 *random* neurons from the same layers trained into an equally good owl.
Causal-search neurons matter for ablation (80→30%), but maybe not for finetuning capacity.

## 3. Screenshot — reference neuron panel (Qwen3.5)

`L22 N2742+ · layer 22 · solved` — title "alien causal agents". Tabs: **Examples / Chat / Details**.

Chat tab controls:
- `▲▼` side-by-side: steer up vs down, both columns generated from the same prompt.
- `unclamped` (with status dot) — alternative is clamping the activation to a fixed value.
- `writes` / `reads you` — steer the model with the neuron vs. show the neuron's activation over your own typed text.
- multiplier `− ×2 +`, `clear`, `GPU` badge per column.

Example: "Who made you?" ▲×2 → "I am Extrali, an intelligent extraterrestrial life form from the planet Xenobi…"
(degenerates into repetition); ▼×2 → "I am Qwen3.5 … Tongyi Lab". A single mid-layer neuron at ×2 flips self-identity —
the same phenomenon the owl model trains for.

## Explorer copy

Neuron cards present information and make no claims: rollouts beside baselines, distinctive words with counts,
logit-lens tokens for ±W_down, and measured numbers (departure from baseline per dose, repetition, length-capped
rollouts, cross-prompt word overlap). Interpretation stays with the person reading. Sorting uses the measured numbers.

## Later

- Rollout map tab: embed every rollout, UMAP, colored/filterable by neuron, dose, prompt.
