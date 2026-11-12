# Opus Magnum Blackbox

Autoregressive RL translation layer and training code for Opus Magnum. 
It converts a level `LevelSpec` into integer tokens, constrains generation with a grammar-aware action mask, decodes token sequences back onto a `Level`, runs the simulation, and scores the result through a Gymnasium environment.

Current state: codec, grammar, masker, reward, environment, and the framework-free training stack (policy model, rollout, REINFORCE trainer, CLI) are implemented and tested. There is one tier-0 training level. `just train` starts the default overfit run; checkpoints and `metrics.jsonl` land under `runs/` (git-ignored). Gate a change with `just lint` + `just test` (the `ty` stage is advisory — see below).

## 1. Scope and related projects

- **Simulator dependency**: the engine lives in the sibling repo `../opus_magnum_sim-live` (editable dependency in `pyproject.toml`, package `opus_magnum_sym`). 
- **DSL docs**: `.oml`/`.oms` language details and authoring style live in `.agents/skills/oms-authoring/SKILL.md`. Load them only when editing level or solution files.
- **This repo owns**: the token vocab and grammar, codec, masking geometry, Gymnasium env, reward, and training corpus. Do not patch the simulator here unless the task includes it.

## 2. Quick start and commands

`uv sync` installs the environment. Python is `>=3.14.10`; the simulator is installed editable from `../opus_magnum_sim-live`.

```bash
uv sync
just test                       # full pytest suite
just test-file tests/test_env.py
just lint                       # ruff check --fix (mutates files)
just format                     # ruff format
just check                      # ty check src
just full                       # check + lint + test
just train                      # default overfit run on tier-0, writes runs/<timestamp>-<level>/
```

The `just om-*` recipes (`om-repl-headless`, `om-run`, `om-validate`) call the simulator's `opus-magnum` CLI, not this package.

**`ty` usage**: advisory only, allowed to fail (accepted baseline diagnostics; `just full` stops at `just check`) — gate changes with `just lint` + `just test`. For a new diagnostic, judge real vs false alarm (e.g. `X | None` narrowing, torch stub quirks): fix real ones, ignore false ones. Never silence with `# ty: ignore` or restructure code for the checker, and never re-add suppressions the user removed.

`tests/test_torch_cuda.py` skips if torch is absent but asserts CUDA is available when torch is installed. A failing CUDA test is an environment problem, not necessarily a code regression.

## 3. Layout

```
src/opus_magnum_blackbox/
├── config.py    # VocabConfig, TokenizerConfig, RewardConfig, Curriculum
├── vocab.py     # flat token id table + enum bridges
├── grammar.py   # incremental parser, Solution IR, SpecModel
├── codec.py     # spec encoder, pure decode, apply-to-Level
├── masking.py   # stateless next-token mask
├── geometry.py  # placement footprint/bounds/overlap for masking
├── reward.py    # pure terminal reward function
├── env.py       # OpusEnv (Gymnasium)
├── model.py     # OpusPolicy: transformer policy/value network
├── rollout.py   # episode collection, teacher-forced batches
├── train.py     # Trainer: masked REINFORCE loop, checkpoints, metrics
├── stages.py    # pipeline runner: TOML stage configs (bc + rl), train/eval banks
├── cli.py       # `train` / `eval` subcommands
├── corpus.py    # recursive levels/ scanner: entries, manifests, group resolution
├── families.py  # tier-1 family: spec builder + simulator-verified reference solver
└── gen_tier1.py # seeded tier-1 bank generator (.oml/.oms files + manifest)
levels/          # top-level tier-0 files + tier1_simple_arms/{train,eval} bank (generated)
tests/           # behavior-focused pytest suite
```

Pipeline: `LevelSpec` → `OpusTokenizer.encode_level_spec` → `OpusEnv`; model emits token ids → `ActionMasker.mask` → `decode_solution` → `apply_solution` → `Level.run` → `compute_reward`.

Core objects:

- `LevelSpec` (from `opus_magnum_sym`): immutable level definition — board size, shapes, entrypoints, allowed blueprints, step limit, optional `target`/`cycling`.
- `Vocab`: maps token names to contiguous integer ids and back; owns category ranges and simulator enum bridges.
- `OpusTokenizer`: `encode_level_spec(spec)` produces padded encoder ids; `decode_solution_to_level(tokens, spec, level)` applies a sequence to a `Level`.
- `ActionMasker`: returns the bool mask of valid next tokens for a history; stateless.
- `OpusEnv`: Gymnasium wrapper around one or more `LevelEntry` objects; owns the tokenizer, masker, and level cache.
- `Level`: simulator runtime from `opus_magnum_sym`; this repo only encodes specs, mutates levels through `place`/`place_entrypoint`/`assign_step_action`, and reads `LevelResult`.

## 4. Wire format invariants

- Token ids are a flat, contiguous layout in this fixed category order: `special, mode, ctype, atom, builtin, entry_slot, rotation, q, r, delta, cell, handle, tick, action`.
- **Never renumber or reorder categories or named tokens.** `tests/` bake concrete ids; changing the vocabulary breaks every trained checkpoint and is a wire-format break.
- Numeric categories use `base(category) + value` and `id - base(category)`. Deltas are shifted by `max_cell_offset`; use `Vocab.value_id`.
- `BUILTINS` is a snapshot of implemented `default_registry` entries in `BPN` order, resolved at import. A registry change needs a fresh process for the vocab to see it.
- `ACT_NOP` is decode-only: it consumes a program triplet but assigns no `StepActionType`.
- Atom tokens use `ATOM_NONE` for untyped encoder cells plus real `AtomType`s; `BASIC` is not a game type.
- Default caps: `max_encoder_len=256`, `max_placements=16`, `max_program_actions=32`, `max_decoder_len=164`, board `10x10`, `max_steps=64`, `max_handles=8`.

## 5. Grammar, decoding, masking

`grammar.py` is the single source of truth for sequence structure. Both the masker and decoder must go through `advance`/`parse`; do not add a second parser.

Sequence shape:

```text
[START_LAYOUT] (name, rotation, q, r)* [END_LAYOUT]
[START_PROGRAM] (handle, tick, action)* [SUBMIT]
```

Rules enforced by `advance`:

- `[END_LAYOUT]` only after every required entrypoint slot has been placed (`SpecModel.required`; zero when `entrypoints_fixed`).
- Handles are non-decreasing: after using handle N, lower handles are invalid.
- A tick may be used at most once per handle. NOP does not consume the tick.
- A trailing incomplete layout quadruplet or program triplet is ignored; complete groups are still applied. `[SUBMIT]` is required to mark `submitted=True`.
- `GrammarError` is a dataclass, not an exception. It carries the token position and message.

`codec.py`:

- `encode_level_spec` emits header, built-ins, per-shape segments, then per-entrypoint segments, padded to `max_encoder_len`.
- `decode_solution` is pure: tokens → `Solution` (placements/actions), no simulator objects.
- `apply_solution` mutates a reset `Level` via `Level.place_entrypoint` / `Level.place` / `Level.assign_step_action`; it converts simulator exceptions into `DecodeReport(ok=False, error_token_pos=...)`.
- `decode_solution_to_level` is just parse + apply. Keep this seam if editing.
- `solution_to_oms` renders a decoded `Solution` back to `.oms` text; the handle -> component-id mapping comes from `DecodeReport.handles`, filled by `apply_solution`.
- Coordinate mapping is owned by `codec.py`: token `Q` → `Hex.s` (column), token `R` → `Hex.r` (row).

`masking.py`:

- `ActionMasker` is stateless: rebuilds `ParseState` from history on every call.
- A broken history or `Phase.DONE` yields a PAD-only mask.
- Layout masking narrows rotations, columns, then rows using `geometry.py`; program masking enforces handle order, unused ticks, base actions, and piston-only `EXTEND`/`RETRACT`.
- `geometry.py` duplicates the simulator's non-mutating placement checks (footprint, bounds, overlap, fixed entrypoints). Keep it behavior-identical to the simulator; do not add simulator side effects here.

`SpecModel` is cached via `spec_model` and `_spec_model`, keyed on scalar spec fields because `LevelSpec` hashes are randomized per process. Preserve that workaround if changing the cache.

## 6. Environment and reward

`OpusEnv` observations (all NumPy `int64` except the mask):

| key | meaning |
|---|---|
| `encoder_input` | padded spec ids, shape `(max_encoder_len,)` |
| `encoder_length` | true spec token count |
| `decoder_history` | padded solution ids, shape `(max_decoder_len,)` |
| `decoder_length` | tokens emitted so far |
| `action_mask` | bool action mask over the full vocab, cast to `int8` |

- Action space: `Discrete(vocab_size)`.
- `reset` samples a `LevelEntry` by weight, reuses a cached `Level`, and encodes its `spec`.
- `step` appends the token even when the mask says invalid, applies `invalid_action_penalty`, and terminates on `[SUBMIT]` or truncates at `max_decoder_len`.
- Finish path: decode → apply → `level.run()` → `compute_reward` → `level.reset()`. The reset is mandatory because applying a solution dirties the level.
- `__getstate__` clears the level cache for pickling/async workers.
- Reward is terminal-only and pure (`reward.py`): static failure/`FAILED` → `failure_penalty + placement_reward * placements`; `ERROR` → `error_base + error_progress * crash_step/spec.steps`; unsolved `COMPLETED` → `unsolved_reward + shaping`; solved → `solve_reward + alpha/beta/gamma` clamped cost/area/cycles bonuses. All weights live in `RewardConfig`.

## 7. Training stack

Framework-free PyTorch, wired `model.py` → `rollout.py` → `train.py` → `cli.py`. No Trainer framework; the loop is hand-written and visible.

- `model.py`: `OpusPolicy` is a pre-norm encoder-decoder transformer over the env's Dict observation. Embedding index `vocab_size` is a model-internal BOS (never in any token stream). Decoder input is the BOS-shifted history; output position `t` is the distribution for action `a_t` and the value of state `s_t`. `masked_distribution` turns invalid logits into `-inf` before softmax, so only mask-valid tokens are ever sampled.
- `rollout.py`: `collect_episode` records per-step actions, old logprobs/values, masks, rewards and discounted returns; `collect_batch` spreads episodes round-robin over env instances and stacks teacher-forced tensors plus `mean_return`/`solve_rate`/`mean_length`. Old logprobs/values travel with each step, so the batch format is PPO-ready.
- `train.py`: `Trainer` — masked REINFORCE with a learned value baseline (`A = G - V` detached, value MSE, entropy bonus, Adam, global grad-norm clip). `update()` is the single PPO seam. Checkpoints (`latest.pt` plus numbered `iter_XXXX.pt`) store model/optimizer/iteration and torch/cuda/numpy RNG states plus all configs; `Trainer.resume(path, envs, train_config=None)` restores them. Metrics append to `metrics.jsonl` (train entries + greedy eval entries).
- CLI: `uv run opus-magnum-blackbox train --level tier0_typed_delivery [--resume ckpt.pt] [--iterations N --episodes N --lr F --seed N --device D --run-dir DIR]`; unspecified options fall back to `TrainConfig` defaults on fresh runs and to the checkpoint's stored config on resume. `... eval ckpt.pt [--level <level|bank/group|path.oml>] [--dump-tokens]` prints the exhaustive per-variant greedy solve rate (one deterministic episode per level entry). `... generate ckpt.pt --level <corpus-name|path.oml> [-n N --seed S --out-dir DIR]` samples N independent (re-seeded) solutions and writes them as `.oms` files (only grammar-valid, applicable sequences are written).
- Sizes/knobs: `ModelConfig` (d_model=128, nhead=4, 2+2 layers), loop knobs in `TrainConfig` (`device=""` auto-resolves to cuda when available).

## 8. Levels and corpus

- `corpus.py` scans `levels/` recursively: one `LevelEntry` per `.oml`, named by the path relative to `levels/` with `/` separators (`tier1_simple_arms/train/v000`). Same-stem `.oms` siblings become the entry's `solutions`.
- A resolve name addresses one level, or a directory for the group of every level under it (recursively, sorted) — pipeline configs use group names (`levels = ["tier1_simple_arms/train"]`).
- Config comes from the nearest `manifest.toml` in or above the level's directory: `cost_max` (int, required), `weight` (default 1.0). Top-level levels without a manifest fall back to `RewardConfig().cost_max`; deeper levels without one raise. `TIER0_LEVELS` stays exported as the explicit tier-0 entry (its name/`cost_max` are load-bearing for the CLI default and run names).
- `levels/tier1_simple_arms/` is a **generated bank** covering the full task space (342 train + 86 eval variants = all 428 verified 6x6 entrance/exit pairs, stratified 80/20 by hex-distance class: 170 at distance 1, 258 at distance 2; all verify in-family. `manifest.toml`, git-ignored): run `just gen-tier1 [seed]` before anything that reads it, including repo-bank tests. The generator (`gen_tier1.py`) instantiates the `families.py` single-atom family — 6x6 board, fixed 1-cell typed entrance/exit, `arm`-only allow-list, `target 2` — enumerates every pair with `mode=enumerate` (the CLI default; `BankParams` defaults to the legacy `sample` draw), cuts each distance class ~20% to eval with the split seed, names variants `vNNN`, records mode/seed/counts/distance breakdown in the manifest, and re-validates every emitted pair from disk.
- Tier-1 curriculum run: `just train-tier1` = idempotent `gen-tier1` + `run configs/tier1_simple_arms.toml` (bc on reference solutions → rl with delivery shaping → rl consolidation). Held-out generalization signal: `greedy_solve_rate` (train bank) vs `eval_solve_rate` (eval bank) in `metrics.jsonl`. The consolidation stage runs to a 1000-iteration ceiling with `eval_patience = 20` (`RLStage` key): the stage ends early — logging a `stage_early_stop` metrics row — once the eval rate has not strictly improved for that many consecutive evals.
- Adding a level: drop a `.oml` at the top level (defaults apply) or into a subdirectory (a `manifest.toml` must exist in or above its directory). The scanner picks it up; no corpus entry to write.
- The live simulator supports `target <n>` (per-exit delivery requirement) and `cycling 0|1` (repeat programs); `levels/tier0_typed_delivery.oml` uses both. Older simulator docs may not mention them.
- `LevelEntry.source` may be a `Path` or an in-memory `LevelSpec`; tests use the latter.

## 9. Testing and style

- Tests are behavior-focused and intentionally assert exact tokens, masks, rewards, and full solves. Update tests only when the contract changes; do not weaken the wire-format guards.
- `tests/conftest.py` provides `spec`, `vocab`, `tokenizer`, `masker`, and `known_solution_tokens` (a hand-translation of the tier-0 solution). Treat `known_solution_tokens` as the canonical encoding regression.
- Test modules: `test_grammar.py` (phase/group rules, mask/decoder agreement), `test_tokenizer.py` (encode/decode/round-trip), `test_masking.py` (per-phase masks), `test_reward.py` (reward branches), `test_env.py` (Gymnasium compliance, determinism, truncation, full solve), `test_model.py` (shapes, causality, BOS isolation, masking, encoder padding), `test_rollout.py` (mask validity, hand-computed returns, batch alignment, determinism), `test_train.py` (smoke run, checkpoint/resume).
- Python `>=3.14.10`; `uv`; Ruff line length 120 with a deliberately broad rule set (docstring and unused-import rules are mostly ignored — do not fight it). Use `numpy` integer arrays on the hot path; avoid new dependencies.

## 10. Files to read first

1. `src/opus_magnum_blackbox/grammar.py` — sequence contract.
2. `src/opus_magnum_blackbox/codec.py` — encode/decode boundary.
3. `src/opus_magnum_blackbox/env.py` — episode lifecycle.
4. `src/opus_magnum_blackbox/reward.py` / `config.py` — scoring and caps.
5. `tests/conftest.py` + `tests/test_env.py` — canonical spec and full solve.

## 11. Do not touch unless asked

- `TODO.md` (user tracker), `.session`, `.agent-shell/transcripts/`, `.venv/`, caches, `uv.lock`.
- Do not commit, push, or rewrite history without an explicit request.
