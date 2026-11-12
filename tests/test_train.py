"""Trainer smoke: iterations run, metrics and checkpoints land, resume restores state, eval is exhaustive per bank entry."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import known_solution_tokens, make_spec
from opus_magnum_sym import LevelSpec
from torch import nn

from opus_magnum_blackbox.config import ModelConfig, TrainConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import Episode, build_batch
from opus_magnum_blackbox.train import Trainer

TINY = ModelConfig(d_model=32, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64)


def make_env() -> OpusEnv:
    return OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")])


def make_config(
    tmp_path: Path, iterations: int = 3, eval_interval: int = 2, checkpoint_interval: int = 2
) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        episodes_per_iteration=2,
        eval_interval=eval_interval,
        eval_episodes=2,
        checkpoint_interval=checkpoint_interval,
        device="cpu",
        run_dir=str(tmp_path),
    )


def make_trainer(tmp_path: Path, iterations: int = 3) -> Trainer:
    env = make_env()
    torch.manual_seed(0)
    policy = OpusPolicy(env.tokenizer.config, TINY)
    return Trainer(policy, [env], make_config(tmp_path, iterations))


def read_metrics(run_dir: Path) -> list[dict]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


class _ScriptPolicy(nn.Module):
    """
    Test double: greedy argmax always picks the scripted token, so eval outcomes are
    known per level (the canonical spec solves; altered specs fail under the script).
    Matches the rollout collectors' two call shapes: `encode`, then forward with
    precomputed memory; `dummy` keeps the graph differentiable for `update`.
    """

    def __init__(self, vocab_size: int, scripts: dict[bytes, list[int]]):
        super().__init__()
        self.model_config = ModelConfig()
        self.vocab_size = vocab_size
        self.scripts = scripts
        self.dummy = nn.Parameter(torch.zeros(1))
        self._encoder: np.ndarray | None = None

    def encode(self, encoder_input, encoder_length):
        self._encoder = encoder_input.detach().cpu().numpy()
        batch = encoder_input.shape[0]
        return torch.zeros(batch, 4, 8), torch.zeros(batch, 4, dtype=torch.bool)

    def masked_distribution(self, logits, action_mask):
        return torch.distributions.Categorical(logits=logits.masked_fill(~action_mask.bool(), float("-inf")))

    def forward(  # noqa: PLR0913, PLR0917
        self, encoder_input, encoder_length, decoder_history, decoder_length, memory=None, memory_pad_mask=None
    ):
        # BOS-shifted like `OpusPolicy`: T input tokens yield T+1 output positions.
        batch, steps = decoder_history.shape
        logits = torch.full((batch, steps + 1, self.vocab_size), -20.0) + self.dummy * 0.0
        encoder = self._encoder
        keys = [encoder[0].tobytes()] * batch if encoder.shape[0] == 1 else [encoder[r].tobytes() for r in range(batch)]
        for row, key in enumerate(keys):
            for position, token in enumerate(self.scripts.get(key, [])):
                logits[row, position, token] = 20.0
        return logits, torch.zeros(batch, steps + 1)


def _script_policy(env: OpusEnv, solved_spec: LevelSpec) -> _ScriptPolicy:
    padded, _ = env.tokenizer.encode_level_spec(solved_spec)
    return _ScriptPolicy(len(env.tokenizer.vocab), {padded.tobytes(): known_solution_tokens(env.tokenizer.vocab)})


def test_smoke_run(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.run()

    entries = read_metrics(trainer.run_dir)
    training = [e for e in entries if "policy_loss" in e]
    evals = [e for e in entries if "greedy_solve_rate" in e]
    assert [e["iteration"] for e in training] == [1, 2, 3]
    assert [e["iteration"] for e in evals] == [2]
    for entry in training:
        for key in ("mean_reward", "solve_rate", "mean_length", "entropy", "policy_loss", "value_loss", "wall_time"):
            assert key in entry
        assert all(isinstance(entry[k], float) for k in ("policy_loss", "value_loss", "entropy"))
        assert all(torch.isfinite(torch.tensor(entry[k])) for k in ("policy_loss", "value_loss", "entropy"))

    assert (trainer.run_dir / "latest.pt").exists()
    assert (trainer.run_dir / "iter_0002.pt").exists()
    payload = torch.load(trainer.run_dir / "latest.pt", weights_only=False)
    assert payload["iteration"] == 3


def test_resume_restores_state(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.run()
    checkpoint_path = trainer.run_dir / "latest.pt"
    payload = torch.load(checkpoint_path, weights_only=False)

    resumed = Trainer.resume(checkpoint_path, [make_env()], train_config=make_config(tmp_path, iterations=5))
    assert resumed.iteration == 3
    assert torch.equal(torch.get_rng_state(), payload["rng"]["torch"])
    current = resumed.policy.state_dict()
    for key, value in payload["model_state"].items():
        assert torch.equal(current[key], value)

    resumed.run()
    assert resumed.iteration == 5
    entries = read_metrics(resumed.run_dir)
    assert [e["iteration"] for e in entries if "policy_loss" in e] == [4, 5]


def test_device_resolution(tmp_path):
    assert Trainer._auto_device() == ("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(device="cpu", run_dir=str(tmp_path))
    trainer = make_trainer(tmp_path)
    assert trainer.device == torch.device(config.device)


def test_single_iteration_no_eval_or_interval_checkpoints(tmp_path):
    config = make_config(tmp_path, iterations=1, eval_interval=10, checkpoint_interval=10)
    env = make_env()
    torch.manual_seed(0)
    policy = OpusPolicy(env.tokenizer.config, TINY)
    trainer = Trainer(policy, [env], config)
    trainer.run()

    entries = read_metrics(trainer.run_dir)
    assert [e["iteration"] for e in entries] == [1]
    assert (trainer.run_dir / "latest.pt").exists()
    assert not (trainer.run_dir / "iter_0001.pt").exists()


def test_greedy_eval_is_exhaustive_per_entry(tmp_path):
    """
    One deterministic greedy episode per bank entry: a script solving exactly one
    entry of a 3-entry bank yields exactly 1/3, and repeated calls agree bit-for-bit.
    """
    solved = make_spec()
    entries = [
        LevelEntry(source=solved, cost_max=50, name="bank/solved"),
        LevelEntry(source=make_spec(target=2, cycling=False), cost_max=50, name="bank/target2"),
        LevelEntry(source=make_spec(steps=5), cost_max=50, name="bank/short"),
    ]
    env = OpusEnv(entries)
    trainer = Trainer(_script_policy(env, solved), [env], TrainConfig(device="cpu", run_dir=str(tmp_path)))

    assert trainer.evaluate() == pytest.approx(1 / 3)
    assert trainer.evaluate() == trainer.evaluate()


def test_best_eval_checkpoint_written_on_strict_improvement(tmp_path):
    spec = make_spec()
    env = OpusEnv([LevelEntry(source=spec, cost_max=50, name="bank/train")])
    eval_env = OpusEnv([LevelEntry(source=spec, cost_max=50, name="bank/eval")])
    config = TrainConfig(
        iterations=4,
        episodes_per_iteration=2,
        eval_interval=2,
        checkpoint_interval=10,
        device="cpu",
        run_dir=str(tmp_path),
    )
    trainer = Trainer(_script_policy(env, spec), [env], config, eval_envs=[eval_env])
    trainer.run()

    evals = [e for e in read_metrics(trainer.run_dir) if "eval_solve_rate" in e]
    assert [e["eval_solve_rate"] for e in evals] == [1.0, 1.0]
    # `best_eval.pt` appears on the first eval; the equal second rate must not overwrite it.
    best = torch.load(trainer.run_dir / "best_eval.pt", weights_only=False)
    assert best["iteration"] == 2
    assert best["best_eval_rate"] == 1.0


def test_resume_restores_best_eval_rate(tmp_path):
    env = make_env()
    eval_env = OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_eval")])
    torch.manual_seed(0)
    policy = OpusPolicy(env.tokenizer.config, TINY)
    trainer = Trainer(policy, [env], make_config(tmp_path, iterations=1), eval_envs=[eval_env])
    trainer._best_eval_rate = 0.5

    resumed = Trainer.resume(trainer.save_checkpoint(), [env], eval_envs=[eval_env])
    assert resumed._best_eval_rate == 0.5


def test_eval_patience_stops_after_constant_evals(tmp_path):
    """
    Constant eval rate: `run(eval_patience=k)` ends the loop after exactly k evals with one
    `stage_early_stop` row; without patience the full iteration target runs.
    """
    spec = make_spec()
    env = OpusEnv([LevelEntry(source=spec, cost_max=50, name="bank/train")])
    eval_env = OpusEnv([LevelEntry(source=spec, cost_max=50, name="bank/eval")])
    config = TrainConfig(
        iterations=8,
        episodes_per_iteration=2,
        eval_interval=2,
        checkpoint_interval=10,
        device="cpu",
        run_dir=str(tmp_path / "early"),
    )

    trainer = Trainer(_script_policy(env, spec), [env], config, eval_envs=[eval_env])
    trainer.run(eval_patience=2)

    assert trainer.iteration == 4  # evals at iterations 2 and 4; the second stale eval ends the stage
    entries = read_metrics(trainer.run_dir)
    stops = [entry for entry in entries if entry.get("stage_early_stop")]
    assert len(stops) == 1
    assert stops[0]["iteration"] == 4
    assert stops[0]["best_eval_rate"] == stops[0]["eval_solve_rate"] == 1.0
    assert [entry["iteration"] for entry in entries if "policy_loss" in entry] == [1, 2, 3, 4]
    assert torch.load(trainer.run_dir / "latest.pt", weights_only=False)["iteration"] == 4

    full = Trainer(_script_policy(env, spec), [env], config, eval_envs=[eval_env])
    full.run()
    assert full.iteration == 8
    assert not [entry for entry in read_metrics(full.run_dir) if entry.get("stage_early_stop")]


def test_eval_patience_ignored_without_eval_envs(tmp_path):
    """
    Patience falls back to never stopping when there is no eval bank to measure.
    """
    env = make_env()
    config = make_config(tmp_path, iterations=3, eval_interval=1)
    trainer = Trainer(OpusPolicy(env.tokenizer.config, TINY), [env], config)
    trainer.run(eval_patience=1)

    assert trainer.iteration == 3
    assert not [entry for entry in read_metrics(trainer.run_dir) if entry.get("stage_early_stop")]


def test_slot_pool_seeding_stable_across_trainers(tmp_path):
    """
    Pool replicas are seeded from (train seed, slot count): two fresh trainers with the
    same config build pools with identical draw streams, and replicas draw independently.
    """

    def fresh_trainer() -> Trainer:
        env = OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name=f"bank/v{i:02d}") for i in range(4)])
        torch.manual_seed(0)
        policy = OpusPolicy(env.tokenizer.config, TINY)
        trainer = Trainer(policy, [env], TrainConfig(device="cpu", seed=11, run_dir=str(tmp_path)))
        trainer._seed()
        return trainer

    pools = [fresh_trainer()._slot_env_pool_for(3) for _ in range(2)]
    draws = [[env.reset()[1]["level"] for _ in range(4)] for env in pools[0]]
    assert draws == [[env.reset()[1]["level"] for _ in range(4)] for env in pools[1]]
    assert len({tuple(seq) for seq in draws}) > 1  # replicas do not draw in lockstep


# --- self-imitation buffer and step (stage 3 of the discovery plan) ---


def make_sil_episode(env: OpusEnv, tokens: list[int], solved: bool, level: str) -> Episode:
    spec = env.levels[0].source
    padded, encoder_length = env.tokenizer.encode_level_spec(spec)
    return Episode(
        encoder_input=padded,
        encoder_length=encoder_length,
        tokens=tokens,
        masks=[env.masker.mask(tokens[:t], spec) for t in range(len(tokens))],
        logprobs=[0.0] * len(tokens),
        values=[0.0] * len(tokens),
        rewards=[0.0] * len(tokens),
        returns=[0.0] * len(tokens),
        solved=solved,
        terminated=True,
        level=level,
    )


def test_sil_buffer_dedupes_per_level_and_caps(tmp_path):
    env = make_env()
    vocab = env.tokenizer.vocab
    good = known_solution_tokens(vocab)
    failed = [vocab.special_id("[SUBMIT]")]
    episodes = [
        make_sil_episode(env, good[:-1], True, "lvl/a"),
        make_sil_episode(env, failed, False, "lvl/b"),
        make_sil_episode(env, good, True, "lvl/a"),  # refresh: the latest solve must win
        make_sil_episode(env, good, True, "lvl/c"),
    ]
    batch = build_batch(episodes, vocab.pad_id)

    trainer = make_trainer(tmp_path)
    trainer.train_config = replace(trainer.train_config, sil_coef=0.1)
    loss, new_levels = trainer._sil_step(batch)
    assert loss is not None
    assert set(trainer._sil_buffer) == {"lvl/a", "lvl/c"}  # failed row never buffered
    assert new_levels == ["lvl/a", "lvl/c"]  # the refresh is not a new level
    assert int(trainer._sil_buffer["lvl/a"].actions.shape[0]) == len(good)  # latest solve kept

    capped = make_trainer(tmp_path / "cap")
    capped.train_config = replace(capped.train_config, sil_coef=0.1, sil_buffer_cap=1)
    capped._sil_step(batch)
    assert set(capped._sil_buffer) == {"lvl/c"}  # oldest levels evicted first


def test_sil_disabled_is_inert(tmp_path):
    env = make_env()
    trainer = make_trainer(tmp_path)  # default sil_coef=0
    vocab = env.tokenizer.vocab
    batch = build_batch([make_sil_episode(env, known_solution_tokens(vocab), True, "lvl/a")], vocab.pad_id)
    assert trainer._sil_step(batch) == (None, [])
    assert trainer._sil_buffer == {}


def test_sil_loss_scales_linearly(tmp_path):
    """
    Same weights and same buffer: the SIL loss returned is the pre-step BC NLL,
    so halving `sil_coef` must halve it.
    """
    env = make_env()
    vocab = env.tokenizer.vocab
    batch = build_batch([make_sil_episode(env, known_solution_tokens(vocab), True, "lvl/a")], vocab.pad_id)

    losses = {}
    for sil_coef in (0.5, 1.0):
        torch.manual_seed(0)
        policy = OpusPolicy(env.tokenizer.config, TINY)
        trainer = Trainer(policy, [env], replace(make_config(tmp_path / str(sil_coef)), sil_coef=sil_coef))
        torch.manual_seed(7)  # same minibatch pick under both coefficients
        losses[sil_coef] = trainer._sil_step(batch)[0]

    assert losses[1.0] == pytest.approx(2.0 * losses[0.5], rel=1e-4)


def test_sil_step_leaves_update_untouched(tmp_path):
    """
    The REINFORCE step is byte-identical with and without SIL; the SIL step itself
    runs off the buffered solved row only.
    """
    env = make_env()
    vocab = env.tokenizer.vocab
    batch = build_batch(
        [
            make_sil_episode(env, known_solution_tokens(vocab), True, "lvl/a"),
            make_sil_episode(env, [vocab.special_id("[SUBMIT]")], False, "lvl/failed"),
        ],
        vocab.pad_id,
    )

    outcomes = []
    for sil_coef in (0.0, 0.1):
        torch.manual_seed(0)
        policy = OpusPolicy(env.tokenizer.config, TINY)
        trainer = Trainer(policy, [env], replace(make_config(tmp_path / f"up{sil_coef}"), sil_coef=sil_coef))
        outcomes.append(trainer.update(batch))
    assert outcomes[0] == outcomes[1]

    torch.manual_seed(0)
    policy = OpusPolicy(env.tokenizer.config, TINY)
    trainer = Trainer(policy, [env], replace(make_config(tmp_path / "sil"), sil_coef=0.1))
    loss, new_levels = trainer._sil_step(batch)
    assert loss is not None
    assert new_levels == ["lvl/a"]
    assert set(trainer._sil_buffer) == {"lvl/a"}


def test_sil_metrics_logged_only_when_enabled(tmp_path):
    spec = make_spec()
    env = OpusEnv([LevelEntry(source=spec, cost_max=50, name="bank/train")])
    base = make_config(tmp_path, iterations=1, eval_interval=10, checkpoint_interval=10)

    off = Trainer(_script_policy(env, spec), [env], base)
    off.run()
    entry = read_metrics(off.run_dir)[0]
    assert "sil_loss" not in entry
    assert "sil_buffer" not in entry

    on = Trainer(_script_policy(env, spec), [env], replace(base, sil_coef=0.1))
    on.run()
    entry = read_metrics(on.run_dir)[0]
    assert entry["sil_loss"] >= 0.0
    assert entry["sil_buffer"] == 1  # both episodes play the same level: one deduped entry
    assert entry["sil_new_levels"] == ["bank/train"]
