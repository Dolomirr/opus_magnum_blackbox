"""Trainer smoke: iterations run, metrics and checkpoints land, resume restores state."""

import json
from pathlib import Path

import pytest
import torch
from conftest import make_spec

from opus_magnum_blackbox.config import ModelConfig, TrainConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
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
