"""Pipeline eval banks: one train env per bank, held-out greedy logging, bank run names."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import make_spec

from opus_magnum_blackbox.config import ModelConfig, TrainConfig
from opus_magnum_blackbox.corpus import LEVELS_DIR
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.stages import PipelineConfig, load_pipeline, run_pipeline
from opus_magnum_blackbox.train import Trainer

TINY = ModelConfig(d_model=32, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64)
TIER0_OMS = LEVELS_DIR / "tier0_typed_delivery.oms"

PIPELINE_TOML = """
[run]
levels = ["tier0_typed_delivery"]
eval_levels = ["tier0_typed_delivery"]
seed = 0
device = "cpu"

[[stage]]
name = "bc"
kind = "pretrain"
epochs = 1
batch_size = 32
augment = false
eval_interval = 1

[[stage]]
name = "rl-a"
kind = "rl"
iterations = 1
episodes = 2

[[stage]]
name = "rl-b"
kind = "rl"
iterations = 1
episodes = 2
eval_interval = 1
eval_episodes = 2
checkpoint_interval = 10
"""


def _bank(count: int, name: str) -> list[LevelEntry]:
    return [
        LevelEntry(source=make_spec(), cost_max=50, name=f"{name}/v{i:02d}", solutions=(TIER0_OMS,))
        for i in range(count)
    ]


def _write_pipeline(tmp_path: Path) -> PipelineConfig:
    path = tmp_path / "pipeline.toml"
    path.write_text(PIPELINE_TOML)
    pipeline = load_pipeline(path)
    return replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))


def read_metrics(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_trainer_keeps_single_train_env_and_separate_eval_envs(tmp_path):
    train, eval_ = _bank(3, "bank/train"), _bank(2, "bank/eval")
    envs = [OpusEnv(train)]
    torch.manual_seed(0)
    trainer = Trainer(
        OpusPolicy(envs[0].tokenizer.config, TINY), envs, TrainConfig(device="cpu"), eval_envs=[OpusEnv(eval_)]
    )

    assert len(trainer.envs) == 1
    assert [entry.name for entry in trainer.envs[0].levels] == [entry.name for entry in train]
    assert len(trainer.eval_envs) == 1
    assert [entry.name for entry in trainer.eval_envs[0].levels] == [entry.name for entry in eval_]


def test_slot_pool_cached_and_entry_envs_separate(tmp_path):
    envs = [OpusEnv(_bank(2, "bank/train"))]
    eval_envs = [OpusEnv(_bank(1, "bank/eval"))]
    torch.manual_seed(0)
    trainer = Trainer(OpusPolicy(envs[0].tokenizer.config, TINY), envs, TrainConfig(device="cpu"), eval_envs=eval_envs)

    pool = trainer._slot_env_pool_for(4)
    assert trainer._slot_env_pool_for(4) is pool  # cached
    assert len(pool) == 4
    assert pool[0] is not pool[1]  # stepping-independent replicas
    assert [entry.name for entry in pool[0].levels] == ["bank/train/v00", "bank/train/v01"]

    entry_env = trainer._entry_env(eval_envs, eval_envs[0].levels[0])
    assert trainer._entry_env(eval_envs, eval_envs[0].levels[0]) is entry_env  # cached per entry
    assert [entry.name for entry in entry_env.levels] == ["bank/eval/v00"]


def test_run_name_for_bank_is_short_and_stable(tmp_path):
    bank = [OpusEnv(_bank(3, "bank/train"))]
    torch.manual_seed(0)
    trainer = Trainer(OpusPolicy(bank[0].tokenizer.config, TINY), bank, TrainConfig(device="cpu"))
    assert trainer.run_dir.name.endswith("-bank_train")
    assert "v00" not in trainer.run_dir.name
    assert len(trainer.run_dir.name) < 60

    single = [OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_typed_delivery")])]
    torch.manual_seed(0)
    trainer = Trainer(OpusPolicy(single[0].tokenizer.config, TINY), single, TrainConfig(device="cpu"))
    assert trainer.run_dir.name.endswith("-tier0_typed_delivery")


def test_pipeline_logs_train_and_eval_bank_rates(tmp_path):
    pipeline = _write_pipeline(tmp_path)
    entries = _bank(3, "bank/train")
    eval_entries = _bank(2, "bank/eval")

    run_dir = run_pipeline(pipeline, entries=entries, eval_entries=eval_entries)

    assert (run_dir / "stage_bc.pt").exists()
    assert (run_dir / "stage_rl-b.pt").exists()
    metrics = read_metrics(run_dir)
    # rl stages log both rates; bc logs the exhaustive eval-bank rate alongside the train one.
    evals = [entry for entry in metrics if "greedy_solve_rate" in entry and entry.get("stage") == "rl-b"]
    assert len(evals) == 1
    assert isinstance(evals[0]["eval_solve_rate"], float)
    bc = [entry for entry in metrics if entry.get("stage") == "bc"]
    assert len(bc) == 1
    assert "greedy_solve_rate" in bc[0]
    assert isinstance(bc[0]["eval_solve_rate"], float)
    train_entries = [entry for entry in metrics if "policy_loss" in entry]
    assert [entry["stage"] for entry in train_entries] == ["rl-a", "rl-b"]


def test_pipeline_resume_rebuilds_eval_envs(tmp_path):
    pipeline = _write_pipeline(tmp_path)
    entries = _bank(3, "bank/train")
    eval_entries = _bank(2, "bank/eval")

    run_dir = run_pipeline(replace(pipeline, stages=pipeline.stages[:2]), entries=entries, eval_entries=eval_entries)
    first = [entry for entry in read_metrics(run_dir) if "policy_loss" in entry]
    assert [entry["stage"] for entry in first] == ["rl-a"]

    continued = run_pipeline(pipeline, entries=entries, eval_entries=eval_entries, resume_dir=run_dir)
    assert continued == run_dir
    assert torch.load(run_dir / "stage_rl-b.pt", weights_only=False)["iteration"] == 2
    evals = [entry for entry in read_metrics(run_dir) if "eval_solve_rate" in entry and entry.get("stage") == "rl-b"]
    assert len(evals) == 1  # eval bank logged after resume, from config not checkpoint


def test_load_pipeline_eval_levels(tmp_path):
    path = tmp_path / "pipeline.toml"
    path.write_text(PIPELINE_TOML)
    pipeline = load_pipeline(path)
    assert pipeline.eval_levels == ("tier0_typed_delivery",)

    path.write_text(PIPELINE_TOML.replace('eval_levels = ["tier0_typed_delivery"]', 'eval_levels = ["tier42"]'))
    with pytest.raises(ValueError, match="unknown levels"):
        load_pipeline(path)

    path.write_text(PIPELINE_TOML.replace("seed = 0", "seed = 0\nnope = 1"))
    with pytest.raises(ValueError, match=r"unknown \[run\] keys"):
        load_pipeline(path)
