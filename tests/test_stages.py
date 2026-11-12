"""Stage runner: TOML parsing/routing and a bc + rl pipeline smoke."""

import json
from dataclasses import replace

import pytest
import torch
from conftest import make_spec

from opus_magnum_blackbox.config import RewardConfig
from opus_magnum_blackbox.corpus import LEVELS_DIR
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.stages import PretrainStage, RLStage, load_pipeline, resolve_entries, run_pipeline

TIER0_OMS = LEVELS_DIR / "tier0_typed_delivery.oms"


def _write_pipeline(tmp_path, text):
    path = tmp_path / "pipeline.toml"
    path.write_text(text)
    return path


def test_load_pipeline_routes_keys_and_preserves_order(tmp_path):
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]
seed = 3
device = "cpu"

[[stage]]
name = "bc"
kind = "pretrain"
epochs = 7
lr = 1e-3
batch_size = 4
augment = false
max_variants = 2

[[stage]]
name = "rl-consolidate"
kind = "rl"
iterations = 9
episodes = 5
lr = 1e-4
entropy_coef = 0.002

[[stage]]
name = "rl-optimise"
kind = "rl"
iterations = 11
alpha = 0.2
beta = 0.2
gamma = 0.2
""",
    )
    pipeline = load_pipeline(path)
    assert pipeline.levels == ("tier0_typed_delivery",)
    assert pipeline.train_config.seed == 3
    assert pipeline.train_config.device == "cpu"
    assert [stage.name for stage in pipeline.stages] == ["bc", "rl-consolidate", "rl-optimise"]

    bc, consolidate, optimise = pipeline.stages
    assert isinstance(bc, PretrainStage)
    assert bc.config.epochs == 7
    assert bc.config.lr == 1e-3
    assert bc.config.batch_size == 4
    assert bc.config.augment is False
    assert bc.config.max_variants == 2

    assert isinstance(consolidate, RLStage)
    assert consolidate.iterations == 9
    assert consolidate.train_config.episodes_per_iteration == 5  # `episodes` alias
    assert consolidate.train_config.lr == 1e-4
    assert consolidate.train_config.entropy_coef == 0.002
    assert consolidate.train_config.seed == 3  # run-level values flow through
    assert consolidate.reward_config == RewardConfig()

    assert isinstance(optimise, RLStage)
    assert optimise.iterations == 11
    # `gamma` routes to the RewardConfig cycles bonus, not the TrainConfig discount.
    assert optimise.reward_config.gamma == 0.2
    assert optimise.reward_config.alpha == optimise.reward_config.beta == 0.2
    assert optimise.train_config.gamma == 1.0


def test_unknown_stage_key_rejected(tmp_path):
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]

[[stage]]
name = "bc"
kind = "pretrain"
entropy_coeff = 0.01
""",
    )
    with pytest.raises(ValueError, match=r"stage 'bc': unknown key 'entropy_coeff'"):
        load_pipeline(path)


def test_eval_patience_routes_to_rl_stage_only(tmp_path):
    text = """
[run]
levels = ["tier0_typed_delivery"]

[[stage]]
name = "rl-consolidate"
kind = "rl"
iterations = 5
eval_patience = 20
"""
    path = _write_pipeline(tmp_path, text)
    stage = load_pipeline(path).stages[0]
    assert isinstance(stage, RLStage)
    assert stage.eval_patience == 20

    path.write_text(text.replace("eval_patience = 20\n", ""))
    assert load_pipeline(path).stages[0].eval_patience is None  # absent key: never stop early

    for bad in ("eval_patience = 0", "eval_patience = -1", "eval_patience = 2.5", "eval_patience = true"):
        path.write_text(text.replace("eval_patience = 20", bad))
        with pytest.raises(ValueError, match="eval_patience"):
            load_pipeline(path)

    # pretrain stages do not take the rl-only key; unknown keys still hard-error
    pretrain = text.replace('name = "rl-consolidate"\nkind = "rl"\niterations = 5', 'name = "bc"\nkind = "pretrain"')
    path.write_text(pretrain)
    with pytest.raises(ValueError, match=r"stage 'bc': unknown key 'eval_patience'"):
        load_pipeline(path)


def test_invalid_kind_and_missing_stages_rejected(tmp_path):
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]

[[stage]]
name = "nope"
kind = "distill"
""",
    )
    with pytest.raises(ValueError, match="kind"):
        load_pipeline(path)

    path = _write_pipeline(tmp_path, '[run]\nlevels = ["tier0_typed_delivery"]\n')
    with pytest.raises(ValueError, match="at least one"):
        load_pipeline(path)

    path = _write_pipeline(tmp_path, "")
    with pytest.raises(ValueError, match=r"\[run\]"):
        load_pipeline(path)


def test_unknown_level_rejected(tmp_path):
    path = _write_pipeline(tmp_path, '[run]\nlevels = ["tier42"]\n\n[[stage]]\nname = "bc"\nkind = "pretrain"\n')
    with pytest.raises(ValueError, match="unknown levels"):
        load_pipeline(path)
    with pytest.raises(ValueError, match="unknown levels"):
        resolve_entries(["tier42"])


def test_pipeline_smoke_bc_then_rl(tmp_path):
    oms = tmp_path / "tier0_test.oms"
    oms.write_text(TIER0_OMS.read_text())
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]
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
""",
    )
    pipeline = load_pipeline(path)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(oms,))]

    run_dir = run_pipeline(pipeline, entries=entries)

    assert (run_dir / "stage_bc.pt").exists()
    assert (run_dir / "stage_rl-a.pt").exists()
    assert (run_dir / "stage_rl-b.pt").exists()
    # Global iteration counter continues across the two rl stages.
    assert torch.load(run_dir / "stage_rl-b.pt", weights_only=False)["iteration"] == 2

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    bc_entries = [entry for entry in metrics if entry.get("stage") == "bc"]
    assert [entry["iteration"] for entry in bc_entries] == [1]
    assert bc_entries[0]["loss"] < 10.0
    train_entries = [entry for entry in metrics if "policy_loss" in entry]
    assert [entry["iteration"] for entry in train_entries] == [1, 2]
    assert [entry["stage"] for entry in train_entries] == ["rl-a", "rl-b"]
    evals = [entry for entry in metrics if "greedy_solve_rate" in entry and entry.get("stage") == "rl-b"]
    assert [entry["iteration"] for entry in evals] == [2]


def _resume_setup(tmp_path):
    oms = tmp_path / "tier0_test.oms"
    oms.write_text(TIER0_OMS.read_text())
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]
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
""",
    )
    pipeline = load_pipeline(path)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(oms,))]
    return pipeline, entries


def test_pipeline_resume_continues_and_refuses_when_done(tmp_path):
    pipeline, entries = _resume_setup(tmp_path)

    run_dir = run_pipeline(replace(pipeline, stages=pipeline.stages[:2]), entries=entries)

    with pytest.raises(ValueError, match="no stage checkpoint"):
        run_pipeline(pipeline, entries=entries, resume_dir=tmp_path / "missing-dir")

    continued = run_pipeline(pipeline, entries=entries, resume_dir=run_dir)
    assert continued == run_dir
    assert (run_dir / "stage_rl-b.pt").exists()
    assert torch.load(run_dir / "stage_rl-b.pt", weights_only=False)["iteration"] == 2

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    bc_entries = [entry for entry in metrics if entry.get("stage") == "bc"]
    assert len(bc_entries) == 1  # completed stages are skipped, not re-run
    train_entries = [entry for entry in metrics if "policy_loss" in entry]
    assert [entry["iteration"] for entry in train_entries] == [1, 2]
    assert [entry["stage"] for entry in train_entries] == ["rl-a", "rl-b"]

    with pytest.raises(ValueError, match="already completed"):
        run_pipeline(pipeline, entries=entries, resume_dir=run_dir)
