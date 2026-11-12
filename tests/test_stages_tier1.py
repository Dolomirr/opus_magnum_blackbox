"""Tier-1 pipeline smoke: tiny generated bank end-to-end through the stage config path."""

import json
from dataclasses import replace
from pathlib import Path

from opus_magnum_blackbox import gen_tier1
from opus_magnum_blackbox.corpus import resolve_entries
from opus_magnum_blackbox.stages import load_pipeline, run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]

TINY_TOML = """
[run]
levels      = ["tier1_simple_arms/train"]
eval_levels = ["tier1_simple_arms/eval"]
seed        = 0
device      = "cpu"

[[stage]]
name            = "bc"
kind            = "pretrain"
epochs          = 2
lr              = 1e-3
batch_size      = 8
augment         = true
max_variants    = 4
eval_interval   = 1
early_stop_loss = 0.5

[[stage]]
name          = "rl-shape"
kind          = "rl"
iterations    = 1
episodes      = 2
eval_interval = 1
eval_episodes = 2

[[stage]]
name                = "rl-consolidate"
kind                = "rl"
iterations          = 1
episodes            = 2
eval_interval       = 1
eval_episodes       = 2
checkpoint_interval = 10
"""


def test_real_config_loads():
    pipeline = load_pipeline(REPO_ROOT / "configs" / "tier1_simple_arms.toml")
    assert pipeline.levels == ("tier1_simple_arms/train",)
    assert pipeline.eval_levels == ("tier1_simple_arms/eval",)
    bc, shape, consolidate = pipeline.stages
    assert [stage.name for stage in pipeline.stages] == ["bc", "rl-shape", "rl-consolidate"]
    assert bc.config.augment is True
    assert shape.reward_config.shaping_weight > 0
    assert consolidate.reward_config.shaping_weight == 0
    # cost/area/cycles optimization is out of scope for this curriculum.
    assert shape.reward_config.alpha == shape.reward_config.beta == shape.reward_config.gamma == 0


def test_tier1_pipeline_smoke(tmp_path):
    levels_root = tmp_path / "levels"
    gen_tier1.generate_bank(
        levels_root / gen_tier1.FAMILY_NAME, params=gen_tier1.BankParams(seed=5, train_count=2, eval_count=1)
    )
    entries = resolve_entries(["tier1_simple_arms/train"], levels_dir=levels_root)
    eval_entries = resolve_entries(["tier1_simple_arms/eval"], levels_dir=levels_root)
    assert [entry.name for entry in entries] == ["tier1_simple_arms/train/v000", "tier1_simple_arms/train/v001"]
    assert [entry.name for entry in eval_entries] == ["tier1_simple_arms/eval/v000"]

    config = tmp_path / "tier1.toml"
    config.write_text(TINY_TOML)
    pipeline = load_pipeline(config)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))

    run_dir = run_pipeline(pipeline, entries=entries, eval_entries=eval_entries)

    assert (run_dir / "stage_bc.pt").exists()
    assert (run_dir / "stage_rl-shape.pt").exists()
    assert (run_dir / "stage_rl-consolidate.pt").exists()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    # Both eval entry kinds land in metrics.jsonl: train-bank greedy plus eval-bank rate;
    # the bc stage logs both too (eval-bank visibility before RL starts).
    rl_evals = [entry for entry in metrics if "greedy_solve_rate" in entry and entry.get("stage") != "bc"]
    assert rl_evals
    assert all("eval_solve_rate" in entry for entry in rl_evals)
    bc = [entry for entry in metrics if entry.get("stage") == "bc"]
    assert bc
    assert all(isinstance(entry["eval_solve_rate"], float) for entry in bc)
    # Global iteration counter continues across the two rl stages.
    rl = [entry for entry in metrics if "policy_loss" in entry]
    assert [entry["stage"] for entry in rl] == ["rl-shape", "rl-consolidate"]
    assert [entry["iteration"] for entry in rl] == [1, 2]
