"""Stage runner: TOML parsing/routing and a bc + rl pipeline smoke."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from conftest import make_spec

from opus_magnum_blackbox.config import ModelConfig, RewardConfig, TrainConfig
from opus_magnum_blackbox.corpus import LEVELS_DIR
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.stages import PretrainStage, RLStage, load_pipeline, resolve_entries, run_pipeline
from opus_magnum_blackbox.train import Trainer

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


def test_skip_when_bc_eval_gte_routes_to_rl_stage_only(tmp_path):
    text = """
[run]
levels = ["tier0_typed_delivery"]

[[stage]]
name = "rl-consolidate"
kind = "rl"
iterations = 5
skip_when_bc_eval_gte = 0.95
"""
    path = _write_pipeline(tmp_path, text)
    stage = load_pipeline(path).stages[0]
    assert isinstance(stage, RLStage)
    assert stage.skip_when_bc_eval_gte == 0.95

    path.write_text(text.replace("skip_when_bc_eval_gte = 0.95\n", ""))
    assert load_pipeline(path).stages[0].skip_when_bc_eval_gte is None  # absent key: never skip

    for bad in ("skip_when_bc_eval_gte = 1.5", "skip_when_bc_eval_gte = -0.5", "skip_when_bc_eval_gte = true", 'skip_when_bc_eval_gte = "0.9"'):
        path.write_text(text.replace("skip_when_bc_eval_gte = 0.95", bad))
        with pytest.raises(ValueError, match="skip_when_bc_eval_gte"):
            load_pipeline(path)

    # pretrain stages do not take the rl-only key; unknown keys still hard-error
    pretrain = text.replace('name = "rl-consolidate"\nkind = "rl"\niterations = 5', 'name = "bc"\nkind = "pretrain"')
    path.write_text(pretrain)
    with pytest.raises(ValueError, match=r"stage 'bc': unknown key 'skip_when_bc_eval_gte'"):
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


def test_autodiscovery_defaults_and_routes(tmp_path):
    text = """
[run]
levels = ["tier0_typed_delivery"]
{dial}
[[stage]]
name = "bc"
kind = "pretrain"
epochs = 100
"""
    path = _write_pipeline(tmp_path, text.format(dial=""))
    pipeline = load_pipeline(path)
    assert pipeline.autodiscovery == 1
    assert pipeline.stages[0].config.epochs == 100  # k=1 keeps the configured count

    path.write_text(text.format(dial="autodiscovery = 1\n"))
    assert load_pipeline(path).stages[0].config.epochs == 100

    path.write_text(text.format(dial="autodiscovery = 5\n"))
    pipeline = load_pipeline(path)
    assert pipeline.autodiscovery == 5
    assert pipeline.stages[0].config.epochs == 56  # round(100 * 5 / 9)

    path.write_text(text.format(dial="autodiscovery = 10\n"))
    assert load_pipeline(path).stages[0].config.epochs == 0

    # rl stages never scale
    path.write_text(
        text.format(dial="autodiscovery = 10\n").replace(
            'name = "bc"\nkind = "pretrain"\nepochs = 100', 'name = "rl-a"\nkind = "rl"\niterations = 100'
        )
    )
    assert load_pipeline(path).stages[0].iterations == 100


def test_autodiscovery_out_of_range_rejected(tmp_path):
    text = """
[run]
levels = ["tier0_typed_delivery"]
autodiscovery = {value}

[[stage]]
name = "bc"
kind = "pretrain"
"""
    path = _write_pipeline(tmp_path, text)
    for value in ("0", "11", "-3", "2.5", "true", '"5"'):
        path.write_text(text.replace("{value}", value))
        with pytest.raises(ValueError, match=r"autodiscovery` must be an integer in 1\.\.10"):
            load_pipeline(path)


def _run_ad_pipeline(tmp_path, autodiscovery: int | None, with_solutions: bool = True):
    """
    Tiny bc + rl pipeline with the dial set (None omits the key); returns run dir and metrics rows.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    oms = tmp_path / "tier0_test.oms"
    oms.write_text(TIER0_OMS.read_text())
    dial = "" if autodiscovery is None else f"autodiscovery = {autodiscovery}\n"
    path = _write_pipeline(
        tmp_path,
        f"""
[run]
levels = ["tier0_typed_delivery"]
seed = 0
device = "cpu"
{dial}
[[stage]]
name = "bc"
kind = "pretrain"
epochs = 9
batch_size = 32
augment = false
eval_interval = 3

[[stage]]
name = "rl-a"
kind = "rl"
iterations = 1
episodes = 2
""",
    )
    pipeline = load_pipeline(path)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    solutions = (oms,) if with_solutions else ()
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=solutions)]
    run_dir = run_pipeline(pipeline, entries=entries)
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return run_dir, metrics


def test_autodiscovery_scales_bc_and_stamps_run(tmp_path):
    run_dir, metrics = _run_ad_pipeline(tmp_path, 5)

    # derived tag: config stem + dial + non-default rl knobs (episodes 2)
    assert run_dir.name.endswith("-pipeline-ad5-ep2")
    assert all(entry["autodiscovery"] == 5 for entry in metrics)  # stamped on every row
    # round(9 * 5 / 9) = 5 bc epochs, logged on eval epoch 3 and the final epoch 5
    bc = [entry for entry in metrics if entry.get("stage") == "bc"]
    assert [entry["iteration"] for entry in bc] == [3, 5]
    rl = [entry for entry in metrics if "policy_loss" in entry]
    assert [entry["iteration"] for entry in rl] == [1]


def test_autodiscovery_ten_skips_bc(tmp_path):
    # no .oms siblings: the skip must not trip the "no solution sequences found" error
    run_dir, metrics = _run_ad_pipeline(tmp_path, 10, with_solutions=False)

    skips = [entry for entry in metrics if entry.get("stage_skip")]
    assert [entry["stage"] for entry in skips] == ["bc"]
    assert skips[0]["iteration"] == 0
    assert skips[0]["autodiscovery"] == 10
    assert "bc epochs to 0" in skips[0]["reason"]
    assert all("loss" not in entry for entry in metrics)  # no bc training happened
    # the skipped stage still checkpoints, so resume treats it as completed
    assert (run_dir / "stage_bc.pt").exists()
    # rl proceeds from the untrained policy
    rl = [entry for entry in metrics if "policy_loss" in entry]
    assert [entry["iteration"] for entry in rl] == [1]
    assert (run_dir / "stage_rl-a.pt").exists()


def test_autodiscovery_one_is_keyless_behavior(tmp_path):
    keyless_dir, keyless = _run_ad_pipeline(tmp_path / "keyless", None)
    dialed_dir, dialed = _run_ad_pipeline(tmp_path / "dialed", 1)

    def rows(metrics):
        return [
            {key: value for key, value in entry.items() if key not in ("timestamp", "wall_time")} for entry in metrics
        ]

    assert rows(dialed) == rows(keyless)
    # k=1 keeps historical run naming on both runs
    assert "-ad" not in keyless_dir.name
    assert "-ad" not in dialed_dir.name


def _run_skip_pipeline(tmp_path, *, eval_bank: bool, threshold: bool):
    """
    Tiny bc + rl-a + rl-b pipeline; `eval_bank` gives bc a held-out bank so the
    bc-end eval rate exists, `threshold` toggles the `skip_when_bc_eval_gte`
    key on rl-a (0.0: any existing rate meets it).
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    oms = tmp_path / "tier0_test.oms"
    oms.write_text(TIER0_OMS.read_text())
    path = _write_pipeline(
        tmp_path,
        f"""
[run]
levels = ["tier0_typed_delivery"]
seed = 0
device = "cpu"
{'eval_levels = ["tier0_typed_delivery"]' if eval_bank else ""}
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
{"skip_when_bc_eval_gte = 0.0" if threshold else ""}
[[stage]]
name = "rl-b"
kind = "rl"
iterations = 1
episodes = 2
""",
    )
    pipeline = load_pipeline(path)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(oms,))]
    run_dir = run_pipeline(pipeline, entries=entries)
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    return run_dir, metrics


def test_rl_stage_skips_when_bc_eval_meets_threshold(tmp_path):
    """Threshold met: the stage is skipped with a `stage_skip` row naming the rate, and still checkpoints."""
    run_dir, metrics = _run_skip_pipeline(tmp_path, eval_bank=True, threshold=True)

    skips = [entry for entry in metrics if entry.get("stage_skip") and entry.get("stage") == "rl-a"]
    assert len(skips) == 1
    assert isinstance(skips[0]["eval_solve_rate"], float)  # the row names the rate
    assert "skip_when_bc_eval_gte 0" in skips[0]["reason"]
    # rl-a never trained: its only iteration slot goes to rl-b
    assert [entry["iteration"] for entry in metrics if "policy_loss" in entry] == [1]
    assert [entry["stage"] for entry in metrics if "policy_loss" in entry] == ["rl-b"]
    # skipped stages still checkpoint, so resume treats them as completed
    assert (run_dir / "stage_rl-a.pt").exists()
    assert torch.load(run_dir / "stage_bc.pt", weights_only=False)["last_bc_eval_rate"] == skips[0]["eval_solve_rate"]


def test_rl_stage_runs_when_bc_rate_unknown_or_key_absent(tmp_path):
    """No held-out bank (rate unknown) or absent key: rl-a runs normally, no skip row."""
    _, no_bank = _run_skip_pipeline(tmp_path / "no-bank", eval_bank=False, threshold=True)
    assert not [entry for entry in no_bank if entry.get("stage_skip")]
    assert [entry["stage"] for entry in no_bank if "policy_loss" in entry] == ["rl-a", "rl-b"]

    _, keyless = _run_skip_pipeline(tmp_path / "keyless", eval_bank=True, threshold=False)
    assert not [entry for entry in keyless if entry.get("stage_skip")]
    assert [entry["stage"] for entry in keyless if "policy_loss" in entry] == ["rl-a", "rl-b"]


def test_run_tag_derivation_from_knobs(tmp_path):
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]
autodiscovery = 9

[[stage]]
name = "rl-shape"
kind = "rl"
iterations = 400
episodes = 32
lr = 1e-4
entropy_coef = 0.005
shaping_weight = 0.5
progress_weight = 0.3
sil_coef = 0.4
sil_batch_size = 32
sil_weighting = "scarcity"
sample_temperature = 2.0

[[stage]]
name = "rl-consolidate"
kind = "rl"
iterations = 2000
episodes = 32
lr = 1e-4
entropy_coef = 0.001
progress_weight = 0.3
sil_coef = 0.4
sil_batch_size = 32
sil_weighting = "scarcity"
sample_temperature = 2.0
eval_patience = 20
""",
    )
    # stem first, then the dial, then per-stage deltas in field order, deduped across stages
    assert load_pipeline(path).tag == "pipeline-ad9-lr0001-ent005-t2-sil4-silb32-silwscarcity-sh5-pg3-ent001"
    path.write_text(path.read_text().replace("autodiscovery = 9\n", ""))
    assert load_pipeline(path).tag == "pipeline-lr0001-ent005-t2-sil4-silb32-silwscarcity-sh5-pg3-ent001"


def test_run_tag_override_wins(tmp_path):
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]
tag = "custom-arm"

[[stage]]
name = "rl-a"
kind = "rl"
iterations = 1
episodes = 2
""",
    )
    assert load_pipeline(path).tag == "custom-arm"

    for bad in ('tag = ""', 'tag = "a/b"', 'tag = "a\\\\b"', "tag = 7"):
        path.write_text(path.read_text().replace('tag = "custom-arm"', bad))
        with pytest.raises(ValueError, match=r"tag` must be a non-empty string without path separators"):
            load_pipeline(path)


def test_config_toml_copied_to_run_dir(tmp_path):
    oms = tmp_path / "tier0_test.oms"
    oms.write_text(TIER0_OMS.read_text())
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]
seed = 0
device = "cpu"
tag = "custom-arm"

[[stage]]
name = "rl-a"
kind = "rl"
iterations = 1
episodes = 2
""",
    )
    pipeline = load_pipeline(path)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(oms,))]

    run_dir = run_pipeline(pipeline, entries=entries)
    assert run_dir.name.endswith("-custom-arm")  # override wins over derivation
    assert (run_dir / "config.toml").read_bytes() == path.read_bytes()

    # programmatic configs carry no source: no provenance copy
    bare_run = run_pipeline(replace(pipeline, source=None), entries=entries)
    assert not (bare_run / "config.toml").exists()


def test_same_config_reruns_get_unique_dirs(tmp_path):
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
name = "rl-a"
kind = "rl"
iterations = 1
episodes = 2
""",
    )
    pipeline = load_pipeline(path)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(oms,))]

    first = run_pipeline(pipeline, entries=entries)
    second = run_pipeline(pipeline, entries=entries)
    assert first != second  # timestamp prefix keeps same-config reruns unique
    assert first.name.split("-", 3)[3] == second.name.split("-", 3)[3]  # same levels label + derived tag


# --- [run] init_from: weights-only warm start (tier-2 stage 3) ---


def _warm_start_checkpoint(tmp_path: Path, model_config: ModelConfig | None = None) -> Path:
    """
    Checkpoint for warm-start tests: real policy weights offset so they cannot match
    a fresh init, plus the stored model config.
    """
    env = OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")])
    policy = OpusPolicy(env.tokenizer.config, model_config or ModelConfig())
    for parameter in policy.parameters():
        parameter.data.add_(1.0)
    path = tmp_path / "warm.pt"
    torch.save({"model_state": policy.state_dict(), "model_config": asdict(model_config or ModelConfig())}, path)
    return path


def _fresh_trainer(tmp_path: Path) -> Trainer:
    env = OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")])
    torch.manual_seed(0)
    return Trainer(
        OpusPolicy(env.tokenizer.config, ModelConfig()), [env], TrainConfig(device="cpu", run_dir=str(tmp_path))
    )


def test_warm_start_loads_weights_only(tmp_path):
    """
    `warm_start` copies the checkpoint's model weights into the fresh policy;
    optimizer state, iteration counter and eval bookkeeping stay untouched.
    """
    checkpoint = _warm_start_checkpoint(tmp_path)
    trainer = _fresh_trainer(tmp_path / "run")
    fresh = {key: value.clone() for key, value in trainer.policy.state_dict().items()}

    trainer.warm_start(checkpoint)

    loaded = torch.load(checkpoint, weights_only=False)["model_state"]
    current = trainer.policy.state_dict()
    for key, value in loaded.items():
        assert torch.equal(current[key], value)
        assert not torch.equal(fresh[key], value)  # the offset makes warm-started weights ≠ fresh
    assert trainer.iteration == 0
    assert trainer.optimizer.state_dict()["state"] == {}  # no optimizer state came along
    assert trainer._best_eval_rate is None


def test_warm_start_rejects_model_config_mismatch(tmp_path):
    checkpoint = _warm_start_checkpoint(tmp_path, ModelConfig(d_model=64))
    trainer = _fresh_trainer(tmp_path / "run")
    with pytest.raises(ValueError, match="model config"):
        trainer.warm_start(checkpoint)


def test_init_from_pipeline_warm_starts_and_logs(tmp_path):
    """
    `[run] init_from` warm-starts the fresh run: a `warm_start` metrics row at iteration 0
    names the source, and the stage counter starts from zero.
    """
    checkpoint = _warm_start_checkpoint(tmp_path)
    oms = tmp_path / "tier0_test.oms"
    oms.write_text(TIER0_OMS.read_text())
    path = _write_pipeline(
        tmp_path,
        f"""
[run]
levels = ["tier0_typed_delivery"]
seed = 0
device = "cpu"
init_from = "{checkpoint}"

[[stage]]
name = "rl-a"
kind = "rl"
iterations = 1
episodes = 2
""",
    )
    pipeline = load_pipeline(path)
    assert pipeline.init_from == checkpoint
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(oms,))]

    run_dir = run_pipeline(pipeline, entries=entries)

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    warm = [entry for entry in metrics if "warm_start" in entry]
    assert [entry["iteration"] for entry in warm] == [0]
    assert warm[0]["warm_start"] == str(checkpoint)
    # fresh iteration counter: the first (and only) rl stage lands on iteration 1
    assert torch.load(run_dir / "stage_rl-a.pt", weights_only=False)["iteration"] == 1


def test_init_from_run_key_validation(tmp_path):
    """
    `init_from` must be a non-empty string path; absent key parses to None;
    unknown [run] keys stay hard errors next to the new known key.
    """
    template = (
        '[run]\nlevels = ["tier0_typed_delivery"]\n{key}\n[[stage]]\nname = "rl-a"\nkind = "rl"\niterations = 1\n'
    )
    path = _write_pipeline(tmp_path, template.format(key=""))
    assert load_pipeline(path).init_from is None

    path.write_text(template.format(key='init_from = "runs/warm.pt"'))
    assert load_pipeline(path).init_from == Path("runs/warm.pt")

    for bad in ("init_from = 7", 'init_from = ""', "init_from = true"):
        path.write_text(template.format(key=bad))
        with pytest.raises(ValueError, match="init_from"):
            load_pipeline(path)

    path.write_text(template.format(key="init_form = 'runs/warm.pt'"))
    with pytest.raises(ValueError, match=r"unknown \[run\] keys"):
        load_pipeline(path)


# --- [run] model: capacity arm (tier-2 E4) ---


def test_model_override_parses_and_defaults_to_size_128(tmp_path):
    """
    An absent `[run] model` builds the current 128/4/2+2 model exactly;
    a present table routes `ModelConfig` fields and tags the run dir with `m<d_model>`.
    """
    template = (
        '[run]\nlevels = ["tier0_typed_delivery"]\n{key}\n[[stage]]\nname = "rl-a"\nkind = "rl"\niterations = 1\n'
    )
    path = _write_pipeline(tmp_path, template.format(key=""))
    pipeline = load_pipeline(path)
    assert pipeline.model_config == ModelConfig()
    assert "m128" not in pipeline.tag

    path.write_text(
        template.format(
            key="[run.model]\nd_model = 256\nnhead = 8\nencoder_layers = 3\ndecoder_layers = 3\ndim_feedforward = 1024"
        )
    )
    pipeline = load_pipeline(path)
    assert pipeline.model_config == ModelConfig(
        d_model=256, nhead=8, encoder_layers=3, decoder_layers=3, dim_feedforward=1024
    )
    assert "m256" in pipeline.tag


def test_model_override_rejects_unknown_and_mistyped_keys(tmp_path):
    template = '[run]\nlevels = ["tier0_typed_delivery"]\n[run.model]\n{key}\n[[stage]]\nname = "rl-a"\nkind = "rl"\niterations = 1\n'
    path = _write_pipeline(tmp_path, template.format(key="d_model = 256\nhidden_size = 512"))
    with pytest.raises(ValueError, match=r"unknown \[run.model\] keys"):
        load_pipeline(path)

    for bad in ("d_model = 256.0", "d_model = true", "dropout = 1", "nhead = '8'"):
        path.write_text(template.format(key=bad))
        with pytest.raises(ValueError, match=r"\[run.model\]"):
            load_pipeline(path)


def test_model_override_builds_the_bigger_policy(tmp_path):
    """The pipeline constructs `OpusPolicy` from the overridden `ModelConfig`."""
    oms = tmp_path / "tier0_test.oms"
    oms.write_text(TIER0_OMS.read_text())
    path = _write_pipeline(
        tmp_path,
        """
[run]
levels = ["tier0_typed_delivery"]
seed = 0
device = "cpu"

[run.model]
d_model = 64
nhead = 2
encoder_layers = 1
decoder_layers = 1
dim_feedforward = 128

[[stage]]
name = "rl-a"
kind = "rl"
iterations = 1
episodes = 2
""",
    )
    pipeline = load_pipeline(path)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(oms,))]

    run_dir = run_pipeline(pipeline, entries=entries)

    payload = torch.load(run_dir / "stage_rl-a.pt", weights_only=False)
    stored = payload["model_config"]
    stored = asdict(stored) if isinstance(stored, ModelConfig) else stored
    assert stored["d_model"] == 64
    assert stored["nhead"] == 2


def test_model_override_still_rejects_mismatched_warm_start(tmp_path):
    """A bigger-model run cannot warm-start from a size-128 checkpoint."""
    checkpoint = _warm_start_checkpoint(tmp_path)  # size-128 weights
    oms = tmp_path / "tier0_test.oms"
    oms.write_text(TIER0_OMS.read_text())
    path = _write_pipeline(
        tmp_path,
        f"""
[run]
levels = ["tier0_typed_delivery"]
seed = 0
device = "cpu"
init_from = "{checkpoint}"

[run.model]
d_model = 64
nhead = 2

[[stage]]
name = "rl-a"
kind = "rl"
iterations = 1
episodes = 2
""",
    )
    pipeline = load_pipeline(path)
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(oms,))]

    with pytest.raises(ValueError, match="model config"):
        run_pipeline(pipeline, entries=entries)
