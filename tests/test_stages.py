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
