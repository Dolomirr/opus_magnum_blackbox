"""
Pipeline runner: one TOML file managing pretrain and RL stages.

Config schema:

- `[run]`: `levels` (corpus names or directory groups), optional `eval_levels`
    (bank for greedy eval), `seed`, `device`,
    `autodiscovery` (int 1..10, default 1;
        scales every pretrain stage's epochs rounded to provided fraction.
        k=10 zeroes them and the stage is skipped).
- `[[stage]]`: `name`, `kind` (`pretrain` or `rl`),
    plus keys routed by field name into `PretrainConfig`/`RewardConfig` (pretrain)
    or `TrainConfig`/`RewardConfig` (rl).
    Unknown keys are a hard error naming the stage;
    `episodes` is alias for `episodes_per_iteration`;
    `eval_patience` is an rl-only stage-level key (early stop on eval plateau).

Stages execute in order against one `Trainer` (one run dir).
Each rl stage's `iterations` is absolute target over the global iteration counter,
and every stage writes `stage_<name>.pt` with the usual checkpoint payload.

`run_pipeline(..., resume_dir=<run dir>)` continues an existing run:
the latest `stage_<name>.pt` matching config stage is restored,
that stage and all earlier ones are skipped,
and remaining stages run against the restored trainer, writing into the same run dir.
"""

# here we go again...

import shutil
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
from opus_magnum_sym import Level, LevelSpec

from opus_magnum_blackbox.config import ModelConfig, PretrainConfig, RewardConfig, TrainConfig
from opus_magnum_blackbox.corpus import resolve_entries
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.profile import ProfiledEnv, ProfiledEnvs
from opus_magnum_blackbox.rollout import RolloutBatch, build_supervised
from opus_magnum_blackbox.solutions import augment, load_solutions
from opus_magnum_blackbox.train import Trainer

# TOML stage keys that differ from the config field names they set.
_ALIASES = {"episodes": "episodes_per_iteration"}


@dataclass(frozen=True)
class PretrainStage:
    name: str
    config: PretrainConfig


@dataclass(frozen=True)
class RLStage:
    name: str
    iterations: int  # stage length; absolutized over the global counter at execution
    train_config: TrainConfig  # run-level values (seed, device) overlaid with the routed keys
    reward_config: RewardConfig
    eval_patience: int | None = None  # end the stage after this many evals without eval-rate improvement


@dataclass(frozen=True)
class PipelineConfig:
    levels: tuple[str, ...]
    train_config: TrainConfig
    stages: tuple[PretrainStage | RLStage, ...]
    profile: bool = False
    eval_levels: tuple[str, ...] = ()
    autodiscovery: int = 1  # supervision switch: 1 keeps full BC, 10 removes it


def load_pipeline(path: str | Path) -> PipelineConfig:
    """
    Parse and validate a pipeline TOML file.
    """
    with Path(path).open("rb") as file:
        data = tomllib.load(file)

    unknown_tables = set(data) - {"run", "stage", "profile"}
    if unknown_tables:
        raise ValueError(f"unknown config tables: {sorted(unknown_tables)}; expected [run], [profile] and [[stage]]")

    run = data.get("run")
    if not isinstance(run, dict) or not run.get("levels"):
        raise ValueError("config must define a [run] table with a non-empty `levels` list")
    resolve_entries(list(run["levels"]))  # corpus validation
    eval_levels = run.get("eval_levels", [])
    if eval_levels:
        resolve_entries(list(eval_levels))  # corpus validation
    unknown_run_keys = set(run) - {"levels", "eval_levels", "seed", "device", "autodiscovery"}
    if unknown_run_keys:
        raise ValueError(
            f"unknown [run] keys: {sorted(unknown_run_keys)}; known keys: "
            + "levels, eval_levels, seed, device, autodiscovery"
        )
    autodiscovery = run.get("autodiscovery", 1)
    # bro, why toml bool is python's int?
    if (
        isinstance(autodiscovery, bool)
        or not isinstance(autodiscovery, int)
        or not 1 <= autodiscovery <= 10
    ): # fmt: skip
        raise ValueError(f"[run] `autodiscovery` must be an integer in 1..10, got {autodiscovery!r}")

    profile = data.get("profile", {})
    unknown_profile_keys = set(profile) - {"enabled"}
    if unknown_profile_keys:
        raise ValueError(f"unknown [profile] keys: {sorted(unknown_profile_keys)}; known keys: enabled")

    train_config = replace(TrainConfig(), seed=run.get("seed", 0), device=run.get("device", ""))

    stages = data.get("stage")
    if not stages:
        raise ValueError("config must define at least one [[stage]]")
    parsed = []
    for raw in stages:
        stage = _parse_stage(raw, train_config)
        if isinstance(stage, PretrainStage):
            # supervision scaled at load time, so stage configs still carry epochs
            epochs = round(stage.config.epochs * (10 - autodiscovery) / 9)
            stage = replace(stage, config=replace(stage.config, epochs=epochs))
        parsed.append(stage)
    return PipelineConfig(
        levels=tuple(run["levels"]),
        train_config=train_config,
        stages=tuple(parsed),
        profile=bool(profile.get("enabled", False)),
        eval_levels=tuple(eval_levels),
        autodiscovery=autodiscovery,
    )


def _parse_stage(raw: dict, run_config: TrainConfig) -> PretrainStage | RLStage:
    name = raw.get("name")
    kind = raw.get("kind")
    if not isinstance(name, str) or not name:
        raise ValueError(f"every [[stage]] needs a non-empty string `name`; got {name!r}")
    if kind not in ("pretrain", "rl"):
        raise ValueError(f"stage {name!r}: `kind` must be 'pretrain' or 'rl', got {kind!r}")

    keys = {key: value for key, value in raw.items() if key not in ("name", "kind")}
    # RewardConfig first: `gamma` names the cycles bonus here, not the TrainConfig discount.
    if kind == "pretrain":
        routed = _route_keys(keys, (PretrainConfig, RewardConfig), name)
        config = replace(PretrainConfig(), **routed.get(PretrainConfig, {}))
        return PretrainStage(name=name, config=config)
    eval_patience = _parse_eval_patience(keys, name)
    routed = _route_keys(keys, (RewardConfig, TrainConfig), name)
    train_fields = routed.get(TrainConfig, {})
    if "iterations" not in train_fields:
        raise ValueError(f"rl stage {name!r}: missing required key `iterations`")
    train_config = replace(run_config, **train_fields)
    reward_config = replace(RewardConfig(), **routed.get(RewardConfig, {}))
    return RLStage(
        name=name,
        iterations=train_fields["iterations"],
        train_config=train_config,
        reward_config=reward_config,
        eval_patience=eval_patience,
    )


def _parse_eval_patience(raw: dict, stage_name: str) -> int | None:
    """
    Stage-level rl key: evals without eval-rate improvement before the stage ends early.
    """
    value = raw.pop("eval_patience", None)
    if value is None:
        return None
    # TOML booleans arrive as Python ints' subclasses; reject them explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"stage {stage_name!r}: `eval_patience` must be a positive integer, got {value!r}")
    return value


def _route_keys(keys: dict, targets: tuple[type, ...], stage_name: str) -> dict[type, dict]:
    """
    Split stage keys by target config field name; unknown key is a hard error.
    """
    routed: dict[type, dict] = {}
    for key, value in keys.items():
        field = _ALIASES.get(key, key)
        for target in targets:
            if any(item.name == field for item in fields(target)):
                routed.setdefault(target, {})[field] = value
                break
        else:
            known = sorted({item.name for target in targets for item in fields(target)} | set(_ALIASES))
            raise ValueError(f"stage {stage_name!r}: unknown key {key!r}; known keys: {known}")
    return routed


def run_pipeline(
    config: PipelineConfig,
    entries: list[LevelEntry] | None = None,
    eval_entries: list[LevelEntry] | None = None,
    resume_dir: str | Path | None = None,
) -> Path:
    """
    Execute all stages in order against one Trainer; returns the run directory.

    `entries`/`eval_entries` override corpus resolution.
    `resume_dir` continues an existing run (see module docstring).
    `config.profile` swaps in `ProfiledEnv`; RL stages then also write
    `profile.jsonl` and the CLI prints the attribution summary afterwards.
    Train entries share one env (per-reset weighted sampling);
    eval bank, gets its own env.
    """
    entries = entries if entries is not None else resolve_entries(list(config.levels))
    if eval_entries is None:
        eval_entries = resolve_entries(list(config.eval_levels)) if config.eval_levels else []
    env_cls = ProfiledEnv if config.profile else OpusEnv
    envs: list[OpusEnv] = [env_cls(entries)]
    eval_envs: list[OpusEnv] = [env_cls(eval_entries)] if eval_entries else []
    profiled = [env for env in (*envs, *eval_envs) if isinstance(env, ProfiledEnv)]
    if resume_dir is None:
        torch.manual_seed(config.train_config.seed)
        trainer = Trainer(
            OpusPolicy(envs[0].tokenizer.config, ModelConfig()),
            envs,
            config.train_config,
            eval_envs=eval_envs,
            autodiscovery=config.autodiscovery,
        )
        trainer._seed()
        start = 0
    else:
        trainer, start = _resume_trainer(config, envs, eval_envs, Path(resume_dir))
        # Resumes restore the checkpoint's stored config; the new run-level one wins.
        trainer.train_config = replace(
            trainer.train_config,
            run_dir=str(trainer.run_dir),
            seed=config.train_config.seed,
            device=config.train_config.device,
        )
        trainer.autodiscovery = config.autodiscovery
    trainer.profile = ProfiledEnvs(profiled) if profiled else None

    for stage in config.stages[start:]:
        if isinstance(stage, PretrainStage):
            if stage.config.epochs == 0:
                # Dial-scaled to zero: skip BC entirely, but still checkpoint so resume sees it done.
                trainer._log_stage_skip(stage.name, f"autodiscovery={config.autodiscovery} scales bc epochs to 0")
            else:
                _run_pretrain_stage(trainer, entries, stage, seed=config.train_config.seed)
        else:
            _run_rl_stage(trainer, envs, stage)
        latest = trainer.save_checkpoint()
        shutil.copyfile(latest, trainer.run_dir / f"stage_{stage.name}.pt")
    return trainer.run_dir


def _resume_trainer(
    config: PipelineConfig,
    envs: list[OpusEnv],
    eval_envs: list[OpusEnv],
    resume_dir: Path,
) -> tuple[Trainer, int]:
    """
    Trainer restored from latest stage checkpoint in `resume_dir`,
    as well as index of the first stage that has not completed yet.
    """
    start = max(
        (index for index, stage in enumerate(config.stages) if (resume_dir / f"stage_{stage.name}.pt").exists()),
        default=-1,
    )
    if start < 0:
        expected = ", ".join(f"stage_{stage.name}.pt" for stage in config.stages)
        raise ValueError(f"resume: no stage checkpoint found in {resume_dir}; expected one of: {expected}")
    if start == len(config.stages) - 1:
        raise ValueError(f"resume: every stage of the config already completed in {resume_dir}")

    checkpoint = resume_dir / f"stage_{config.stages[start].name}.pt"
    trainer = Trainer.resume(checkpoint, envs, eval_envs=eval_envs)
    # Current config wins for run-level values: writes land in the resumed dir,
    # seed/device may have changed since the checkpoint.
    trainer.run_dir = resume_dir
    return trainer, start + 1


def _run_pretrain_stage(trainer: Trainer, entries: list[LevelEntry], stage: PretrainStage, seed: int) -> None:
    tokenizer, masker = trainer.envs[0].tokenizer, trainer.envs[0].masker
    batches: list[RolloutBatch] = []
    variant_seed = seed
    for entry in entries:
        spec = _entry_spec(entry)
        variants: list[list[int]] = []
        for sequence in load_solutions(entry):
            variants.append(sequence)
            if stage.config.augment:
                variants.extend(augment(sequence, spec, seed=variant_seed, max_variants=stage.config.max_variants))
            variant_seed += 1
        if variants:
            batches.append(build_supervised(variants, spec, tokenizer, masker))
    if not batches:
        raise ValueError(f"pretrain stage {stage.name!r}: no solution sequences found on the configured levels")
    trainer.pretrain(batches, stage.config, stage=stage.name)


def _run_rl_stage(trainer: Trainer, envs: list[OpusEnv], stage: RLStage) -> None:
    trainer.stage = stage.name
    for env in envs:
        env.reward_config = stage.reward_config
    trainer.reward_config = stage.reward_config
    trainer.train_config = replace(stage.train_config, iterations=trainer.iteration + stage.iterations)
    for group in trainer.optimizer.param_groups:
        group["lr"] = trainer.train_config.lr
    trainer.run(eval_patience=stage.eval_patience)


def _entry_spec(entry: LevelEntry) -> LevelSpec:
    level = Level.from_file(entry.source) if isinstance(entry.source, Path) else Level(entry.source)
    return level.spec
