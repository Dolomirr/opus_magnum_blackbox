"""
Pipeline runner: one TOML file managing pretrain and RL stages.

Config schema:

- `[run]`: `levels` (corpus names), `seed`, `device`.
- `[[stage]]`: `name`, `kind` (`pretrain` or `rl`),
    plus keys routed by field name into `PretrainConfig`/`RewardConfig` (pretrain)
    or `TrainConfig`/`RewardConfig` (rl).
    Unknown keys are a hard error naming the stage;
    `episodes` is alias for `episodes_per_iteration`.

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


@dataclass(frozen=True)
class PipelineConfig:
    levels: tuple[str, ...]
    train_config: TrainConfig
    stages: tuple[PretrainStage | RLStage, ...]
    profile: bool = False


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
    resolve_entries(run["levels"])  # corpus validation
    unknown_run_keys = set(run) - {"levels", "seed", "device"}
    if unknown_run_keys:
        raise ValueError(f"unknown [run] keys: {sorted(unknown_run_keys)}; known keys: levels, seed, device")

    profile = data.get("profile", {})
    unknown_profile_keys = set(profile) - {"enabled"}
    if unknown_profile_keys:
        raise ValueError(f"unknown [profile] keys: {sorted(unknown_profile_keys)}; known keys: enabled")

    train_config = replace(TrainConfig(), seed=run.get("seed", 0), device=run.get("device", ""))

    stages = data.get("stage")
    if not stages:
        raise ValueError("config must define at least one [[stage]]")
    parsed = tuple(_parse_stage(raw, train_config) for raw in stages)
    return PipelineConfig(
        levels=tuple(run["levels"]),
        train_config=train_config,
        stages=parsed,
        profile=bool(profile.get("enabled", False)),
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
    routed = _route_keys(keys, (RewardConfig, TrainConfig), name)
    train_fields = routed.get(TrainConfig, {})
    if "iterations" not in train_fields:
        raise ValueError(f"rl stage {name!r}: missing required key `iterations`")
    train_config = replace(run_config, **train_fields)
    reward_config = replace(RewardConfig(), **routed.get(RewardConfig, {}))
    return RLStage(
        name=name, iterations=train_fields["iterations"], train_config=train_config, reward_config=reward_config
    )


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
    resume_dir: str | Path | None = None,
) -> Path:
    """
    Execute all stages in order against one Trainer; returns the run directory.

    `entries` overrides corpus resolution (tests pass in-memory specs).
    `resume_dir` continues an existing run (see module docstring).
    `config.profile` swaps in `ProfiledEnv`; RL stages then also write
    `profile.jsonl` and the CLI prints the attribution summary afterwards.
    """
    entries = entries if entries is not None else resolve_entries(list(config.levels))
    if config.profile:
        profiled: list[ProfiledEnv] = [ProfiledEnv([entry]) for entry in entries]
        envs: list[OpusEnv] = list(profiled)
    else:
        profiled = []
        envs = [OpusEnv([entry]) for entry in entries]
    if resume_dir is None:
        torch.manual_seed(config.train_config.seed)
        trainer = Trainer(OpusPolicy(envs[0].tokenizer.config, ModelConfig()), envs, config.train_config)
        trainer._seed()
        start = 0
    else:
        trainer, start = _resume_trainer(config, envs, Path(resume_dir))
        # Resumes restore the checkpoint's stored config; the new run-level one wins.
        trainer.train_config = replace(
            trainer.train_config,
            run_dir=str(trainer.run_dir),
            seed=config.train_config.seed,
            device=config.train_config.device,
        )
    trainer.profile = ProfiledEnvs(profiled) if profiled else None

    for stage in config.stages[start:]:
        if isinstance(stage, PretrainStage):
            _run_pretrain_stage(trainer, entries, stage, seed=config.train_config.seed)
        else:
            _run_rl_stage(trainer, envs, stage)
        latest = trainer.save_checkpoint()
        shutil.copyfile(latest, trainer.run_dir / f"stage_{stage.name}.pt")
    return trainer.run_dir


def _resume_trainer(config: PipelineConfig, envs: list[OpusEnv], resume_dir: Path) -> tuple[Trainer, int]:
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
    trainer = Trainer.resume(checkpoint, envs)
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
    trainer.run()


def _entry_spec(entry: LevelEntry) -> LevelSpec:
    level = Level.from_file(entry.source) if isinstance(entry.source, Path) else Level(entry.source)
    return level.spec
