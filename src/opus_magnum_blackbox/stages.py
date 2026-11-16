"""
Pipeline runner: one TOML file managing pretrain and RL stages.

Config schema:

- `[run]`: `levels` (corpus names or directory groups), optional `eval_levels`
    (bank for greedy eval), `seed`, `device`,
    `autodiscovery` (int 1..10, default 1;
        scales every pretrain stage's epochs rounded to provided fraction.
        k=10 zeroes them and the stage is skipped),
    `tag` (optional run-dir suffix override; default derives the tag from the
        config plus RL config)
        defaults, e.g. `tier1_scarcity_t2-ad9-sil4-scarcity-t2`),
    `init_from` (optional checkpoint path; a fresh run loads the checkpoint's
        model weights before stage 1 - fresh optimizer, iteration counter and
        run dir; model-config mismatch is a hard error. Ignored on resume).
    `encoder_features` (optional list of derived relational encoder feature
        names; empty = off, byte-identical observations and model shapes).
        Threaded env->observation->model: the env appends the per-spec feature
        vector to observations, the policy projects it into the encoder.
- `[[stage]]`: `name`, `kind` (`pretrain` or `rl`),
    plus keys routed by field name into `PretrainConfig`/`RewardConfig` (pretrain)
    or `TrainConfig`/`RewardConfig` (rl).
    Unknown keys are a hard error naming the stage;
    `episodes` is alias for `episodes_per_iteration`;
    `eval_patience` is an rl-only stage-level key (early stop on eval plateau);
    `skip_when_bc_eval_gte` is an rl-only stage-level key (skip the stage when
    the trainer's most recent bc-end held-out eval rate meets the threshold;
    the `stage_skip` metrics row names the rate).

Stages execute in order against one `Trainer` (one run dir).
Each rl stage's `iterations` is absolute target over the global iteration counter,
and every stage writes `stage_<name>.pt` with the usual checkpoint payload.

`run_pipeline(..., resume_dir=<run dir>)` continues an existing run:
the latest `stage_<name>.pt` matching config stage is restored,
that stage and all earlier ones are skipped,
and remaining stages run against the restored trainer, writing into the same run dir.
The resolved TOML is copied into the run dir as `config.toml`.
"""

# here we go again...

import shutil
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import torch
from opus_magnum_sym import Level, LevelSpec

from opus_magnum_blackbox.config import ModelConfig, PretrainConfig, RewardConfig, TrainConfig, encoder_feature_dim
from opus_magnum_blackbox.corpus import resolve_entries
from opus_magnum_blackbox.env import LevelEntry, OpusEnv, spec_encoder_features
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.profile import ProfiledEnv, ProfiledEnvs
from opus_magnum_blackbox.rollout import RolloutBatch, build_supervised
from opus_magnum_blackbox.solutions import augment, load_solutions
from opus_magnum_blackbox.train import Trainer

# TOML stage keys that differ from the config field names they set.
_ALIASES = {"episodes": "episodes_per_iteration"}

# Run-bookkeeping fields: never part of the derived knob tag.
_TAG_SKIP = {"iterations", "seed", "device", "run_dir"}

# "smort"
_TAG_ABBREV = {
    ("TrainConfig", "lr"): "lr",
    ("TrainConfig", "episodes_per_iteration"): "ep",
    ("TrainConfig", "entropy_coef"): "ent",
    ("TrainConfig", "explore_epsilon"): "eps",
    ("TrainConfig", "sample_temperature"): "t",
    ("TrainConfig", "sil_coef"): "sil",
    ("TrainConfig", "sil_batch_size"): "silb",
    ("TrainConfig", "sil_buffer_cap"): "silcap",
    ("TrainConfig", "sil_weighting"): "silw",
    ("TrainConfig", "sil_label"): "sill",
    ("TrainConfig", "value_coef"): "vl",
    ("TrainConfig", "gamma"): "ga",
    ("TrainConfig", "grad_clip"): "gc",
    ("TrainConfig", "eval_interval"): "ei",
    ("TrainConfig", "checkpoint_interval"): "ci",
    ("RewardConfig", "solve_reward"): "sr",
    ("RewardConfig", "failure_penalty"): "fp",
    ("RewardConfig", "error_base"): "eb",
    ("RewardConfig", "error_progress"): "eprog",
    ("RewardConfig", "unsolved_reward"): "ur",
    ("RewardConfig", "alpha"): "a",
    ("RewardConfig", "beta"): "b",
    ("RewardConfig", "gamma"): "cy",
    ("RewardConfig", "shaping_weight"): "sh",
    ("RewardConfig", "progress_weight"): "pg",
    ("RewardConfig", "placement_reward"): "plc",
    ("RewardConfig", "invalid_action_penalty"): "invp",
    ("RewardConfig", "cost_max"): "cost",
}


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
    skip_when_bc_eval_gte: float | None = None  # skip entirely when the bc-end held-out eval rate meets this


@dataclass(frozen=True)
class PipelineConfig:
    levels: tuple[str, ...]
    train_config: TrainConfig
    stages: tuple[PretrainStage | RLStage, ...]
    profile: bool = False
    eval_levels: tuple[str, ...] = ()
    autodiscovery: int = 1  # supervision switch: 1 keeps full BC, 10 removes it
    tag: str = ""  # run-dir suffix; empty derives from the source stem + non-default knobs
    init_from: Path | None = None  # warm-start checkpoint; model weights only
    encoder_features: tuple[str, ...] = ()  # derived relational encoder side channel; empty = off
    model_config: ModelConfig = field(
        default_factory=ModelConfig
    )  # `[run] model` overrides; default is the size-128 model
    source: Path | None = None  # resolved TOML path; copied into the run dir as config.toml


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
    unknown_run_keys = set(run) - {
        "levels",
        "eval_levels",
        "seed",
        "device",
        "autodiscovery",
        "tag",
        "init_from",
        "model",
        "encoder_features",
    }
    if unknown_run_keys:
        raise ValueError(
            f"unknown [run] keys: {sorted(unknown_run_keys)}; known keys: "
            + "levels, eval_levels, seed, device, autodiscovery, tag, init_from, model, encoder_features"
        )
    raw_features = run.get("encoder_features", [])
    if not isinstance(raw_features, list) or any(not isinstance(name, str) for name in raw_features):
        raise ValueError(f"[run] `encoder_features` must be a list of feature names, got {raw_features!r}")
    encoder_features = tuple(raw_features)
    try:
        encoder_feature_dim(encoder_features)  # knob validation
    except ValueError as error:
        raise ValueError(f"[run] {error}") from error
    model_config = _parse_model_config(run.get("model", {}))
    init_from = run.get("init_from")
    if init_from is not None and (not isinstance(init_from, str) or not init_from):
        raise ValueError(f"[run] `init_from` must be a non-empty checkpoint path string, got {init_from!r}")
    autodiscovery = run.get("autodiscovery", 1)
    # bro, why toml bool is python's int?
    if (
        isinstance(autodiscovery, bool)
        or not isinstance(autodiscovery, int)
        or not 1 <= autodiscovery <= 10
    ):  # fmt: skip
        raise ValueError(f"[run] `autodiscovery` must be an integer in 1..10, got {autodiscovery!r}")

    tag = run.get("tag")
    if tag is not None and (not isinstance(tag, str) or not tag or "/" in tag or "\\" in tag):
        # the tag becomes one run-dir path segment; keep it a single safe component
        raise ValueError(f"[run] `tag` must be a non-empty string without path separators, got {tag!r}")

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
        tag=tag if tag is not None else _derive_tag(Path(path), tuple(parsed), autodiscovery, model_config),
        init_from=Path(init_from) if init_from else None,
        encoder_features=encoder_features,
        model_config=model_config,
        source=Path(path),
    )


def _parse_model_config(raw: dict) -> ModelConfig:
    """
    `[run] model` table: `ModelConfig` field overrides; unknown key is a hard error.
    An absent table keeps the default (size-128) model exactly.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"[run] `model` must be a table, got {raw!r}")  # noqa: TRY004 (config validation)
    known = {item.name: item.type for item in fields(ModelConfig)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ValueError(f"unknown [run.model] keys: {unknown}; known keys: {sorted(known)}")
    for key, value in raw.items():
        expected = known[key]
        if isinstance(value, bool) or not isinstance(value, int if expected is int else float):
            raise ValueError(f"[run.model] `{key}` must be {expected.__name__}, got {value!r}")  # noqa: TRY004
    return replace(ModelConfig(), **raw)


def _derive_tag(
    source: Path,
    stages: tuple[PretrainStage | RLStage, ...],
    autodiscovery: int,
    model_config: ModelConfig,
) -> str:
    """
    Run-dir suffix: config stem + autodiscovery dial + RL knobs that differ from
    `TrainConfig`/`RewardConfig` defaults, deduped in stage order. Config stems are
    unique per config file, so two configs can never produce indistinguishable run dirs.
    """
    tokens = [source.stem]
    if autodiscovery > 1:
        tokens.append(f"ad{autodiscovery}")
    # non-default model shapes: distinguish capacity arms from the size-128 runs
    if model_config != ModelConfig():
        tokens.append(f"m{model_config.d_model}")
    seen = set(tokens)
    for stage in stages:
        if not isinstance(stage, RLStage):
            continue
        for cls, config in ((TrainConfig, stage.train_config), (RewardConfig, stage.reward_config)):
            for item in fields(cls):
                if item.name in _TAG_SKIP:
                    continue
                value = getattr(config, item.name)
                if value == item.default:
                    continue
                token = _tag_token(cls.__name__, item.name, value)
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
    return "-".join(tokens)


def _tag_token(cls_name: str, field_name: str, value: object) -> str:
    name = _TAG_ABBREV.get((cls_name, field_name), field_name.replace("_", ""))
    if isinstance(value, bool):
        return name  # non-default bool: presence is the token
    if isinstance(value, float):
        text = f"{value:g}"
        # 0.4 -> "4", 2.0 -> "2", 1.5 -> "15", 3e-4 -> "0003"
        return name + (text[2:] if text.startswith("0.") else text.replace(".", ""))
    return f"{name}{value}"


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
    skip_when_bc_eval_gte = _parse_skip_when_bc_eval_gte(keys, name)
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
        skip_when_bc_eval_gte=skip_when_bc_eval_gte,
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


def _parse_skip_when_bc_eval_gte(raw: dict, stage_name: str) -> float | None:
    """
    Stage-level rl key: skip the stage when the bc-end held-out eval rate meets this.
    """
    value = raw.pop("skip_when_bc_eval_gte", None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"stage {stage_name!r}: `skip_when_bc_eval_gte` must be a number in [0, 1], got {value!r}")
    return float(value)


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
    envs: list[OpusEnv] = [env_cls(entries, encoder_features=config.encoder_features)]
    eval_envs: list[OpusEnv] = (
        [env_cls(eval_entries, encoder_features=config.encoder_features)] if eval_entries else []
    )
    profiled = [env for env in (*envs, *eval_envs) if isinstance(env, ProfiledEnv)]
    if resume_dir is None:
        torch.manual_seed(config.train_config.seed)
        trainer = Trainer(
            OpusPolicy(envs[0].tokenizer.config, config.model_config, encoder_features=config.encoder_features),
            envs,
            config.train_config,
            eval_envs=eval_envs,
            autodiscovery=config.autodiscovery,
            tag=config.tag,
        )
        trainer._seed()
        if config.init_from is not None:
            trainer.warm_start(config.init_from)
            trainer._log_warm_start(str(config.init_from))
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
    if config.source is not None:
        trainer.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.source, trainer.run_dir / "config.toml")  # provenance

    for stage in config.stages[start:]:
        if isinstance(stage, PretrainStage):
            if stage.config.epochs == 0:
                # Dial-scaled to zero: skip BC entirely, but still checkpoint so resume sees it done.
                trainer._log_stage_skip(stage.name, f"autodiscovery={config.autodiscovery} scales bc epochs to 0")
            else:
                _run_pretrain_stage(trainer, entries, stage, seed=config.train_config.seed)
        else:
            threshold = stage.skip_when_bc_eval_gte
            bc_rate = trainer._last_bc_eval_rate
            if threshold is not None and bc_rate is not None and bc_rate >= threshold:
                # Cloning already at ceiling: RL can only risk it. Skip, but still checkpoint
                # so resume sees the stage done.
                trainer._log_stage_skip(
                    stage.name,
                    f"bc-end eval {bc_rate:.3f} meets skip_when_bc_eval_gte {threshold:g}",
                    eval_solve_rate=bc_rate,
                )
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
    features = trainer.envs[0].encoder_features
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
            side_channel = spec_encoder_features(spec, features) if features else None
            batches.append(build_supervised(variants, spec, tokenizer, masker, encoder_features=side_channel))
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
