"""
Tier-3 placed bank generator: 16-config placed family as variant files.

Every variant is a `vNNN.oml` spec plus its same-stem reference `vNNN.oms`,
emitted from the placed family generator and re-validated from disk after
writing. The config space is fully enumerated (4 elementals x target 1-2 x
two allow-lists); the seeded split cuts one eval config per atom stratum
(75/25: 12 train / 4 eval), so every axis value keeps train coverage - each
stratum holds two configs per (target, allow-list) value, and dropping one
always leaves both.

Unfixed entrypoints are the family point: `.oml` variants carry
`entrypoints_fixed 0` and position-less `entrance`/`exit` lines; the
reference `.oms` places both entrypoints before the arm. The manifest's
per-config provenance records the canonical plan (entrance/exit cells, arm
placement) the family solver chose for each config.

Same parameters yield the same bank; regeneration overwrites in place and
removes stale variant files.
"""

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from opus_magnum_sym import AtomType, Level, LevelSpec

from opus_magnum_blackbox.corpus import LEVELS_DIR
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.families import (
    BoardSize,
    FamilyError,
    PlacedConfig,
    PlacedPlan,
    placed_configs,
    placed_reference_oms,
    placed_reference_tokens,
    placed_spec_plan,
)
from opus_magnum_blackbox.solutions import load_solutions

FAMILY_NAME = "tier3_placed"
BANK_DIR = LEVELS_DIR / FAMILY_NAME

# reward normalization default recorded for the corpus scanner
MANIFEST_COST_MAX = 50
MANIFEST_WEIGHT = 1.0

# default split seed; the procedural tier3_placed pool rebuilds this exact split
BANK_SEED = 0

TRAIN_COUNT = 12
EVAL_COUNT = 4

type ConfigKey = tuple[str, int, bool] # (atom name, target, long_arms) identity of one config


@dataclass(frozen=True)
class BankParams:
    """
    Bank parameters shared by the split, emission and manifest provenance.
    """

    seed: int = BANK_SEED
    steps: int = 30
    size: BoardSize = (6, 6)


@dataclass(frozen=True)
class Variant:
    """
    One emitted level/solution pair.
    """

    oml: Path
    oms: Path
    config: PlacedConfig
    spec: LevelSpec
    plan: PlacedPlan  # the canonical plan the family solver chose for the config

    @property
    def key(self) -> ConfigKey:
        return (self.config.atom.name, self.config.target, self.config.long_arms)


@dataclass(frozen=True)
class Bank:
    """
    Emitted bank: variant lists per split and manifest path.
    """

    root: Path
    train: list[Variant]
    eval: list[Variant]
    manifest: Path

    @property
    def variants(self) -> list[Variant]:
        return [*self.train, *self.eval]


def split_configs(*, seed: int = BANK_SEED) -> tuple[tuple[PlacedConfig, ...], tuple[PlacedConfig, ...]]:
    """
    Stratified train/eval cut of all 16 configs: one seeded-shuffled eval pick
    per atom stratum (4 configs each), 12 train / 4 eval overall. Deterministic
    for a given seed; the procedural `tier3_placed` pool rebuilds the train side.
    """
    rng = random.Random(seed)
    train: list[PlacedConfig] = []
    eval_: list[PlacedConfig] = []
    seen: list[AtomType] = []
    for config in placed_configs():
        if config.atom not in seen:
            seen.append(config.atom)
    for atom in seen:
        stratum = [config for config in placed_configs() if config.atom == atom]
        rng.shuffle(stratum)
        eval_.append(stratum[0])
        train.extend(stratum[1:])
    return tuple(train), tuple(eval_)


def generate_bank(out_dir: Path = BANK_DIR, *, params: BankParams | None = None) -> Bank:
    """
    Emit bank as files and validate every pair from disk.
    """
    params = params or BankParams()
    train_configs, eval_configs = split_configs(seed=params.seed)
    train = [_emit_variant(config, index, out_dir / "train", params=params) for index, config in enumerate(train_configs)]
    eval_ = [_emit_variant(config, index, out_dir / "eval", params=params) for index, config in enumerate(eval_configs)]
    _clean_stale(out_dir / "train", [path for variant in train for path in (variant.oml, variant.oms)])
    _clean_stale(out_dir / "eval", [path for variant in eval_ for path in (variant.oml, variant.oms)])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.toml"
    manifest.write_text(_manifest_text(params, train, eval_))
    # self-ignore
    (out_dir / ".gitignore").write_text("# Generated bank; reproduce with `just gen-tier3`.\n*\n")
    return Bank(out_dir, train, eval_, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the tier-3 placed-entrypoint level bank (deterministic, idempotent)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BankParams.seed,
        help="config split seed",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=BankParams.steps,
        help="sim step budget per variant",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="bank root directory (default: the tier3_placed bank)",
    )
    args = parser.parse_args(argv)
    params = BankParams(seed=args.seed, steps=args.steps)
    bank = generate_bank(args.out_dir or BANK_DIR, params=params)
    print(f"wrote {len(bank.train)} train + {len(bank.eval)} eval variants to {bank.root}")
    for split, variants in (("train", bank.train), ("eval", bank.eval)):
        atoms = sorted({variant.config.atom.name.lower() for variant in variants})
        targets = sorted({variant.config.target for variant in variants})
        long_n = sum(1 for variant in variants if variant.config.long_arms)
        print(f"  {split}: atoms {atoms}, targets {targets}, long-arm configs {long_n}")
    return 0


def _emit_variant(config: PlacedConfig, index: int, out_dir: Path, *, params: BankParams) -> Variant:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, plan = placed_spec_plan(config, steps=params.steps, size=params.size)
    variant = Variant(out_dir / f"v{index:03d}.oml", out_dir / f"v{index:03d}.oms", config, spec, plan)
    variant.oml.write_text(_oml_text(spec, out_dir.name, index))
    variant.oms.write_text(placed_reference_oms(spec))
    _check_variant(variant)
    return variant


def _oml_text(spec: LevelSpec, split: str, index: int) -> str:
    rows, cols = spec.size
    lines = [
        "# DO NOT EDIT",
        "# Generated by `gen-tier3`; regenerate instead of editing.",
        "",
        f"# Tier-3 placed family, {split} variant {index:03d}.",
        "#",
        "# Solution-placed 1-cell typed entrance and exit;",
        "# reference solution places both, then delivers with one static arm.",
        "",
        "",
        "",
        f"size {rows} {cols}",
        f"steps {spec.steps}",
        "entrypoints_fixed 0",
        "",
        f"target {spec.target}",
        "",
        "# 1-cell typed shapes, no links; entrypoints placed by the solution.",
    ]
    for shape in spec.shapes:
        cell = shape.cells[0]
        type_suffix = f":{cell.atom_type.name.lower()}" if cell.atom_type is not None else ""
        lines.append(
            f"shape {shape.name} {shape.comp_type.name.lower()} cells=({cell.offset.r},{cell.offset.s}){type_suffix} links="
        )
    lines.append("")
    for ep in spec.entrypoints:
        if ep.pos is not None:
            raise FamilyError(f"bank variants need solution-placed entrypoints; {ep.shape} carries a position")
        lines.append(f"{ep.kind.name.lower()} {ep.shape}")
    lines += [
        "",
        "# Blueprints available to solution.",
        f"allow {' '.join(sorted(spec.allowed_blueprints))}",
        "",
    ]
    return "\n".join(lines)


def _check_variant(variant: Variant) -> None:
    """
    Re-load the emitted files from disk and fail loudly on any mismatch: the
    level must equal the family spec, and `load_solutions` decodes, mask-walks,
    applies and solves the pair.
    """
    disk_spec = Level.from_file(variant.oml).spec
    if disk_spec != variant.spec:
        raise FamilyError(f"{variant.oml.name}: emitted spec differs from the family spec")
    entry = LevelEntry(source=variant.oml, cost_max=MANIFEST_COST_MAX, name=variant.oml.stem, solutions=(variant.oms,))
    tokens = load_solutions(entry)[0]
    if tokens != placed_reference_tokens(variant.spec):
        raise FamilyError(f"{variant.oms.name}: emitted solution differs from the reference sequence for {variant.oml.name}")


def _clean_stale(directory: Path, keep: list[Path]) -> None:
    """
    Remove variant files left over from an earlier generation with other counts,
    so the bank stays a pure function of its parameters.
    """
    if not directory.is_dir():
        return
    kept = {path.name for path in keep}
    for path in directory.glob("v*"):
        if path.name not in kept:
            path.unlink()


def _manifest_text(params: BankParams, train: list[Variant], eval_: list[Variant]) -> str:
    """
    Per-family manifest consumed by the corpus scanner: cost/weight defaults plus
    provenance for regeneration - seed, counts, and per-config canonical plans.
    """
    lines = [
        "# DO NOT EDIT",
        "",
        f"# {FAMILY_NAME} bank manifest; consumed by corpus scanner.",
        f"# Regenerate bank with `gen-tier3` (seed {params.seed}).",
        "# All 16 placed-family configs, stratified by atom, cut 12 train / 4 eval.",
        "",
        f"cost_max = {MANIFEST_COST_MAX}",
        f"weight = {MANIFEST_WEIGHT}",
        "",
        "[provenance]",
        'mode = "enumerate"',
        f"seed = {params.seed}",
        f"train_count = {len(train)}",
        f"eval_count = {len(eval_)}",
        f"steps = {params.steps}",
        f"size = [{params.size[0]}, {params.size[1]}]",
    ]
    for split, variants in (("train", train), ("eval", eval_)):
        for variant in variants:
            plan = variant.plan
            lines += [
                "",
                "[[provenance.configs]]",
                f'atom = "{variant.config.atom.name.lower()}"',
                f"target = {variant.config.target}",
                f"long_arms = {str(variant.config.long_arms).lower()}",
                f'split = "{split}"',
                f'variant = "{split}/{variant.oml.stem}"',
                f"entrance = [{plan.entrance.r}, {plan.entrance.s}]",
                f"exit = [{plan.exit.r}, {plan.exit.s}]",
                f'arm = "{plan.arm.blueprint}"',
                f"base = [{plan.arm.base.r}, {plan.arm.base.s}]",
                f"rotation = {plan.arm.rotation}",
            ]
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
