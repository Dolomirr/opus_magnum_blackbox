"""
Tier-1 bank generator: deterministic `levels/tier1_simple_arms/` variant files.

Every variant is a `vNN.oml` spec plus its same-stem reference `vNN.oms`,
emitted from stage-1 family generator and re-validated from disk after writing.
The same seed yields the same bank; regeneration overwrites in place and removes stale variant files.
"""

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from opus_magnum_sym import AtomType, Hex, Level, LevelSpec

from opus_magnum_blackbox.corpus import LEVELS_DIR
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.families import (
    FamilyError,
    hex_distance,
    reference_oms,
    reference_tokens,
    single_atom_spec,
)
from opus_magnum_blackbox.solutions import load_solutions

FAMILY_NAME = "tier1_simple_arms"
BANK_DIR = LEVELS_DIR / FAMILY_NAME

# reward normalization default recorded for the corpus scanner (stage 3)
MANIFEST_COST_MAX = 50
MANIFEST_WEIGHT = 1.0


@dataclass(frozen=True)
class BankParams:
    """
    Family parameters shared by sampling, emission and manifest provenance.
    """

    seed: int = 0
    train_count: int = 48
    eval_count: int = 16
    target: int = 2
    steps: int = 30
    size: tuple[int, int] = (6, 6)
    atom: AtomType = AtomType.WATER


@dataclass(frozen=True)
class Variant:
    """
    One emitted level/solution pair.
    """

    oml: Path
    oms: Path
    entrance: Hex
    exit: Hex

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.entrance.r, self.entrance.s, self.exit.r, self.exit.s)


@dataclass(frozen=True)
class Bank:
    """
    Emitted bank: variant lists per split + manifest path.
    """

    root: Path
    train: list[Variant]
    eval: list[Variant]
    manifest: Path

    @property
    def variants(self) -> list[Variant]:
        return [*self.train, *self.eval]


def sample_pairs(
    rng: random.Random,
    train_count: int,
    eval_count: int,
    size: tuple[int, int],
) -> tuple[list[tuple[Hex, Hex]], list[tuple[Hex, Hex]]]:
    """
    Seeded entrance/exit pairs.

    Entrance uniform on board, exit uniform among cells at hex-distance 1-2,
    deduplicated, split into disjoint train/eval slices.
    Deterministic for a given seed; raises when counts are larger then pair pool.
    """
    rows, cols = size
    cells = [Hex(r, s) for r in range(rows) for s in range(cols)]
    near = {cell: [other for other in cells if 1 <= hex_distance(cell, other) <= 2] for cell in cells}
    pool = sum(len(neighbours) for neighbours in near.values())
    total = train_count + eval_count
    if total > pool:
        raise ValueError(f"{train_count} + {eval_count} variants requested, but the {rows}x{cols} board only has {pool} pairs")
    taken: set[tuple[int, int, int, int]] = set()
    pairs: list[tuple[Hex, Hex]] = []
    while len(pairs) < total:
        entrance = rng.choice(cells)
        exit_ = rng.choice(near[entrance])
        key = (entrance.r, entrance.s, exit_.r, exit_.s)
        if key in taken:
            continue
        taken.add(key)
        pairs.append((entrance, exit_))
    return pairs[:train_count], pairs[train_count:]


def generate_bank(out_dir: Path = BANK_DIR, *, params: BankParams | None = None) -> Bank:
    """
    Emit bank as files and validate every pair from disk.
    """
    params = params or BankParams()
    rng = random.Random(params.seed)
    train_pairs, eval_pairs = sample_pairs(rng, params.train_count, params.eval_count, params.size)
    train = [
        _emit_variant(entrance, exit_, index, out_dir / "train", params)
        for index, (entrance, exit_) in enumerate(train_pairs)
    ]
    eval_ = [
        _emit_variant(entrance, exit_, index, out_dir / "eval", params)
        for index, (entrance, exit_) in enumerate(eval_pairs)
    ]
    _clean_stale(out_dir / "train", [path for variant in train for path in (variant.oml, variant.oms)])
    _clean_stale(out_dir / "eval", [path for variant in eval_ for path in (variant.oml, variant.oms)])
    manifest = out_dir / "manifest.toml"
    manifest.write_text(_manifest_text(params))
    return Bank(out_dir, train, eval_, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the tier-1 simple-arms level bank (deterministic, idempotent).")
    parser.add_argument("--seed", type=int, default=BankParams.seed, help="bank sampling seed")
    parser.add_argument("--train-count", type=int, default=BankParams.train_count, help="number of train variants")
    parser.add_argument("--eval-count", type=int, default=BankParams.eval_count, help="number of eval variants")
    parser.add_argument("--target", type=int, default=BankParams.target, help="deliveries per exit")
    parser.add_argument("--steps", type=int, default=BankParams.steps, help="sim step budget per variant")
    parser.add_argument("--out-dir", type=Path, default=BANK_DIR, help="bank root directory")
    args = parser.parse_args(argv)
    params = BankParams(
        seed=args.seed,
        train_count=args.train_count,
        eval_count=args.eval_count,
        target=args.target,
        steps=args.steps,
    )
    bank = generate_bank(args.out_dir, params=params)
    print(f"wrote {len(bank.train)} train + {len(bank.eval)} eval variants to {bank.root}")
    return 0


def _emit_variant(entrance: Hex, exit_: Hex, index: int, out_dir: Path, params: BankParams) -> Variant:
    spec = single_atom_spec(entrance, exit_, target=params.target, steps=params.steps, size=params.size, atom=params.atom)
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = Variant(out_dir / f"v{index:02d}.oml", out_dir / f"v{index:02d}.oms", entrance, exit_)
    variant.oml.write_text(_oml_text(spec, out_dir.name, index))
    variant.oms.write_text(reference_oms(spec))
    _check_pair(variant.oml, variant.oms)
    return variant


def _oml_text(spec: LevelSpec, split: str, index: int) -> str:
    rows, cols = spec.size
    atom_names = sorted({cell.atom_type.name.lower() for shape in spec.shapes for cell in shape.cells if cell.atom_type})
    lines = [
        "# DO NOT EDIT",
        "# Generated by `gen-tier1`; regenerate instead of editing.",
        "",
        f"# Tier-1 single-atom family, {split} variant {index:02d}.",
        "#",
        "# Fixed 1-cell typed entrance and exit one arm apart;",
        f"# reference solution moves one `arm` through {spec.target} delivery cycles.",
        "",
        "",
        "",
        f"size {rows} {cols}",
        f"steps {spec.steps}",
        "entrypoints_fixed 1",
        f"atom_types {' '.join(atom_names)}",
        "",
        f"target {spec.target}",
        "",
        "# 1-cell typed shapes, no links; entrypoints pinned.",
    ]
    for shape in spec.shapes:
        cell = shape.cells[0]
        type_suffix = f":{cell.atom_type.name.lower()}" if cell.atom_type is not None else ""
        lines.append(
            f"shape {shape.name} {shape.comp_type.name.lower()} cells=({cell.offset.r},{cell.offset.s}){type_suffix} links="
        )
    lines.append("")
    for ep in spec.entrypoints:
        if ep.pos is None:
            raise FamilyError(f"bank variants need fixed entrypoints; {ep.shape} has no position")
        lines.append(f"{ep.kind.name.lower()} {ep.shape} {ep.pos.r} {ep.pos.s}")
    lines += [
        "",
        "# Blueprints available to solution.",
        f"allow {' '.join(sorted(spec.allowed_blueprints))}",
        "",
    ]
    return "\n".join(lines)


def _check_pair(oml: Path, oms: Path) -> None:
    """
    Re-load the emitted files from disk and fail loudly on any mismatch:
    `load_solutions` decodes, mask-walks, applies and solves the pair.
    """
    spec = Level.from_file(oml).spec
    entry = LevelEntry(source=oml, cost_max=MANIFEST_COST_MAX, name=oml.stem, solutions=(oms,))
    tokens = load_solutions(entry)[0]
    if tokens != reference_tokens(spec):
        raise FamilyError(f"{oms.name}: emitted solution differs from the reference sequence for {oml.name}")


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


def _manifest_text(params: BankParams) -> str:
    """
    Per-family manifest consumed by the corpus scanner (stage 3): cost/weight
    defaults plus provenance for regeneration.
    """
    return "\n".join(
        [
            "DO NOT EDIT",
            "",
            f"# {FAMILY_NAME} family manifest; consumed by corpus scanner.",
            f"# Regenerate bank with `gen-tier1` (seed {params.seed}).",
            "",
            f"cost_max = {MANIFEST_COST_MAX}",
            f"weight = {MANIFEST_WEIGHT}",
            "",
            "[provenance]",
            f"seed = {params.seed}",
            f"train_count = {params.train_count}",
            f"eval_count = {params.eval_count}",
            f"target = {params.target}",
            f"steps = {params.steps}",
            f"size = [{params.size[0]}, {params.size[1]}]",
            f'atom = "{params.atom.name.lower()}"',
            "",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
