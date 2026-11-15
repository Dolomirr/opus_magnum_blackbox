"""
Tier-2 anywhere bank generator: deterministic variant banks under `levels/`.

Every variant is a `vNNN.oml` spec plus its same-stem reference `vNNN.oms`,
emitted from the anywhere family generator and re-validated from disk after writing.

Two bank shapes, selected by mode:

- `sample` (mixed bank, `levels/tier2_anywhere/`): a seeded stratified sample, not enumeration.
  Classification sweeps every ordered entrance/exit pair once and
  groups it into a stratum — (hex distance, single/handoff per the winning plan's
  arm count); pairs the family rejects are counted per distance, not raised. The
  per-class split quotas (`train_single`/`train_handoff`/`eval_single`/
  `eval_handoff`) are apportioned over each class's strata proportionally to
  stratum population (largest remainder, capped at capacity), every stratum is
  seeded-shuffled and cut train-first, so the splits are disjoint by construction
  and each kept pair is uniform within its stratum.
- `classes` (per-class banks, `levels/tier2_single/` + `levels/tier2_handoff/`):
  full enumeration. Every in-family pair of a class is kept, cut ~80/20 train/eval
  with the eval share apportioned over the class's strata — (winning ring, hex
  distance) for singles, hex distance for handoffs — so together the banks
  partition the classified pair space exactly. Disjoint by construction; zero
  rejects on the full 6x6 board.

Handoffs deliver exactly one molecule (target 1) — two cycles would sit at the
32-triplet cap — and single arms deliver target 2, so the class rule follows the
winning plan.

Same parameters yield the same bank; regeneration overwrites in place and removes stale variant files.
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
    BoardSize,
    FamilyError,
    anywhere_reference_oms,
    anywhere_reference_tokens,
    anywhere_spec_plan,
    hex_distance,
)
from opus_magnum_blackbox.solutions import load_solutions

FAMILY_NAME = "tier2_anywhere"
BANK_DIR = LEVELS_DIR / FAMILY_NAME

# reward normalization default recorded for the corpus scanner
MANIFEST_COST_MAX = 50
MANIFEST_WEIGHT = 1.0

SINGLE = "single"  # one arm serves the pair, target 2
HANDOFF = "handoff"  # two arms hand the atom over, target 1

CLASS_BANK_NAMES = {SINGLE: "tier2_single", HANDOFF: "tier2_handoff"}
SINGLE_BANK_DIR = LEVELS_DIR / CLASS_BANK_NAMES[SINGLE]
HANDOFF_BANK_DIR = LEVELS_DIR / CLASS_BANK_NAMES[HANDOFF]

# fmt: off
type EntranceExit       = tuple[Hex, Hex]                          # (entrance, exit) cells of one family pair
type PairKey            = tuple[int, int, int, int]                # entrance (r, s) + exit (r, s)
type Stratum            = tuple[int, str]                          # (hex distance, SINGLE | HANDOFF)
type ClassifiedPair     = tuple[Hex, Hex, LevelSpec]               # pair with its verified spec
type StratumPairs       = dict[Stratum, list[ClassifiedPair]]      # classified pairs keyed by stratum
type StratumPopulations = dict[Stratum, int]                       # per-stratum pair counts
type StratumQuotas      = dict[Stratum, tuple[int, int]]           # per-stratum (train, eval) quotas
type StratumCounts      = dict[Stratum, dict[str, int]]            # stratum -> {"train", "eval"} counts
type DistanceCounts     = dict[int, int]                           # FamilyError rejects per distance class
type ClassStratum       = tuple[str, int, int]                     # (SINGLE | HANDOFF, arm ring, hex distance); ring 0 for handoffs
type ClassStrata        = dict[ClassStratum, list[ClassifiedPair]] # class-stratified pairs
type ClassQuotas        = dict[ClassStratum, tuple[int, int]]      # per-stratum (train, eval) quotas
# fmt: on


@dataclass(frozen=True)
class BankParams:
    """
    Family parameters shared by classification, quota apportioning, emission and manifest provenance.

    `train_*`/`eval_*` are per-class quotas; per-stratum quotas are derived
    proportionally to stratum population (see `stratify_quotas`).
    """

    seed: int = 0
    train_single: int = 150
    train_handoff: int = 300
    eval_single: int = 40
    eval_handoff: int = 80
    steps: int = 30
    size: BoardSize = (6, 6)
    atom: AtomType = AtomType.WATER


@dataclass(frozen=True)
class EnumBankParams:
    """
    Per-class enumerated-bank parameters: every in-family pair of a class is
    kept, cut ~`(100 - eval_percent)`/`eval_percent` train/eval, stratified by
    (ring, distance) for singles and by distance for handoffs.
    """

    seed: int = 0
    eval_percent: int = 20
    steps: int = 30
    size: BoardSize = (6, 6)
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
    spec: LevelSpec

    @property
    def key(self) -> PairKey:
        return (self.entrance.r, self.entrance.s, self.exit.r, self.exit.s)

    @property
    def kind(self) -> str:
        """`HANDOFF` when the spec was built at the handoff class target (1), else `SINGLE`."""
        return HANDOFF if self.spec.target == 1 else SINGLE


@dataclass(frozen=True)
class Bank:
    """
    Emitted bank: variant lists per split, per-stratum quotas and manifest path.

    `rejected` counts classification `FamilyError` rejects per distance class.
    """

    root: Path
    train: list[Variant]
    eval: list[Variant]
    manifest: Path
    quotas: StratumQuotas
    rejected: DistanceCounts

    @property
    def variants(self) -> list[Variant]:
        return [*self.train, *self.eval]


@dataclass(frozen=True)
class ClassBank:
    """
    One emitted per-class bank: every in-family pair of the class, split train/eval.

    `quotas` are keyed by class stratum; train + eval covers the full class population.
    """

    root: Path
    train: list[Variant]
    eval: list[Variant]
    manifest: Path
    quotas: ClassQuotas

    @property
    def variants(self) -> list[Variant]:
        return [*self.train, *self.eval]


def classify_pairs(
    *,
    steps: int = 30,
    size: BoardSize = (6, 6),
    atom: AtomType = AtomType.WATER,
) -> tuple[StratumPairs, DistanceCounts]:
    """
    Every in-family ordered pair grouped by stratum, plus `FamilyError` rejects per distance.

    Iterates all cells in sorted (r, s) order, entrance-major; each pair is solved
    once by `anywhere_spec_plan`, whose winning plan names the stratum: one arm ->
    single, two arms -> handoff. Out-of-family pairs never raise, they are counted.
    """
    rows, cols = size
    cells = [Hex(r, s) for r in range(rows) for s in range(cols)]
    strata: StratumPairs = {}
    rejected: DistanceCounts = {}
    for entrance in cells:
        for exit_ in cells:
            if exit_ == entrance:
                continue
            distance = hex_distance(entrance, exit_)
            try:
                spec, plan = anywhere_spec_plan(entrance, exit_, steps=steps, size=size, atom=atom)
            except FamilyError:
                rejected[distance] = rejected.get(distance, 0) + 1
                continue
            kind = HANDOFF if len(plan.arms) >= 2 else SINGLE
            strata.setdefault((distance, kind), []).append((entrance, exit_, spec))
    return strata, rejected


def classify_class_pairs(
    *,
    steps: int = 30,
    size: BoardSize = (6, 6),
    atom: AtomType = AtomType.WATER,
) -> tuple[ClassStrata, DistanceCounts]:
    """
    Every in-family ordered pair grouped by class stratum.
    """
    rows, cols = size
    cells = [Hex(r, s) for r in range(rows) for s in range(cols)]
    strata: ClassStrata = {}
    rejected: DistanceCounts = {}
    for entrance in cells:
        for exit_ in cells:
            if exit_ == entrance:
                continue
            distance = hex_distance(entrance, exit_)
            try:
                spec, plan = anywhere_spec_plan(entrance, exit_, steps=steps, size=size, atom=atom)
            except FamilyError:
                rejected[distance] = rejected.get(distance, 0) + 1
                continue
            if len(plan.arms) >= 2:
                stratum = (HANDOFF, 0, distance)
            else:
                blueprint = plan.arms[0].blueprint
                ring = 1 if blueprint == "arm" else int(blueprint.removeprefix("arm_l"))
                stratum = (SINGLE, ring, distance)
            strata.setdefault(stratum, []).append((entrance, exit_, spec))
    return strata, rejected


def stratify_quotas(populations: StratumPopulations, params: BankParams) -> StratumQuotas:
    """
    Per-stratum (train, eval) quotas: each class's split totals are apportioned over
    its strata proportionally to population, eval first, train over remaining capacity.
    """
    quotas: StratumQuotas = {}
    for kind, train_total, eval_total in (
        (SINGLE, params.train_single, params.eval_single),
        (HANDOFF, params.train_handoff, params.eval_handoff),
    ):
        pops = {key: pop for key, pop in populations.items() if key[1] == kind}
        eval_shares = _apportion(eval_total, pops)
        train_shares = _apportion(train_total, {key: pop - eval_shares[key] for key, pop in pops.items()})
        quotas.update({key: (train_shares[key], eval_shares[key]) for key in pops})
    return quotas


def enum_quotas(populations: dict[ClassStratum, int], eval_percent: int) -> ClassQuotas:
    """
    Per-stratum (train, eval) quotas for enumerated bank:
    half-up `eval_percent` share of population is portioned over strata
    proportionally to population (largest remainder), each stratum's train
    quota is its remainder - so train + eval covers every pair exactly once.
    """
    eval_total = (sum(populations.values()) * eval_percent + 50) // 100
    eval_shares = _apportion(eval_total, populations)
    return {key: (pop - eval_shares[key], eval_shares[key]) for key, pop in populations.items()}


def stratified_sample(
    strata: StratumPairs,
    quotas: StratumQuotas,
    rng: random.Random,
) -> tuple[list[ClassifiedPair], list[ClassifiedPair]]:
    """
    Seeded shuffle within every stratum, cut train-first by its (train, eval) quota.

    Deterministic for a given rng state; strata are disjoint, so the splits are too.
    """
    train: list[ClassifiedPair] = []
    eval_: list[ClassifiedPair] = []
    for stratum in sorted(strata):
        pairs = list(strata[stratum])
        rng.shuffle(pairs)
        train_quota, eval_quota = quotas.get(stratum, (0, 0))
        train.extend(pairs[:train_quota])
        eval_.extend(pairs[train_quota : train_quota + eval_quota])
    train_keys = {(pair[0].r, pair[0].s, pair[1].r, pair[1].s) for pair in train}
    eval_keys = {(pair[0].r, pair[0].s, pair[1].r, pair[1].s) for pair in eval_}
    overlap = train_keys & eval_keys
    if overlap:
        raise ValueError(f"train/eval splits overlap at {sorted(overlap)}")
    return train, eval_


def generate_bank(out_dir: Path = BANK_DIR, *, params: BankParams | None = None) -> Bank:
    """
    Emit bank as files and validate every pair from disk.

    Classification sweeps the whole ordered pair space once, quotas are apportioned
    per stratum, then sampling, emission and on-disk re-validation run per split.
    """
    params = params or BankParams()
    rng = random.Random(params.seed)
    strata, rejected = classify_pairs(steps=params.steps, size=params.size, atom=params.atom)
    if not strata:
        raise ValueError(f"classification kept no pairs on the {params.size[0]}x{params.size[1]} board")
    populations = {stratum: len(pairs) for stratum, pairs in strata.items()}
    quotas = stratify_quotas(populations, params)
    train_pairs, eval_pairs = stratified_sample(strata, quotas, rng)
    if not train_pairs and not eval_pairs:
        raise ValueError(f"quotas kept no pairs on the {params.size[0]}x{params.size[1]} board")
    train = [
        _emit_variant(spec, entrance, exit_, index, out_dir / "train")
        for index, (entrance, exit_, spec) in enumerate(train_pairs)
    ]
    eval_ = [
        _emit_variant(spec, entrance, exit_, index, out_dir / "eval")
        for index, (entrance, exit_, spec) in enumerate(eval_pairs)
    ]
    _clean_stale(out_dir / "train", [path for variant in train for path in (variant.oml, variant.oms)])
    _clean_stale(out_dir / "eval", [path for variant in eval_ for path in (variant.oml, variant.oms)])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.toml"
    breakdown = _stratum_breakdown(train, eval_)
    manifest.write_text(_manifest_text(params, len(train), len(eval_), breakdown, rejected))
    # self-ignore
    (out_dir / ".gitignore").write_text("# Generated bank; reproduce with `just gen-tier2`.\n*\n")
    return Bank(out_dir, train, eval_, manifest, quotas, rejected)


def generate_class_banks(out_dir: Path = LEVELS_DIR, *, params: EnumBankParams | None = None) -> dict[str, ClassBank]:
    """
    Emit both per-class banks as files and validate every pair from disk.

    One classification sweep feeds both banks: every in-family pair of a class is
    kept, cut ~`(100 - eval_percent)`/`eval_percent` train/eval with the eval
    share apportioned over the class's strata. Splits stay disjoint and their
    union complete by construction; the two banks together partition the
    classified pair space exactly.
    """
    params = params or EnumBankParams()
    strata, rejected = classify_class_pairs(steps=params.steps, size=params.size, atom=params.atom)
    rng = random.Random(params.seed)
    banks: dict[str, ClassBank] = {}
    for kind in (SINGLE, HANDOFF):
        class_strata = {key: pairs for key, pairs in strata.items() if key[0] == kind}
        if not class_strata:
            raise ValueError(f"classification kept no {kind} pairs on the {params.size[0]}x{params.size[1]} board")
        populations = {key: len(pairs) for key, pairs in class_strata.items()}
        quotas = enum_quotas(populations, params.eval_percent)
        train_pairs, eval_pairs = stratified_sample(class_strata, quotas, rng)
        kept = len(train_pairs) + len(eval_pairs)
        if kept != sum(populations.values()):
            raise ValueError(f"{kind} split kept {kept} of {sum(populations.values())} classified pairs")
        root = out_dir / CLASS_BANK_NAMES[kind]
        train = [
            _emit_variant(spec, entrance, exit_, index, root / "train")
            for index, (entrance, exit_, spec) in enumerate(train_pairs)
        ]
        eval_ = [
            _emit_variant(spec, entrance, exit_, index, root / "eval")
            for index, (entrance, exit_, spec) in enumerate(eval_pairs)
        ]
        _clean_stale(root / "train", [path for variant in train for path in (variant.oml, variant.oms)])
        _clean_stale(root / "eval", [path for variant in eval_ for path in (variant.oml, variant.oms)])
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.toml"
        manifest.write_text(_class_manifest_text(kind, params, len(train), len(eval_), quotas, sum(rejected.values())))
        # self-ignore
        (root / ".gitignore").write_text("# Generated bank; reproduce with `just gen-tier2-classes`.\n*\n")
        banks[CLASS_BANK_NAMES[kind]] = ClassBank(root, train, eval_, manifest, quotas)
    return banks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the tier-2 anywhere level banks (deterministic, idempotent)."
    )
    parser.add_argument(
        "--mode",
        choices=("sample", "classes"),
        default="sample",
        help="sample: quota-stratified mixed bank; classes: enumerated per-class banks (single + handoff)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BankParams.seed,
        help="sampling and split seed",
    )
    parser.add_argument(
        "--train-single",
        type=int,
        default=BankParams.train_single,
        help="train quota for single-arm variants",
    )
    parser.add_argument(
        "--train-handoff",
        type=int,
        default=BankParams.train_handoff,
        help="train quota for handoff variants",
    )
    parser.add_argument(
        "--eval-single",
        type=int,
        default=BankParams.eval_single,
        help="eval quota for single-arm variants",
    )
    parser.add_argument(
        "--eval-handoff",
        type=int,
        default=BankParams.eval_handoff,
        help="eval quota for handoff variants",
    )
    parser.add_argument(
        "--eval-percent",
        type=int,
        default=EnumBankParams.eval_percent,
        help="eval share of each class bank, percent",
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
        help="bank root directory (default: levels/ for classes, the tier2_anywhere bank for sample)",
    )
    args = parser.parse_args(argv)
    if args.mode == "sample":
        params = BankParams(
            seed=args.seed,
            train_single=args.train_single,
            train_handoff=args.train_handoff,
            eval_single=args.eval_single,
            eval_handoff=args.eval_handoff,
            steps=args.steps,
        )
        bank = generate_bank(args.out_dir or BANK_DIR, params=params)
        print(f"wrote {len(bank.train)} train + {len(bank.eval)} eval variants to {bank.root}")
        for kind in (SINGLE, HANDOFF):
            train_n = sum(1 for variant in bank.train if variant.kind == kind)
            eval_n = sum(1 for variant in bank.eval if variant.kind == kind)
            print(f"  {kind}: train {train_n}, eval {eval_n}")
        print(f"  rejected pairs: {sum(bank.rejected.values())}")
        return 0
    params = EnumBankParams(seed=args.seed, eval_percent=args.eval_percent, steps=args.steps)
    banks = generate_class_banks(args.out_dir or LEVELS_DIR, params=params)
    for bank in banks.values():
        print(f"wrote {len(bank.train)} train + {len(bank.eval)} eval variants to {bank.root}")
        for (kind, ring, distance), (train_n, eval_n) in sorted(bank.quotas.items()):
            label = f"ring {ring} distance {distance}" if kind == SINGLE else f"distance {distance}"
            print(f"  {label}: train {train_n}, eval {eval_n}")
    return 0


def _apportion(total: int, capacity: dict[Stratum, int]) -> dict[Stratum, int]:
    """
    Largest-remainder split of `total` proportional to `capacity`, every share capped
    at its capacity, so the returned total shrinks when capacity runs out.

    Exact integer math: floor shares of `total * capacity / pool`, then hand the
    leftover to the largest remainders, ties broken by stratum order.
    """
    pool = sum(capacity.values())
    if total <= 0 or pool <= 0:
        return dict.fromkeys(capacity, 0)
    total = min(total, pool)
    shares = {key: total * cap // pool for key, cap in capacity.items()}
    leftover = total - sum(shares.values())
    by_remainder = sorted(capacity, key=lambda key: (-(total * capacity[key] % pool), key))
    for key in by_remainder[:leftover]:
        shares[key] += 1
    return shares


def _emit_variant(spec: LevelSpec, entrance: Hex, exit_: Hex, index: int, out_dir: Path) -> Variant:
    out_dir.mkdir(parents=True, exist_ok=True)
    variant = Variant(out_dir / f"v{index:03d}.oml", out_dir / f"v{index:03d}.oms", entrance, exit_, spec)
    variant.oml.write_text(_oml_text(spec, out_dir.name, index))
    variant.oms.write_text(anywhere_reference_oms(spec))
    _check_pair(variant.oml, variant.oms, spec)
    return variant


def _oml_text(spec: LevelSpec, split: str, index: int) -> str:
    rows, cols = spec.size
    atom_names = sorted(
        {cell.atom_type.name.lower() for shape in spec.shapes for cell in shape.cells if cell.atom_type}
    )
    lines = [
        "# DO NOT EDIT",
        "# Generated by `gen-tier2`; regenerate instead of editing.",
        "",
        f"# Tier-2 anywhere family, {split} variant {index:03d}.",
        "#",
        "# Fixed 1-cell typed entrance and exit anywhere on the board;",
        "# reference solution delivers with one arm (target 2) or a two-arm handoff (target 1).",
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


def _check_pair(oml: Path, oms: Path, spec: LevelSpec) -> None:
    """
    Re-load the emitted files from disk and fail loudly on any mismatch: the level
    must carry the classified pair's entrypoints and class rule, and `load_solutions`
    decodes, mask-walks, applies and solves the pair.
    """
    disk_spec = Level.from_file(oml).spec
    if (disk_spec.entrypoints, disk_spec.target, disk_spec.steps) != (spec.entrypoints, spec.target, spec.steps):
        raise FamilyError(f"{oml.name}: emitted spec differs from the classified spec")
    entry = LevelEntry(source=oml, cost_max=MANIFEST_COST_MAX, name=oml.stem, solutions=(oms,))
    tokens = load_solutions(entry)[0]
    if tokens != anywhere_reference_tokens(spec):
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


def _stratum_breakdown(train: list[Variant], eval_: list[Variant]) -> StratumCounts:
    """
    Kept pair counts per stratum and split, strata in sorted order.
    """
    breakdown: StratumCounts = {}
    for split, variants in (("train", train), ("eval", eval_)):
        for variant in variants:
            stratum = (hex_distance(variant.entrance, variant.exit), variant.kind)
            counts = breakdown.setdefault(stratum, {"train": 0, "eval": 0})
            counts[split] += 1
    return {stratum: breakdown[stratum] for stratum in sorted(breakdown)}


def _manifest_text(
    params: BankParams,
    train_count: int,
    eval_count: int,
    breakdown: StratumCounts,
    rejected: DistanceCounts,
) -> str:
    """
    Per-family manifest consumed by the corpus scanner: cost/weight defaults plus
    provenance for regeneration — quotas, per-stratum counts, class split, rejects.
    """
    classes: dict[str, dict[str, int]] = {kind: {"train": 0, "eval": 0} for kind in (SINGLE, HANDOFF)}
    for (_, kind), counts in breakdown.items():
        classes[kind]["train"] += counts["train"]
        classes[kind]["eval"] += counts["eval"]
    lines = [
        "# DO NOT EDIT",
        "",
        f"# {FAMILY_NAME} family manifest; consumed by corpus scanner.",
        f"# Regenerate bank with `gen-tier2` (mode sample, seed {params.seed}).",
        "# Targets follow the class rule: 2 for single-arm variants, 1 for handoffs.",
        "",
        f"cost_max = {MANIFEST_COST_MAX}",
        f"weight = {MANIFEST_WEIGHT}",
        "",
        "[provenance]",
        'mode = "sample"',
        f"seed = {params.seed}",
        f"train_count = {train_count}",
        f"eval_count = {eval_count}",
        f"steps = {params.steps}",
        f"size = [{params.size[0]}, {params.size[1]}]",
        f'atom = "{params.atom.name.lower()}"',
        f"rejected = {sum(rejected.values())}",
        "",
        "[provenance.quotas]",
        f"train_single = {params.train_single}",
        f"train_handoff = {params.train_handoff}",
        f"eval_single = {params.eval_single}",
        f"eval_handoff = {params.eval_handoff}",
    ]
    for kind in (SINGLE, HANDOFF):
        lines += [
            "",
            f"[provenance.classes.{kind}]",
            f"train = {classes[kind]['train']}",
            f"eval = {classes[kind]['eval']}",
        ]
    for stratum, counts in breakdown.items():
        lines += [
            "",
            f"[provenance.strata.{stratum[0]}.{stratum[1]}]",
            f"train = {counts['train']}",
            f"eval = {counts['eval']}",
        ]
    lines.append("")
    return "\n".join(lines)


def _class_manifest_text(  # noqa: PLR0913, PLR0917
    kind: str,
    params: EnumBankParams,
    train_count: int,
    eval_count: int,
    quotas: ClassQuotas,
    rejected: int,
) -> str:
    """
    Per-class manifest consumed by the corpus scanner: cost/weight defaults plus
    provenance for regeneration — mode/seed/eval share, counts, and the
    per-stratum (train, eval) quotas. Single strata nest ring then distance,
    handoff strata nest distance only.
    """
    stratification = "(ring, distance)" if kind == SINGLE else "distance"
    lines = [
        "# DO NOT EDIT",
        "",
        f"# {CLASS_BANK_NAMES[kind]} bank manifest; consumed by corpus scanner.",
        f"# Regenerate bank with `gen-tier2-classes` (mode enumerate, seed {params.seed}).",
        f"# Every in-family {kind} pair, stratified by {stratification},",
        f"# split ~{100 - params.eval_percent}/{params.eval_percent} train/eval.",
        "",
        f"cost_max = {MANIFEST_COST_MAX}",
        f"weight = {MANIFEST_WEIGHT}",
        "",
        "[provenance]",
        'mode = "enumerate"',
        f"seed = {params.seed}",
        f"eval_percent = {params.eval_percent}",
        f"train_count = {train_count}",
        f"eval_count = {eval_count}",
        f"steps = {params.steps}",
        f"size = [{params.size[0]}, {params.size[1]}]",
        f'atom = "{params.atom.name.lower()}"',
        f"rejected = {rejected}",
        "",
    ]
    for (_, ring, distance), (train_quota, eval_quota) in sorted(quotas.items()):
        table = f"strata.{ring}.{distance}" if kind == SINGLE else f"strata.{distance}"
        lines += [
            f"[provenance.{table}]",
            f"train = {train_quota}",
            f"eval = {eval_quota}",
            "",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
