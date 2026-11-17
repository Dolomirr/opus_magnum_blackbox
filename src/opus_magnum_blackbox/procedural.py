"""
Procedural BC source: solver-fed supervised pairs, no materialized bank.

Pool construction reuses the family generators' own split code to reproduce
train splits exactly; entries excluded by split are re-checked against the
run's eval levels by the guard.

Per epoch, samples entries seeded by (seed, epoch), solves each,
and builds supervised batch  exactly as bank entries in `_run_pretrain_stage`.
Epoch-keyed seeding keeps every epoch's data deterministic and independent of global rng.

Pools are registered per family in `POOL_REGISTRY`: pool name -> pool builder,
canonical reference-tokens function, and config-identity functions.
"""

import random
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from opus_magnum_sym import ComponentType, Hex, LevelSpec

from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.config import PretrainConfig
from opus_magnum_blackbox.env import spec_encoder_features
from opus_magnum_blackbox.families import FamilyError, PlacedConfig, hex_distance
from opus_magnum_blackbox.families._common import _entrypoint_pos
from opus_magnum_blackbox.families.tier2_anywhere import anywhere_reference_tokens
from opus_magnum_blackbox.families.tier3_placed import placed_config, placed_reference_tokens, placed_spec
from opus_magnum_blackbox.gen_tier2 import (
    SINGLE,
    ClassifiedPair,
    EnumBankParams,
    classify_class_pairs,
    enum_quotas,
    stratified_sample,
)
from opus_magnum_blackbox.gen_tier3 import split_configs
from opus_magnum_blackbox.masking import ActionMasker
from opus_magnum_blackbox.rollout import RolloutBatch, build_supervised
from opus_magnum_blackbox.solutions import augment
from opus_magnum_blackbox.train import bc_cycle_indices

# fmt: off
type Tier2Pair  = tuple[Hex, Hex, LevelSpec]      # classified anywhere pair (gen_tier2.ClassifiedPair)
type Tier3Entry = tuple[PlacedConfig, LevelSpec]  # placed-family config with its verified unfixed spec
type Pool       = tuple                           # homogeneous entry tuples; shape is family-specific
# fmt: on


@dataclass(frozen=True)
class PoolFamily:
    """
    Per-family procedural wiring: pool construction, canonical reference
    supervision, and config identity.
    """

    # fmt: off
    build            : Callable[[int], Pool]            # seed -> pool entries
    reference_tokens : Callable[[LevelSpec], list[int]]
    entry_spec       : Callable[[object], LevelSpec]
    entry_key        : Callable[[object], object]
    spec_key         : Callable[[LevelSpec], object]
    entry_tag        : Callable[[object], str]
    # fmt: on


def pair_key(entrance: Hex, exit_: Hex) -> tuple[int, int, int, int]:
    """
    Entrance (r, s) + exit (r, s) coordinates of one family pair.
    """
    return (entrance.r, entrance.s, exit_.r, exit_.s)


@lru_cache(maxsize=1)
def single_train_pairs() -> tuple[ClassifiedPair, ...]:
    """
    The tier2_single train split, rebuilt from classification:
    every in-family single pair cut train-first by generator's own split.

    Reproduces generated bank's train pairs exactly, generator consumes
    same seeded shuffle for single class first, then re-checks
    eval exclusion at stage start.
    """
    strata, _ = classify_class_pairs()
    single = {key: pairs for key, pairs in strata.items() if key[0] == SINGLE}
    populations = {key: len(pairs) for key, pairs in single.items()}
    params = EnumBankParams()
    quotas = enum_quotas(populations, params.eval_percent)
    train, _eval = stratified_sample(single, quotas, random.Random(params.seed))
    return tuple(train)


@lru_cache(maxsize=1)
def tier3_train_entries() -> tuple[Tier3Entry, ...]:
    """
    The tier3_placed train split, rebuilt from the generator's own split:
    12 of the 16 configs, each with its solver-verified unfixed spec.
    """
    train, _eval = split_configs()
    return tuple((config, placed_spec(config)) for config in train)


def _ring1_pool() -> tuple[Tier2Pair, ...]:
    return tuple(pair for pair in single_train_pairs() if hex_distance(pair[0], pair[1]) <= 2)


def _stratified_half_pool(seed: int) -> tuple[Tier2Pair, ...]:
    by_distance: dict[int, list[Tier2Pair]] = {}
    for pair in single_train_pairs():
        by_distance.setdefault(hex_distance(pair[0], pair[1]), []).append(pair)
    rng = random.Random(seed)
    picked: list[Tier2Pair] = []
    for distance in sorted(by_distance):
        pairs = list(by_distance[distance])
        rng.shuffle(pairs)
        picked.extend(pairs[: len(pairs) // 2])
    return tuple(picked)


def _tier2_entry_spec(entry: Tier2Pair) -> LevelSpec:
    return entry[2]


def _tier2_entry_key(entry: Tier2Pair) -> tuple[int, int, int, int]:
    return pair_key(entry[0], entry[1])


def _tier2_spec_key(spec: LevelSpec) -> tuple[int, int, int, int]:
    return pair_key(_entrypoint_pos(spec, ComponentType.ENTRANCE), _entrypoint_pos(spec, ComponentType.EXIT))


def _tier2_entry_tag(entry: Tier2Pair) -> str:
    entrance, exit_, _ = entry
    return f"{entrance.r}:{entrance.s}:{exit_.r}:{exit_.s}"


def _tier2_family(build: Callable[[int], Pool]) -> PoolFamily:
    return PoolFamily(
        build=build,
        reference_tokens=anywhere_reference_tokens,
        entry_spec=_tier2_entry_spec,
        entry_key=_tier2_entry_key,
        spec_key=_tier2_spec_key,
        entry_tag=_tier2_entry_tag,
    )


def _tier3_entry_spec(entry: Tier3Entry) -> LevelSpec:
    return entry[1]


def _tier3_entry_key(entry: Tier3Entry) -> tuple[str, int, bool]:
    config, _ = entry
    return (config.atom.name, config.target, config.long_arms)


def _tier3_spec_key(spec: LevelSpec) -> tuple[str, int, bool]:
    config = placed_config(spec)
    return (config.atom.name, config.target, config.long_arms)


def _tier3_entry_tag(entry: Tier3Entry) -> str:
    return ":".join(map(str, _tier3_entry_key(entry)))


def _tier3_pool(_seed: int) -> tuple[Tier3Entry, ...]:
    return tier3_train_entries()


POOL_REGISTRY: dict[str, PoolFamily] = {
    "full": _tier2_family(lambda _seed: single_train_pairs()),
    "ring1": _tier2_family(lambda _seed: _ring1_pool()),
    "stratified_half": _tier2_family(_stratified_half_pool),
    "tier3_placed": PoolFamily(
        build=_tier3_pool,
        reference_tokens=placed_reference_tokens,
        entry_spec=_tier3_entry_spec,
        entry_key=_tier3_entry_key,
        spec_key=_tier3_spec_key,
        entry_tag=_tier3_entry_tag,
    ),
}

POOL_NAMES = tuple(POOL_REGISTRY)


def pool_family(name: str) -> PoolFamily:
    """
    Registry record for one pool name: builder, reference supervision and config identity.
    """
    try:
        return POOL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown procedural pool {name!r}; expected one of {list(POOL_NAMES)}") from None


def build_pool(name: str, *, seed: int = 0) -> Pool:
    """
    Classified train entries of configured pool, deterministic.
    """
    return pool_family(name).build(seed)


def guard_excludes_eval(pool: Pool, eval_specs: list[LevelSpec], family: PoolFamily) -> None:
    """
    Hard guard: no pool entry's config may appear on the eval bank, keyed by
    family config identity. `eval_specs` are the specs of the run's eval
    levels; overlap raises with the offending identities.
    """
    eval_keys = {family.spec_key(spec) for spec in eval_specs}
    overlap = sorted({family.entry_key(entry) for entry in pool} & eval_keys)
    if overlap:
        raise ValueError(
            f"procedural pool overlaps the eval bank at {len(overlap)} configs, e.g. {overlap[:8]}; "
            "the pool must be drawn from the train split only"
        )


def procedural_epoch_batches(  # noqa: PLR0913, PLR0917
    pool: Pool,
    family: PoolFamily,
    config: PretrainConfig,
    tokenizer: OpusTokenizer,
    masker: ActionMasker,
    features: tuple[str, ...],
    *,
    seed: int,
    epoch: int,
) -> tuple[list[RolloutBatch], int]:
    """
    One epoch's supervised batches plus the solver-failure count.
    """
    if config.bc_cycle > len(pool):
        raise ValueError(f"bc_cycle {config.bc_cycle} exceeds the {config.procedural_pool!r} pool size {len(pool)}")
    batches: list[RolloutBatch] = []
    failures = 0
    for pick in bc_cycle_indices(len(pool), config.bc_cycle, seed, epoch):
        entry = pool[pick]
        spec = family.entry_spec(entry)
        try:
            tokens = family.reference_tokens(spec)
        except FamilyError:
            failures += 1
            continue
        variants = [tokens]
        if config.augment:
            aug_seed = zlib.crc32(f"bc-proc-aug:{seed}:{epoch}:{family.entry_tag(entry)}".encode())
            variants.extend(augment(tokens, spec, seed=aug_seed, max_variants=config.max_variants))
        side_channel = spec_encoder_features(spec, features) if features else None
        batches.append(build_supervised(variants, spec, tokenizer, masker, encoder_features=side_channel))
    if not batches:
        raise ValueError(f"procedural epoch {epoch}: every sampled pair failed to solve ({failures} failures)")
    return batches, failures
