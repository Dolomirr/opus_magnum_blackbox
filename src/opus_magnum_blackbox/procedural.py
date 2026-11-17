"""
Procedural BC source: solver-fed supervised pairs, no materialized bank.

Pool construction reuses the tier-2 anywhere classification to reproduce
tier2_single train split exactly; part of pairs are excluded by split
and re-checked against run's eval levels.

Per epoch, samples pairs seeded by (seed, epoch), solves each,
and builds supervised batch  exactly as bank entries in `_run_pretrain_stage`.
Epoch-keyed seeding keeps every epoch's data deterministic and independent of global rng.
"""

import random
import zlib
from functools import lru_cache

from opus_magnum_sym import ComponentType, Hex, LevelSpec

from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.config import PretrainConfig
from opus_magnum_blackbox.env import spec_encoder_features
from opus_magnum_blackbox.families import FamilyError, hex_distance
from opus_magnum_blackbox.families._common import _entrypoint_pos
from opus_magnum_blackbox.families.tier2_anywhere import anywhere_reference_tokens
from opus_magnum_blackbox.gen_tier2 import (
    SINGLE,
    ClassifiedPair,
    EnumBankParams,
    classify_class_pairs,
    enum_quotas,
    stratified_sample,
)
from opus_magnum_blackbox.masking import ActionMasker
from opus_magnum_blackbox.rollout import RolloutBatch, build_supervised
from opus_magnum_blackbox.solutions import augment
from opus_magnum_blackbox.train import bc_cycle_indices

POOL_NAMES = ("full", "stratified_half", "ring1")


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


def build_pool(name: str, *, seed: int = 0) -> tuple[ClassifiedPair, ...]:
    """
    Classified train pairs of configured pool, deterministic.
    """
    train = single_train_pairs()
    if name == "full":
        return train
    if name == "ring1":
        return tuple(pair for pair in train if hex_distance(pair[0], pair[1]) <= 2)
    if name == "stratified_half":
        by_distance: dict[int, list[ClassifiedPair]] = {}
        for pair in train:
            by_distance.setdefault(hex_distance(pair[0], pair[1]), []).append(pair)
        rng = random.Random(seed)
        picked: list[ClassifiedPair] = []
        for distance in sorted(by_distance):
            pairs = list(by_distance[distance])
            rng.shuffle(pairs)
            picked.extend(pairs[: len(pairs) // 2])
        return tuple(picked)
    raise ValueError(f"unknown procedural pool {name!r}; expected one of {list(POOL_NAMES)}")


def guard_excludes_eval(pool: tuple[ClassifiedPair, ...], eval_specs: list[LevelSpec]) -> None:
    """
    Hard guard: no pool pair may appear on the eval bank. `eval_specs` are the
    specs of the run's eval levels; overlap raises with the offending pairs.
    """
    eval_keys = {
        pair_key(_entrypoint_pos(spec, ComponentType.ENTRANCE), _entrypoint_pos(spec, ComponentType.EXIT))
        for spec in eval_specs
    }
    overlap = sorted({pair_key(entrance, exit_) for entrance, exit_, _ in pool} & eval_keys)
    if overlap:
        raise ValueError(
            f"procedural pool overlaps the eval bank at {len(overlap)} pairs, e.g. {overlap[:8]}; "
            "the pool must be drawn from the train-pair space only"
        )


def procedural_epoch_batches(  # noqa: PLR0913
    pool: tuple[ClassifiedPair, ...],
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
        entrance, exit_, spec = pool[pick]
        try:
            tokens = anywhere_reference_tokens(spec)
        except FamilyError:
            failures += 1
            continue
        variants = [tokens]
        if config.augment:
            aug_seed = zlib.crc32(f"bc-proc-aug:{seed}:{epoch}:{entrance.r}:{entrance.s}:{exit_.r}:{exit_.s}".encode())
            variants.extend(augment(tokens, spec, seed=aug_seed, max_variants=config.max_variants))
        side_channel = spec_encoder_features(spec, features) if features else None
        batches.append(build_supervised(variants, spec, tokenizer, masker, encoder_features=side_channel))
    if not batches:
        raise ValueError(f"procedural epoch {epoch}: every sampled pair failed to solve ({failures} failures)")
    return batches, failures
