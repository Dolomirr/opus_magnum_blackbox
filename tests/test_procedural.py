"""Procedural BC source: pool construction, per-epoch solver-fed batches, guards, config plumbing."""

import json
from collections import Counter
from dataclasses import replace

import pytest
import torch
from conftest import make_spec
from opus_magnum_sym import Hex

from opus_magnum_blackbox import stages
from opus_magnum_blackbox.config import PretrainConfig
from opus_magnum_blackbox.corpus import LEVELS_DIR
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.families import anywhere_spec, anywhere_spec_plan, hex_distance, placed_spec
from opus_magnum_blackbox.gen_tier2 import SINGLE_BANK_DIR
from opus_magnum_blackbox.gen_tier3 import split_configs
from opus_magnum_blackbox.procedural import (
    build_pool,
    guard_excludes_eval,
    pair_key,
    pool_family,
    procedural_epoch_batches,
)
from opus_magnum_blackbox.stages import PretrainStage, load_pipeline, run_pipeline

TIER0_OMS = LEVELS_DIR / "tier0_typed_delivery.oms"

# Verified anywhere pairs spanning single (target 2) and handoff (target 1) classes;
# small enough that no test needs the full classification sweep.
TINY_CANDIDATES = [
    (Hex(0, 0), Hex(0, 1)),  # d1 single
    (Hex(0, 0), Hex(1, 1)),  # d2 single
    (Hex(2, 2), Hex(2, 4)),  # d2 single
    (Hex(0, 0), Hex(4, 3)),  # d7 handoff
    (Hex(1, 1), Hex(3, 2)),  # d3 handoff
    (Hex(2, 2), Hex(4, 4)),  # d4 single
]


@pytest.fixture(scope="module")
def tiny_pool():
    return tuple((entrance, exit_, anywhere_spec_plan(entrance, exit_)[0]) for entrance, exit_ in TINY_CANDIDATES)


def _proc_config(**overrides) -> PretrainConfig:
    return replace(PretrainConfig(bc_cycle=3, augment=False), **overrides)


def _assert_same_data(first, second):
    for batch_a, batch_b in zip(first, second, strict=True):
        for tensor_a, tensor_b in zip(
            (
                batch_a.encoder_input,
                batch_a.encoder_length,
                batch_a.actions,
                batch_a.decoder_length,
                batch_a.action_masks,
                batch_a.valid,
            ),
            (
                batch_b.encoder_input,
                batch_b.encoder_length,
                batch_b.actions,
                batch_b.decoder_length,
                batch_b.action_masks,
                batch_b.valid,
            ),
            strict=True,
        ):
            assert torch.equal(tensor_a, tensor_b)


def test_epoch_batches_solve_deterministically_and_are_epoch_keyed(tokenizer, masker, tiny_pool):
    config = _proc_config()
    batches, failures = procedural_epoch_batches(tiny_pool, pool_family("full"), config, tokenizer, masker, (), seed=0, epoch=1)
    assert failures == 0
    assert len(batches) == config.bc_cycle  # one supervised batch per sampled pair

    torch.rand(1000)  # advance the global stream: the epoch draw must not depend on it
    repeated, repeated_failures = procedural_epoch_batches(
        tiny_pool, pool_family("full"), config, tokenizer, masker, (), seed=0, epoch=1
    )
    assert repeated_failures == 0
    _assert_same_data(batches, repeated)

    other_epoch, other_failures = procedural_epoch_batches(
        tiny_pool, pool_family("full"), config, tokenizer, masker, (), seed=0, epoch=2
    )
    assert other_failures == 0
    assert any(
        not torch.equal(batch_a.encoder_input, batch_b.encoder_input)
        for batch_a, batch_b in zip(batches, other_epoch, strict=True)
    )  # different epochs draw different fresh instances

    other_seed, _ = procedural_epoch_batches(tiny_pool, pool_family("full"), config, tokenizer, masker, (), seed=9, epoch=1)
    assert any(
        not torch.equal(batch_a.encoder_input, batch_b.encoder_input)
        for batch_a, batch_b in zip(batches, other_seed, strict=True)
    )  # the draw is keyed by the run seed too


def test_epoch_batches_augment_variants_and_reject_oversized_cycle(tokenizer, masker, tiny_pool):
    config = _proc_config(augment=True, max_variants=4)
    batches, failures = procedural_epoch_batches(tiny_pool, pool_family("full"), config, tokenizer, masker, (), seed=0, epoch=1)
    assert failures == 0
    for batch in batches:
        # every emitted row is mask-valid padding-free supervision (augment validates on build)
        assert batch.valid.all()
        assert batch.decoder_length.eq(batch.valid.sum(dim=1)).all()

    with pytest.raises(ValueError, match="exceeds the 'full' pool size 6"):
        procedural_epoch_batches(tiny_pool, pool_family("full"), _proc_config(bc_cycle=7), tokenizer, masker, (), seed=0, epoch=1)


def test_procedural_keys_route_and_validate_at_load(tmp_path):
    text = """
[run]
levels = ["tier0_typed_delivery"]

[[stage]]
name = "bc"
kind = "pretrain"
epochs = 3
bc_source = "procedural"
procedural_pool = "ring1"
bc_cycle = 131
"""
    path = tmp_path / "pipeline.toml"
    path.write_text(text)
    stage = load_pipeline(path).stages[0]
    assert isinstance(stage, PretrainStage)
    assert stage.config.bc_source == "procedural"
    assert stage.config.procedural_pool == "ring1"
    assert stage.config.bc_cycle == 131

    # defaults preserve bank mode exactly
    defaults = PretrainConfig()
    assert defaults.bc_source == "bank"
    assert defaults.procedural_pool == "full"

    block = 'bc_source = "procedural"\nprocedural_pool = "ring1"\nbc_cycle = 131\n'
    for bad_block, match in (
        ('bc_source = "fresh"\nprocedural_pool = "ring1"\nbc_cycle = 131\n', "bc_source"),
        ('bc_source = "procedural"\nprocedural_pool = "everything"\nbc_cycle = 131\n', "procedural_pool"),
        ('bc_source = "procedural"\nprocedural_pool = "ring1"\n', "bc_cycle"),
    ):
        path.write_text(text.replace(block, bad_block))
        with pytest.raises(ValueError, match=match):
            load_pipeline(path)


def test_guard_rejects_eval_overlap_and_passes_disjoint_pools(tiny_pool):
    disjoint = anywhere_spec(Hex(0, 0), Hex(3, 3))  # d6 single, not among the tiny pairs
    guard_excludes_eval(tiny_pool, [disjoint], pool_family("full"))
    with pytest.raises(ValueError, match="overlaps the eval bank"):
        guard_excludes_eval(tiny_pool, [tiny_pool[1][2]], pool_family("full"))


def _bank_pair_keys(split: str) -> set[tuple[int, int, int, int]]:
    keys = set()
    for oml in sorted((SINGLE_BANK_DIR / split).glob("v*.oml")):
        coords = {}
        for line in oml.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] in ("entrance", "exit"):
                coords[parts[0]] = (int(parts[2]), int(parts[3]))
        keys.add((*coords["entrance"], *coords["exit"]))
    return keys


def test_full_pool_reproduces_train_bank_and_excludes_eval():
    pool = build_pool("full")
    pool_keys = {pair_key(entrance, exit_) for entrance, exit_, _ in pool}
    assert len(pool) == 525
    # exact reproduction of the tier2_single train split: the 131 eval pairs are
    # excluded by the generator's own split code, not by re-reading the bank
    assert pool_keys == _bank_pair_keys("train")
    assert pool_keys.isdisjoint(_bank_pair_keys("eval"))


def test_pool_compositions_match_spec():
    full = build_pool("full")
    half = build_pool("stratified_half")
    ring1 = build_pool("ring1")

    def by_distance(pool):
        return Counter(hex_distance(entrance, exit_) for entrance, exit_, _ in pool)

    # manifest strata (ring, distance): (1,1)=136, (1,2)=207, (2,4)=96, (3,3)=72, (3,6)=14
    assert by_distance(full) == {1: 136, 2: 207, 3: 72, 4: 96, 6: 14}
    # seeded 50% per distance stratum, matching the static s50 subsample's profile
    assert len(half) == 262
    assert by_distance(half) == {1: 68, 2: 103, 3: 36, 4: 48, 6: 7}
    # ring1 == the d1/d2 pairs: ring-2/3 reference solutions are never seen
    assert by_distance(ring1) == {1: 136, 2: 207}
    assert len(ring1) == 343
    assert set(ring1) <= set(full)
    assert set(half) <= set(full)

    assert build_pool("stratified_half", seed=0) == half  # deterministic
    other_seed = build_pool("stratified_half", seed=1)
    assert len(other_seed) == 262
    assert by_distance(other_seed) == by_distance(half)
    assert other_seed != half

    with pytest.raises(ValueError, match="unknown procedural pool"):
        build_pool("everything")


def test_real_pool_epoch_build_has_zero_failures(tokenizer, masker):
    pool = build_pool("full")
    config = _proc_config(bc_cycle=8)
    batches, failures = procedural_epoch_batches(pool, pool_family("full"), config, tokenizer, masker, (), seed=0, epoch=1)
    assert failures == 0  # every pool pair carries a simulator-verified spec
    assert len(batches) == 8


def _write_pipeline(tmp_path, text):
    path = tmp_path / "pipeline.toml"
    path.write_text(text)
    return path


def _read_stage_rows(run_dir, stage):
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if json.loads(line).get("stage") == stage
    ]


PROCEDURAL_TOML = """
[run]
levels = ["tier0_typed_delivery"]
seed = 0
device = "cpu"

[[stage]]
name = "bc"
kind = "pretrain"
epochs = 2
batch_size = 32
augment = false
eval_interval = 1
bc_source = "procedural"
procedural_pool = "full"
bc_cycle = 3
"""


def test_pipeline_procedural_stage_runs_and_logs_failures(tmp_path, monkeypatch, tiny_pool):
    monkeypatch.setattr(stages, "build_pool", lambda name, *, seed=0: tiny_pool)
    pipeline = load_pipeline(_write_pipeline(tmp_path, PROCEDURAL_TOML))
    pipeline = replace(pipeline, train_config=replace(pipeline.train_config, run_dir=str(tmp_path)))
    entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")]
    eval_entries = [LevelEntry(source=anywhere_spec(Hex(0, 0), Hex(3, 3)), cost_max=50, name="eval_test")]

    run_dir = run_pipeline(pipeline, entries=entries, eval_entries=eval_entries)

    rows = _read_stage_rows(run_dir, "bc")
    assert [row["iteration"] for row in rows] == [1, 2]
    assert all(row["bc_solve_failures"] == 0 for row in rows)  # solver failure rate, logged

    # bank mode (absent keys) runs the unchanged path: no failure key on its rows
    bank_pipeline = load_pipeline(
        _write_pipeline(
            tmp_path, PROCEDURAL_TOML.replace('bc_source = "procedural"\nprocedural_pool = "full"\nbc_cycle = 3\n', "")
        )
    )
    bank_pipeline = replace(
        bank_pipeline, train_config=replace(bank_pipeline.train_config, run_dir=str(tmp_path / "bank"))
    )
    bank_entries = [LevelEntry(source=make_spec(), cost_max=50, name="tier0_test", solutions=(TIER0_OMS,))]
    bank_run = run_pipeline(bank_pipeline, entries=bank_entries)
    assert all("bc_solve_failures" not in row for row in _read_stage_rows(bank_run, "bc"))


# --- tier3_placed pool (placed-family registry entry) ---


def test_tier3_pool_is_train_split_with_verified_specs():
    pool = build_pool("tier3_placed")
    train, eval_ = split_configs()
    assert len(pool) == len(train) == 12
    assert [config for config, _ in pool] == list(train)  # exactly the generator's train split, in emit order
    assert all(not spec.entrypoints_fixed for _, spec in pool)
    assert all(spec == placed_spec(config) for config, spec in pool)
    pool_keys = {pool_family("tier3_placed").entry_key(entry) for entry in pool}
    eval_keys = {(config.atom.name, config.target, config.long_arms) for config in eval_}
    assert not pool_keys & eval_keys
    assert len(pool_keys) == 12  # config identity is unique per entry


def test_tier3_epoch_batches_zero_failures_and_seed_epoch_keying(tokenizer, masker):
    pool = build_pool("tier3_placed")
    family = pool_family("tier3_placed")
    config = _proc_config(bc_cycle=4)
    batches, failures = procedural_epoch_batches(pool, family, config, tokenizer, masker, (), seed=0, epoch=1)
    assert failures == 0
    assert len(batches) == 4

    torch.rand(1000)  # advance the global stream: the draw must not depend on it
    repeated, repeated_failures = procedural_epoch_batches(pool, family, config, tokenizer, masker, (), seed=0, epoch=1)
    assert repeated_failures == 0
    _assert_same_data(batches, repeated)

    other_seed, _ = procedural_epoch_batches(pool, family, config, tokenizer, masker, (), seed=9, epoch=1)
    assert any(
        not torch.equal(batch_a.encoder_input, batch_b.encoder_input)
        for batch_a, batch_b in zip(batches, other_seed, strict=True)
    )


def test_tier3_guard_passes_eval_split_and_raises_on_train_overlap():
    family = pool_family("tier3_placed")
    pool = build_pool("tier3_placed")
    train, eval_ = split_configs()
    guard_excludes_eval(pool, [placed_spec(config) for config in eval_], family)
    with pytest.raises(ValueError, match="overlaps the eval bank"):
        guard_excludes_eval(pool, [placed_spec(train[0])], family)


def test_tier3_pool_routes_and_validates_at_load(tmp_path):
    text = """
[run]
levels = ["tier0_typed_delivery"]

[[stage]]
name = "bc"
kind = "pretrain"
epochs = 1
bc_source = "procedural"
procedural_pool = "tier3_placed"
bc_cycle = 4
"""
    path = tmp_path / "pipeline.toml"
    path.write_text(text)
    stage = load_pipeline(path).stages[0]
    assert isinstance(stage, PretrainStage)
    assert stage.config.bc_source == "procedural"
    assert stage.config.procedural_pool == "tier3_placed"
    assert stage.config.bc_cycle == 4

    path.write_text(text.replace('procedural_pool = "tier3_placed"', 'procedural_pool = "everything"'))
    with pytest.raises(ValueError, match="procedural_pool"):
        load_pipeline(path)
