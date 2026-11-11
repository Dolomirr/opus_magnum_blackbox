"""Tier-1 bank generator: caps guards, deterministic emission, on-disk validation."""

import tomllib

import pytest
from opus_magnum_sym import Hex

from opus_magnum_blackbox import gen_tier1
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.families import FamilyError, single_atom_spec, verify
from opus_magnum_blackbox.solutions import load_solutions

SMALL_TRAIN, SMALL_EVAL = 4, 2


@pytest.fixture(scope="module")
def small_bank(tmp_path_factory):
    params = gen_tier1.BankParams(seed=3, train_count=SMALL_TRAIN, eval_count=SMALL_EVAL)
    return gen_tier1.generate_bank(tmp_path_factory.mktemp("tier1_bank"), params=params)


# --- caps guards (families.single_atom_spec) ---


def test_target_over_program_cap_raises():
    with pytest.raises(FamilyError, match="program-action cap"):
        single_atom_spec(Hex(0, 0), Hex(0, 1), target=5)


def test_steps_over_tick_cap_raises():
    with pytest.raises(FamilyError, match="tick-token cap"):
        single_atom_spec(Hex(0, 0), Hex(0, 1), steps=65)


def test_program_longer_than_budget_raises():
    with pytest.raises(FamilyError, match="does not fit"):
        single_atom_spec(Hex(0, 0), Hex(0, 1), target=4, steps=32)  # ticks 1..32 need steps > 32


def test_caps_boundaries_pass():
    """Target 4 is exactly on the program-action cap with one spare step."""
    spec = single_atom_spec(Hex(0, 0), Hex(0, 1), target=4, steps=33)
    assert verify(spec) > 0


# --- bank emission ---


def test_generation_is_deterministic(tmp_path):
    params = gen_tier1.BankParams(seed=7, train_count=3, eval_count=2)
    first = gen_tier1.generate_bank(tmp_path / "one", params=params)
    second = gen_tier1.generate_bank(tmp_path / "two", params=params)
    names_one = sorted(p.relative_to(first.root).as_posix() for p in first.root.rglob("*") if p.is_file())
    names_two = sorted(p.relative_to(second.root).as_posix() for p in second.root.rglob("*") if p.is_file())
    assert names_one == names_two
    for name in names_one:
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes(), name


def test_regeneration_overwrites_in_place(tmp_path):
    root = tmp_path / "bank"
    gen_tier1.generate_bank(root, params=gen_tier1.BankParams(seed=5, train_count=4, eval_count=2))
    bank = gen_tier1.generate_bank(root, params=gen_tier1.BankParams(seed=5, train_count=2, eval_count=2))
    assert [variant.oml.name for variant in bank.train] == ["v00.oml", "v01.oml"]
    assert sorted(p.name for p in (root / "train").iterdir()) == ["v00.oml", "v00.oms", "v01.oml", "v01.oms"]


def test_sampler_rejects_impossible_counts(tmp_path):
    with pytest.raises(ValueError, match="only has"):
        gen_tier1.generate_bank(tmp_path, params=gen_tier1.BankParams(seed=0, train_count=1000, eval_count=0))


@pytest.mark.parametrize("index", range(SMALL_TRAIN + SMALL_EVAL))
def test_generated_pair_loads_and_solves(small_bank, index):
    variant = small_bank.variants[index]
    entry = LevelEntry(
        source=variant.oml,
        cost_max=gen_tier1.MANIFEST_COST_MAX,
        name=variant.oml.stem,
        solutions=(variant.oms,),
    )
    tokens = load_solutions(entry)
    assert len(tokens) == 1
    assert tokens[0]


def test_train_eval_disjoint_with_matching_manifest(small_bank):
    train_keys = {variant.key for variant in small_bank.train}
    eval_keys = {variant.key for variant in small_bank.eval}
    assert len(small_bank.train) == SMALL_TRAIN
    assert len(small_bank.eval) == SMALL_EVAL
    assert not train_keys & eval_keys
    manifest = tomllib.loads(small_bank.manifest.read_text())
    assert manifest["cost_max"] == gen_tier1.MANIFEST_COST_MAX
    assert manifest["weight"] == gen_tier1.MANIFEST_WEIGHT
    provenance = manifest["provenance"]
    assert provenance["seed"] == 3
    assert provenance["train_count"] == SMALL_TRAIN
    assert provenance["eval_count"] == SMALL_EVAL
    assert provenance["target"] == 2
    assert provenance["steps"] == 30
    assert provenance["size"] == [6, 6]
    assert provenance["atom"] == "water"


# --- the checked-in repo bank ---


def test_repo_bank_matches_manifest():
    manifest = tomllib.loads((gen_tier1.BANK_DIR / "manifest.toml").read_text())
    for split, count in (
        ("train", manifest["provenance"]["train_count"]),
        ("eval", manifest["provenance"]["eval_count"]),
    ):
        omls = sorted((gen_tier1.BANK_DIR / split).glob("v*.oml"))
        assert len(omls) == count, split
        assert all(oml.with_suffix(".oms").exists() for oml in omls), split


@pytest.mark.parametrize("relative", ["train/v00", "train/v24", "train/v47", "eval/v15"])
def test_repo_bank_subset_solves(relative):
    oml = gen_tier1.BANK_DIR / f"{relative}.oml"
    entry = LevelEntry(
        source=oml,
        cost_max=gen_tier1.MANIFEST_COST_MAX,
        name=oml.stem,
        solutions=(gen_tier1.BANK_DIR / f"{relative}.oms",),
    )
    assert load_solutions(entry)[0]
