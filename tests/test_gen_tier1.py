"""Tier-1 bank generator: caps guards, deterministic emission, on-disk validation, enumerate/split modes."""

import random
import tomllib

import pytest
from opus_magnum_sym import Hex

from opus_magnum_blackbox import gen_tier1
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.families import FamilyError, hex_distance, single_atom_spec, verify
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
    assert [variant.oml.name for variant in bank.train] == ["v000.oml", "v001.oml"]
    assert sorted(p.name for p in (root / "train").iterdir()) == [
        "v000.oml",
        "v000.oms",
        "v001.oml",
        "v001.oms",
    ]


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
    assert provenance["mode"] == "sample"
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


@pytest.mark.parametrize("relative", ["train/v000", "train/v171", "train/v341", "eval/v085"])
def test_repo_bank_subset_solves(relative):
    oml = gen_tier1.BANK_DIR / f"{relative}.oml"
    entry = LevelEntry(
        source=oml,
        cost_max=gen_tier1.MANIFEST_COST_MAX,
        name=oml.stem,
        solutions=(gen_tier1.BANK_DIR / f"{relative}.oms",),
    )
    assert load_solutions(entry)[0]


def test_repo_bank_manifest_distances_match_files():
    """
    The manifest's stratified-split breakdown must be derivable from the emitted
    files themselves: per distance class and split, plus the totals.
    """
    manifest = tomllib.loads((gen_tier1.BANK_DIR / "manifest.toml").read_text())
    provenance = manifest["provenance"]
    assert provenance["mode"] == "enumerate"
    actual: dict[str, dict[str, int]] = {}
    for split in ("train", "eval"):
        for oml in sorted((gen_tier1.BANK_DIR / split).glob("v*.oml")):
            coords = {}
            for line in oml.read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in ("entrance", "exit"):
                    coords[parts[0]] = (int(parts[2]), int(parts[3]))
            distance = hex_distance(Hex(*coords["entrance"]), Hex(*coords["exit"]))
            counts = actual.setdefault(str(distance), {"train": 0, "eval": 0})
            counts[split] += 1
    assert actual == {str(distance): counts for distance, counts in provenance["distances"].items()}
    assert sum(counts["train"] for counts in actual.values()) == provenance["train_count"]
    assert sum(counts["eval"] for counts in actual.values()) == provenance["eval_count"]


# --- enumerate mode: the full verified task space ---


def _hex_pairs(pairs: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[tuple[Hex, Hex]]:
    return [(Hex(*entrance), Hex(*exit)) for entrance, exit in pairs]


def test_enumerate_pairs_exact_set_on_tiny_board():
    classes, rejected = gen_tier1.enumerate_pairs(size=(2, 2))
    expected_d1 = [
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (0, 0)),
        ((0, 1), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (0, 0)),
        ((1, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (0, 1)),
        ((1, 1), (1, 0)),
    ]
    expected_d2 = [((0, 0), (1, 1)), ((1, 1), (0, 0))]
    assert classes[1] == _hex_pairs(expected_d1)
    assert classes[2] == _hex_pairs(expected_d2)
    assert rejected == {1: 0, 2: 0}


def test_enumerate_pairs_rejects_unverifiable_pairs():
    """
    A 1-row board has no in-board cell adjacent to both endpoints, so no pair
    gets an arm base: everything at distance 1-2 counts as rejected, nothing raises.
    """
    classes, rejected = gen_tier1.enumerate_pairs(size=(1, 2))
    assert classes == {1: [], 2: []}
    assert rejected == {1: 2, 2: 0}


def _fake_classes() -> dict[int, list[tuple[Hex, Hex]]]:
    return {
        1: [(Hex(0, i), Hex(0, i + 1)) for i in range(10)],
        2: [(Hex(1, i), Hex(1, i + 2)) for i in range(30)],
    }


def test_stratified_split_deterministic_disjoint_stratified():
    classes = _fake_classes()
    first = gen_tier1.stratified_split(classes, random.Random(5), eval_fraction=0.2)
    assert gen_tier1.stratified_split(classes, random.Random(5), eval_fraction=0.2) == first  # deterministic
    train, eval_ = first
    train_keys = {(entrance.r, entrance.s, exit.r, exit.s) for entrance, exit in train}
    eval_keys = {(entrance.r, entrance.s, exit.r, exit.s) for entrance, exit in eval_}
    assert not train_keys & eval_keys  # disjoint
    # ~20% of every class to eval; both classes land in both splits
    assert len(train) == 32
    assert len(eval_) == 8
    assert sum(1 for entrance, _ in eval_ if entrance.r == 0) == 2
    assert sum(1 for entrance, _ in eval_ if entrance.r == 1) == 6
    assert sum(1 for entrance, _ in train if entrance.r == 0) == 8
    assert sum(1 for entrance, _ in train if entrance.r == 1) == 24


def test_enumerate_bank_is_deterministic(tmp_path):
    params = gen_tier1.BankParams(mode="enumerate", seed=11, size=(2, 2))
    first = gen_tier1.generate_bank(tmp_path / "one", params=params)
    second = gen_tier1.generate_bank(tmp_path / "two", params=params)
    names = sorted(p.relative_to(first.root).as_posix() for p in first.root.rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(second.root).as_posix() for p in second.root.rglob("*") if p.is_file())
    for name in names:
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes(), name


def test_enumerate_bank_split_and_manifest(tmp_path):
    bank = gen_tier1.generate_bank(tmp_path, params=gen_tier1.BankParams(mode="enumerate", seed=11, size=(2, 2)))
    # 2x2 board: 10 distance-1 pairs -> 2 eval, 2 distance-2 pairs -> round(0.4) = 0 eval
    assert len(bank.train) == 10
    assert len(bank.eval) == 2
    train_keys = {variant.key for variant in bank.train}
    eval_keys = {variant.key for variant in bank.eval}
    assert not train_keys & eval_keys
    assert bank.rejected == {1: 0, 2: 0}

    manifest = tomllib.loads(bank.manifest.read_text())
    provenance = manifest["provenance"]
    assert provenance["mode"] == "enumerate"
    assert provenance["seed"] == 11
    assert provenance["train_count"] == 10
    assert provenance["eval_count"] == 2
    assert provenance["eval_fraction"] == 0.2
    assert provenance["distances"] == {"1": {"train": 8, "eval": 2}, "2": {"train": 2, "eval": 0}}

    for variant in bank.variants:
        entry = LevelEntry(
            source=variant.oml, cost_max=gen_tier1.MANIFEST_COST_MAX, name=variant.oml.stem, solutions=(variant.oms,)
        )
        assert load_solutions(entry)[0]


def test_enumerate_bank_without_verifiable_pairs_raises(tmp_path):
    with pytest.raises(ValueError, match="kept no pairs"):
        gen_tier1.generate_bank(tmp_path, params=gen_tier1.BankParams(mode="enumerate", size=(1, 2)))


def test_unknown_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="bogus"):
        gen_tier1.generate_bank(tmp_path, params=gen_tier1.BankParams(mode="bogus"))
