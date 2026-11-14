"""Tier-2 bank generator: stratified quotas, deterministic emission, on-disk validation, reject handling."""

import random
import tomllib

import pytest
from opus_magnum_sym import Hex

from opus_magnum_blackbox import gen_tier2
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.families import hex_distance
from opus_magnum_blackbox.solutions import load_solutions

SMALL_TRAIN, SMALL_EVAL = 6, 3


@pytest.fixture(scope="module")
def small_bank(tmp_path_factory):
    params = gen_tier2.BankParams(
        seed=5, train_single=4, train_handoff=2, eval_single=2, eval_handoff=1, size=(3, 3)
    )
    return gen_tier2.generate_bank(tmp_path_factory.mktemp("tier2_bank"), params=params)


# --- quota math ---


def test_apportion_proportional_capped_deterministic():
    strata = {(1, "single"), (2, "single"), (3, "handoff")}
    # exact split, no leftovers
    assert gen_tier2._apportion(4, {(1, "single"): 3, (2, "single"): 1}) == {(1, "single"): 3, (2, "single"): 1}
    # largest remainder gets the leftover; ties break by stratum order
    assert gen_tier2._apportion(1, dict.fromkeys(strata, 1)) == {(1, "single"): 1, (2, "single"): 0, (3, "handoff"): 0}
    # shares cap at capacity; the clamped total shrinks with it
    assert gen_tier2._apportion(10, {(1, "single"): 2, (2, "single"): 9}) == {(1, "single"): 2, (2, "single"): 8}
    assert gen_tier2._apportion(50, {(1, "single"): 2, (2, "single"): 3}) == {(1, "single"): 2, (2, "single"): 3}
    # degenerate capacities
    assert gen_tier2._apportion(0, dict.fromkeys(strata, 5)) == dict.fromkeys(strata, 0)
    assert gen_tier2._apportion(5, {}) == {}


def _fake_pair(entrance: tuple[int, int], exit_: tuple[int, int]) -> gen_tier2.ClassifiedPair:
    return (Hex(*entrance), Hex(*exit_), None)  # spec stand-in: sampling never touches it


def _fake_strata() -> gen_tier2.StratumPairs:
    return {
        (1, gen_tier2.SINGLE): [_fake_pair((0, i), (0, i + 1)) for i in range(4)],
        (2, gen_tier2.SINGLE): [_fake_pair((1, i), (1, i + 2)) for i in range(6)],
        (3, gen_tier2.HANDOFF): [_fake_pair((2, i), (2, i + 3)) for i in range(4)],
    }


def test_stratify_quotas_proportional_within_class():
    strata = _fake_strata()
    params = gen_tier2.BankParams(train_single=4, train_handoff=2, eval_single=2, eval_handoff=1)
    quotas = gen_tier2.stratify_quotas({key: len(pairs) for key, pairs in strata.items()}, params)
    assert quotas == {
        (1, gen_tier2.SINGLE): (2, 1),
        (2, gen_tier2.SINGLE): (2, 1),
        (3, gen_tier2.HANDOFF): (2, 1),
    }


def test_stratified_sample_quota_exact_disjoint_deterministic():
    strata = _fake_strata()
    quotas = gen_tier2.stratify_quotas(
        {key: len(pairs) for key, pairs in strata.items()},
        gen_tier2.BankParams(train_single=4, train_handoff=2, eval_single=2, eval_handoff=1),
    )
    first = gen_tier2.stratified_sample(strata, quotas, random.Random(9))
    assert gen_tier2.stratified_sample(strata, quotas, random.Random(9)) == first
    train, eval_ = first
    assert len(train) == 6
    assert len(eval_) == 3
    train_keys = {(pair[0].r, pair[0].s, pair[1].r, pair[1].s) for pair in train}
    eval_keys = {(pair[0].r, pair[0].s, pair[1].r, pair[1].s) for pair in eval_}
    assert not train_keys & eval_keys
    per_distance: dict[int, dict[str, int]] = {}
    for split, pairs in (("train", train), ("eval", eval_)):
        for entrance, exit_, _ in pairs:
            counts = per_distance.setdefault(hex_distance(entrance, exit_), {"train": 0, "eval": 0})
            counts[split] += 1
    assert per_distance == {1: {"train": 2, "eval": 1}, 2: {"train": 2, "eval": 1}, 3: {"train": 2, "eval": 1}}


# --- classification ---


def test_classify_counts_rejects_without_raising():
    """A 1x2 board has no third cell for an arm base: both pairs reject, nothing raises."""
    strata, rejected = gen_tier2.classify_pairs(size=(1, 2))
    assert strata == {}
    assert rejected == {1: 2}


def test_classification_separates_classes():
    strata, rejected = gen_tier2.classify_pairs(size=(3, 3))
    assert rejected == {}
    assert {stratum: len(pairs) for stratum, pairs in sorted(strata.items())} == {
        (1, gen_tier2.SINGLE): 32,
        (2, gen_tier2.SINGLE): 30,
        (3, gen_tier2.HANDOFF): 8,
        (4, gen_tier2.SINGLE): 2,
    }
    for (_, kind), pairs in strata.items():
        for _, _, spec in pairs:
            assert spec.target == (1 if kind == gen_tier2.HANDOFF else 2)
    assert all(spec is not None for pairs in strata.values() for _, _, spec in pairs)


# --- bank emission ---


def test_generation_is_deterministic(tmp_path):
    params = gen_tier2.BankParams(seed=7, train_single=2, train_handoff=1, eval_single=1, eval_handoff=1, size=(3, 3))
    first = gen_tier2.generate_bank(tmp_path / "one", params=params)
    second = gen_tier2.generate_bank(tmp_path / "two", params=params)
    names_one = sorted(p.relative_to(first.root).as_posix() for p in first.root.rglob("*") if p.is_file())
    names_two = sorted(p.relative_to(second.root).as_posix() for p in second.root.rglob("*") if p.is_file())
    assert names_one == names_two
    for name in names_one:
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes(), name


def test_regeneration_overwrites_in_place(tmp_path):
    root = tmp_path / "bank"
    full = gen_tier2.BankParams(seed=5, train_single=4, train_handoff=2, eval_single=2, eval_handoff=1, size=(3, 3))
    gen_tier2.generate_bank(root, params=full)
    smaller = gen_tier2.BankParams(seed=5, train_single=2, train_handoff=1, eval_single=1, eval_handoff=1, size=(3, 3))
    bank = gen_tier2.generate_bank(root, params=smaller)
    assert [variant.oml.name for variant in bank.train] == ["v000.oml", "v001.oml", "v002.oml"]
    assert sorted(p.name for p in (root / "train").iterdir()) == [
        "v000.oml",
        "v000.oms",
        "v001.oml",
        "v001.oms",
        "v002.oml",
        "v002.oms",
    ]


def test_generate_bank_without_verifiable_pairs_raises(tmp_path):
    with pytest.raises(ValueError, match="kept no pairs"):
        gen_tier2.generate_bank(tmp_path, params=gen_tier2.BankParams(size=(1, 2)))


def test_bank_generation_survives_unsolvable_pairs(tmp_path):
    """
    2x3 at steps 14: single cycles no longer fit the budget, the two d1 pairs have no
    plan left and are counted as rejects — and the handoff-only bank still generates.
    """
    params = gen_tier2.BankParams(
        seed=2, train_single=4, train_handoff=6, eval_single=2, eval_handoff=3, steps=14, size=(2, 3)
    )
    bank = gen_tier2.generate_bank(tmp_path, params=params)
    assert bank.rejected == {1: 2}
    # no singles exist at steps 14: their quotas stay unmet
    assert len(bank.train) == 6
    assert len(bank.eval) == 3
    assert all(variant.kind == gen_tier2.HANDOFF for variant in bank.variants)
    manifest = tomllib.loads(bank.manifest.read_text())
    assert manifest["provenance"]["rejected"] == 2
    assert manifest["provenance"]["classes"] == {
        "single": {"train": 0, "eval": 0},
        "handoff": {"train": 6, "eval": 3},
    }


@pytest.mark.parametrize("index", range(SMALL_TRAIN + SMALL_EVAL))
def test_generated_pair_loads_and_solves(small_bank, index):
    variant = small_bank.variants[index]
    entry = LevelEntry(
        source=variant.oml,
        cost_max=gen_tier2.MANIFEST_COST_MAX,
        name=variant.oml.stem,
        solutions=(variant.oms,),
    )
    tokens = load_solutions(entry)
    assert len(tokens) == 1
    assert tokens[0]


def test_train_eval_disjoint_with_matching_manifest(small_bank):
    assert len(small_bank.train) == SMALL_TRAIN
    assert len(small_bank.eval) == SMALL_EVAL
    train_keys = {variant.key for variant in small_bank.train}
    eval_keys = {variant.key for variant in small_bank.eval}
    assert not train_keys & eval_keys
    assert sum(1 for variant in small_bank.train if variant.kind == gen_tier2.SINGLE) == 4
    assert sum(1 for variant in small_bank.train if variant.kind == gen_tier2.HANDOFF) == 2
    assert sum(1 for variant in small_bank.eval if variant.kind == gen_tier2.SINGLE) == 2
    assert sum(1 for variant in small_bank.eval if variant.kind == gen_tier2.HANDOFF) == 1
    manifest = tomllib.loads(small_bank.manifest.read_text())
    assert manifest["cost_max"] == gen_tier2.MANIFEST_COST_MAX
    assert manifest["weight"] == gen_tier2.MANIFEST_WEIGHT
    provenance = manifest["provenance"]
    assert provenance["mode"] == "sample"
    assert provenance["seed"] == 5
    assert provenance["train_count"] == SMALL_TRAIN
    assert provenance["eval_count"] == SMALL_EVAL
    assert provenance["steps"] == 30
    assert provenance["size"] == [3, 3]
    assert provenance["atom"] == "water"
    assert provenance["rejected"] == 0
    assert provenance["quotas"] == {
        "train_single": 4,
        "train_handoff": 2,
        "eval_single": 2,
        "eval_handoff": 1,
    }
    assert provenance["classes"] == {
        "single": {"train": 4, "eval": 2},
        "handoff": {"train": 2, "eval": 1},
    }


def test_quotas_apportioned_from_populations(small_bank):
    """Emitted per-stratum counts equal the quotas re-derived from a fresh classification sweep."""
    params = gen_tier2.BankParams(
        seed=5, train_single=4, train_handoff=2, eval_single=2, eval_handoff=1, size=(3, 3)
    )
    strata, _ = gen_tier2.classify_pairs(size=(3, 3))
    populations = {stratum: len(pairs) for stratum, pairs in strata.items()}
    assert gen_tier2.stratify_quotas(populations, params) == small_bank.quotas
    breakdown = gen_tier2._stratum_breakdown(small_bank.train, small_bank.eval)
    for stratum, (train_quota, eval_quota) in small_bank.quotas.items():
        counts = breakdown.get(stratum, {"train": 0, "eval": 0})
        assert counts["train"] == train_quota, stratum
        assert counts["eval"] == eval_quota, stratum


def _flat_strata(nested: dict) -> dict[str, dict[str, int]]:
    """`provenance.strata` as parsed from TOML (`{"1": {"single": ...}}`) flattened to `"1.single"` keys."""
    return {
        f"{distance}.{kind}": counts
        for distance, kinds in nested.items()
        for kind, counts in kinds.items()
    }


def test_manifest_strata_match_files(small_bank):
    """
    The manifest's per-stratum counts must be derivable from the emitted files
    themselves: per (distance, class) and split, plus the class and grand totals.
    """
    manifest = tomllib.loads(small_bank.manifest.read_text())
    provenance = manifest["provenance"]
    actual: dict[str, dict[str, int]] = {}
    for split in ("train", "eval"):
        for oml in sorted((small_bank.root / split).glob("v*.oml")):
            coords, target = {}, None
            for line in oml.read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in ("entrance", "exit"):
                    coords[parts[0]] = (int(parts[2]), int(parts[3]))
                if parts and parts[0] == "target":
                    target = int(parts[1])
            distance = hex_distance(Hex(*coords["entrance"]), Hex(*coords["exit"]))
            stratum = f"{distance}.{'handoff' if target == 1 else 'single'}"
            counts = actual.setdefault(stratum, {"train": 0, "eval": 0})
            counts[split] += 1
    assert _flat_strata(provenance["strata"]) == actual
    class_sums = {"single": {"train": 0, "eval": 0}, "handoff": {"train": 0, "eval": 0}}
    for stratum, counts in actual.items():
        kind = stratum.split(".", 1)[1]
        for split in ("train", "eval"):
            class_sums[kind][split] += counts[split]
    assert class_sums == provenance["classes"]
    assert sum(counts["train"] for counts in actual.values()) == provenance["train_count"]
    assert sum(counts["eval"] for counts in actual.values()) == provenance["eval_count"]


# --- the checked-in repo bank ---


def test_repo_bank_matches_manifest():
    manifest = tomllib.loads((gen_tier2.BANK_DIR / "manifest.toml").read_text())
    for split, count in (
        ("train", manifest["provenance"]["train_count"]),
        ("eval", manifest["provenance"]["eval_count"]),
    ):
        omls = sorted((gen_tier2.BANK_DIR / split).glob("v*.oml"))
        assert len(omls) == count, split
        assert all(oml.with_suffix(".oms").exists() for oml in omls), split


def test_repo_bank_manifest_strata_match_files():
    provenance = tomllib.loads((gen_tier2.BANK_DIR / "manifest.toml").read_text())["provenance"]
    actual: dict[str, dict[str, int]] = {}
    for split in ("train", "eval"):
        for oml in sorted((gen_tier2.BANK_DIR / split).glob("v*.oml")):
            coords, target = {}, None
            for line in oml.read_text().splitlines():
                parts = line.split()
                if parts and parts[0] in ("entrance", "exit"):
                    coords[parts[0]] = (int(parts[2]), int(parts[3]))
                if parts and parts[0] == "target":
                    target = int(parts[1])
            distance = hex_distance(Hex(*coords["entrance"]), Hex(*coords["exit"]))
            stratum = f"{distance}.{'handoff' if target == 1 else 'single'}"
            counts = actual.setdefault(stratum, {"train": 0, "eval": 0})
            counts[split] += 1
    assert _flat_strata(provenance["strata"]) == actual
    assert sum(counts["train"] for counts in actual.values()) == provenance["train_count"]
    assert sum(counts["eval"] for counts in actual.values()) == provenance["eval_count"]


@pytest.mark.parametrize("relative", ["train/v000", "train/v449", "eval/v000", "eval/v119"])
def test_repo_bank_subset_solves(relative):
    oml = gen_tier2.BANK_DIR / f"{relative}.oml"
    entry = LevelEntry(
        source=oml,
        cost_max=gen_tier2.MANIFEST_COST_MAX,
        name=oml.stem,
        solutions=(gen_tier2.BANK_DIR / f"{relative}.oms",),
    )
    assert load_solutions(entry)[0]
