"""Tier-2 bank generator: stratified quotas, deterministic emission, on-disk validation, reject handling."""

import random
import tomllib
from pathlib import Path

import pytest
from opus_magnum_sym import Hex

from opus_magnum_blackbox import gen_tier2
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.families import hex_distance
from opus_magnum_blackbox.solutions import load_solutions

SMALL_TRAIN, SMALL_EVAL = 6, 3
CLASS_EVAL_PERCENT = 25


@pytest.fixture(scope="module")
def small_bank(tmp_path_factory):
    params = gen_tier2.BankParams(seed=5, train_single=4, train_handoff=2, eval_single=2, eval_handoff=1, size=(3, 3))
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
    params = gen_tier2.BankParams(seed=5, train_single=4, train_handoff=2, eval_single=2, eval_handoff=1, size=(3, 3))
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
    return {f"{distance}.{kind}": counts for distance, kinds in nested.items() for kind, counts in kinds.items()}


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


# --- per-class enumerated banks ---


def test_enum_quotas_half_up_apportioned():
    """Half-up eval share of the whole class, apportioned over strata by largest remainder."""
    assert gen_tier2.enum_quotas({(gen_tier2.SINGLE, 1, 1): 656}, 20) == {(gen_tier2.SINGLE, 1, 1): (525, 131)}
    assert gen_tier2.enum_quotas({(gen_tier2.HANDOFF, 0, 3): 604}, 20) == {(gen_tier2.HANDOFF, 0, 3): (483, 121)}
    quotas = gen_tier2.enum_quotas({(gen_tier2.SINGLE, 1, 1): 3, (gen_tier2.SINGLE, 2, 1): 7}, 20)
    assert quotas == {(gen_tier2.SINGLE, 1, 1): (2, 1), (gen_tier2.SINGLE, 2, 1): (6, 1)}
    assert gen_tier2.enum_quotas({}, 20) == {}
    assert gen_tier2.enum_quotas({(gen_tier2.SINGLE, 1, 1): 10}, 0) == {(gen_tier2.SINGLE, 1, 1): (10, 0)}


@pytest.fixture(scope="module")
def small_class_strata():
    return gen_tier2.classify_class_pairs(size=(3, 3))


def test_class_classification_separates_ring_strata(small_class_strata):
    """3x3: singles key the winning ring (`arm` -> 1, `arm_l2` -> 2), handoffs pin ring 0."""
    strata, rejected = small_class_strata
    assert rejected == {}
    assert {stratum: len(pairs) for stratum, pairs in sorted(strata.items())} == {
        (gen_tier2.SINGLE, 1, 1): 32,
        (gen_tier2.SINGLE, 1, 2): 30,
        (gen_tier2.SINGLE, 2, 4): 2,
        (gen_tier2.HANDOFF, 0, 3): 8,
    }
    for (kind, ring, distance), pairs in strata.items():
        assert ring >= (1 if kind == gen_tier2.SINGLE else 0)
        for entrance, exit_, spec in pairs:
            assert hex_distance(entrance, exit_) == distance
            assert spec.target == (2 if kind == gen_tier2.SINGLE else 1)


def test_class_classification_agrees_with_flat_grouping():
    """Regrouping class strata by (distance, kind) reproduces the flat classification's pair sets."""
    class_strata, _ = gen_tier2.classify_class_pairs(size=(3, 3))
    flat, _ = gen_tier2.classify_pairs(size=(3, 3))
    regrouped: dict[tuple[int, str], set[tuple[int, int, int, int]]] = {}
    for (kind, _, distance), pairs in class_strata.items():
        keys = regrouped.setdefault((distance, kind), set())
        keys.update((entrance.r, entrance.s, exit_.r, exit_.s) for entrance, exit_, _ in pairs)
    flat_keys = {
        stratum: {(entrance.r, entrance.s, exit_.r, exit_.s) for entrance, exit_, _ in pairs}
        for stratum, pairs in flat.items()
    }
    assert regrouped == flat_keys


@pytest.fixture(scope="module")
def small_class_banks(tmp_path_factory):
    params = gen_tier2.EnumBankParams(seed=5, eval_percent=CLASS_EVAL_PERCENT, size=(3, 3))
    return gen_tier2.generate_class_banks(tmp_path_factory.mktemp("tier2_class_banks"), params=params)


def test_class_banks_partition_classified_pairs(small_class_banks, small_class_strata):
    """Splits are disjoint and union-complete per class; the banks cover every classified pair exactly."""
    strata, rejected = small_class_strata
    assert rejected == {}
    classified: dict[str, set] = {}
    for (kind, _, _), pairs in strata.items():
        classified.setdefault(kind, set()).update(
            (entrance.r, entrance.s, exit_.r, exit_.s) for entrance, exit_, _ in pairs
        )
    bank_keys: dict[str, set] = {}
    for kind in (gen_tier2.SINGLE, gen_tier2.HANDOFF):
        bank = small_class_banks[gen_tier2.CLASS_BANK_NAMES[kind]]
        assert all(variant.kind == kind for variant in bank.variants)
        train_keys = {variant.key for variant in bank.train}
        eval_keys = {variant.key for variant in bank.eval}
        assert not train_keys & eval_keys
        assert train_keys | eval_keys == classified[kind]
        bank_keys[kind] = train_keys | eval_keys
    assert not bank_keys[gen_tier2.SINGLE] & bank_keys[gen_tier2.HANDOFF]


def test_class_bank_quotas_match_populations(small_class_banks, small_class_strata):
    """Emitted per-stratum counts equal the quotas re-derived from a fresh classification sweep."""
    strata, _ = small_class_strata
    for kind in (gen_tier2.SINGLE, gen_tier2.HANDOFF):
        bank = small_class_banks[gen_tier2.CLASS_BANK_NAMES[kind]]
        populations = {key: len(pairs) for key, pairs in strata.items() if key[0] == kind}
        assert bank.quotas == gen_tier2.enum_quotas(populations, CLASS_EVAL_PERCENT)


def test_class_bank_manifests_match_files(small_class_banks):
    """Per-bank provenance: counts, mode, eval share, and per-stratum tables derived from the files themselves."""
    for kind in (gen_tier2.SINGLE, gen_tier2.HANDOFF):
        bank = small_class_banks[gen_tier2.CLASS_BANK_NAMES[kind]]
        manifest = tomllib.loads(bank.manifest.read_text())
        assert manifest["cost_max"] == gen_tier2.MANIFEST_COST_MAX
        assert manifest["weight"] == gen_tier2.MANIFEST_WEIGHT
        provenance = manifest["provenance"]
        assert provenance["mode"] == "enumerate"
        assert provenance["seed"] == 5
        assert provenance["eval_percent"] == CLASS_EVAL_PERCENT
        assert provenance["train_count"] == len(bank.train)
        assert provenance["eval_count"] == len(bank.eval)
        assert provenance["rejected"] == 0
        assert provenance["steps"] == 30
        assert provenance["size"] == [3, 3]
        assert provenance["atom"] == "water"
        assert _flat_class_strata(provenance["strata"], kind) == _file_stratum_counts(bank.root)


def test_class_bank_generation_is_deterministic(tmp_path, small_class_banks):
    banks = gen_tier2.generate_class_banks(
        tmp_path, params=gen_tier2.EnumBankParams(seed=5, eval_percent=CLASS_EVAL_PERCENT, size=(3, 3))
    )
    for name, bank in small_class_banks.items():
        names = sorted(path.relative_to(bank.root).as_posix() for path in bank.root.rglob("*") if path.is_file())
        other = sorted(
            path.relative_to(banks[name].root).as_posix() for path in banks[name].root.rglob("*") if path.is_file()
        )
        assert names == other
        for rel in names:
            assert (bank.root / rel).read_bytes() == (banks[name].root / rel).read_bytes(), rel


def test_class_bank_regeneration_overwrites_in_place(tmp_path):
    root = tmp_path / "banks"
    gen_tier2.generate_class_banks(root, params=gen_tier2.EnumBankParams(seed=5, eval_percent=25, size=(3, 3)))
    banks = gen_tier2.generate_class_banks(root, params=gen_tier2.EnumBankParams(seed=5, eval_percent=10, size=(3, 3)))
    single = banks["tier2_single"]
    assert (len(single.train), len(single.eval)) == (58, 6)
    assert sorted(path.name for path in (single.root / "eval").iterdir()) == [
        "v000.oml",
        "v000.oms",
        "v001.oml",
        "v001.oms",
        "v002.oml",
        "v002.oms",
        "v003.oml",
        "v003.oms",
        "v004.oml",
        "v004.oms",
        "v005.oml",
        "v005.oms",
    ]


def _flat_class_strata(nested: dict, kind: str) -> dict[str, dict[str, int]]:
    """`provenance.strata` as parsed from TOML, flattened to file-derivable keys: `"ring.distance"` for singles, `"distance"` for handoffs."""
    if kind == gen_tier2.SINGLE:
        return {
            f"{ring}.{distance}": counts for ring, distances in nested.items() for distance, counts in distances.items()
        }
    return dict(nested)


def _variant_stratum(oml: Path) -> str:
    """File-derivable stratum key of one variant: the winning ring from the `.oms` arm placement, distance from the `.oml`."""
    coords, target = {}, None
    for line in oml.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in ("entrance", "exit"):
            coords[parts[0]] = (int(parts[2]), int(parts[3]))
        if parts and parts[0] == "target":
            target = int(parts[1])
    distance = hex_distance(Hex(*coords["entrance"]), Hex(*coords["exit"]))
    if target == 1:
        return f"{distance}"
    oms = oml.with_suffix(".oms")
    for line in oms.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "place" and parts[1].startswith("arm"):
            ring = 1 if parts[1] == "arm" else int(parts[1].removeprefix("arm_l"))
            return f"{ring}.{distance}"
    raise AssertionError(f"{oms.name}: no arm placement")


def _file_stratum_counts(root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "eval"):
        for oml in sorted((root / split).glob("v*.oml")):
            entry = counts.setdefault(_variant_stratum(oml), {"train": 0, "eval": 0})
            entry[split] += 1
    return counts


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


# --- the checked-in per-class repo banks ---


def test_repo_class_banks_counts_and_manifest():
    """The enumerated banks hold all 656 single + 604 handoff pairs, cut ~80/20 per the half-up eval share."""
    for bank_dir, total in ((gen_tier2.SINGLE_BANK_DIR, 656), (gen_tier2.HANDOFF_BANK_DIR, 604)):
        manifest = tomllib.loads((bank_dir / "manifest.toml").read_text())
        provenance = manifest["provenance"]
        assert manifest["cost_max"] == gen_tier2.MANIFEST_COST_MAX
        assert provenance["mode"] == "enumerate"
        assert provenance["eval_percent"] == 20
        assert provenance["train_count"] + provenance["eval_count"] == total
        assert provenance["eval_count"] == (total * 20 + 50) // 100
        assert provenance["rejected"] == 0
        assert provenance["size"] == [6, 6]
        for split, count in (("train", provenance["train_count"]), ("eval", provenance["eval_count"])):
            omls = sorted((bank_dir / split).glob("v*.oml"))
            assert len(omls) == count, (bank_dir.name, split)
            assert all(oml.with_suffix(".oms").exists() for oml in omls), (bank_dir.name, split)


def test_repo_class_banks_partition_pair_space():
    """Both banks together cover exactly the 1260 ordered 6x6 pairs; nothing overlaps within or across banks."""
    bank_keys: dict[str, set] = {}
    for bank_dir in (gen_tier2.SINGLE_BANK_DIR, gen_tier2.HANDOFF_BANK_DIR):
        keys = [_oml_key(oml) for split in ("train", "eval") for oml in sorted((bank_dir / split).glob("v*.oml"))]
        assert len(keys) == len(set(keys)), bank_dir.name
        bank_keys[bank_dir.name] = set(keys)
    assert not bank_keys["tier2_single"] & bank_keys["tier2_handoff"]
    cells = [Hex(row, col) for row in range(6) for col in range(6)]
    universe = {(e.r, e.s, x.r, x.s) for e in cells for x in cells if e != x}
    assert bank_keys["tier2_single"] | bank_keys["tier2_handoff"] == universe


def test_repo_single_bank_strata_match_files():
    """Single-bank manifest strata (ring x distance) match the files, and ring totals hit the measured basis."""
    provenance = tomllib.loads((gen_tier2.SINGLE_BANK_DIR / "manifest.toml").read_text())["provenance"]
    actual = _file_stratum_counts(gen_tier2.SINGLE_BANK_DIR)
    assert _flat_class_strata(provenance["strata"], gen_tier2.SINGLE) == actual
    ring_totals: dict[int, int] = {}
    for stratum, counts in actual.items():
        ring = int(stratum.split(".", 1)[0])
        ring_totals[ring] = ring_totals.get(ring, 0) + counts["train"] + counts["eval"]
    assert ring_totals == {1: 428, 2: 120, 3: 108}


def test_repo_handoff_bank_strata_match_files():
    provenance = tomllib.loads((gen_tier2.HANDOFF_BANK_DIR / "manifest.toml").read_text())["provenance"]
    assert _flat_class_strata(provenance["strata"], gen_tier2.HANDOFF) == _file_stratum_counts(
        gen_tier2.HANDOFF_BANK_DIR
    )


def _oml_key(oml: Path) -> tuple[int, int, int, int]:
    coords: dict[str, tuple[int, int]] = {}
    for line in oml.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in ("entrance", "exit"):
            coords[parts[0]] = (int(parts[2]), int(parts[3]))
    return (*coords["entrance"], *coords["exit"])


@pytest.mark.parametrize("bank_dir", [gen_tier2.SINGLE_BANK_DIR, gen_tier2.HANDOFF_BANK_DIR])
@pytest.mark.parametrize("split", ["train", "eval"])
def test_repo_class_bank_endpoints_solve(bank_dir, split):
    count = tomllib.loads((bank_dir / "manifest.toml").read_text())["provenance"][f"{split}_count"]
    for index in (0, count - 1):
        oml = bank_dir / f"{split}/v{index:03d}.oml"
        entry = LevelEntry(
            source=oml,
            cost_max=gen_tier2.MANIFEST_COST_MAX,
            name=oml.stem,
            solutions=(oml.with_suffix(".oms"),),
        )
        assert load_solutions(entry)[0]


@pytest.mark.corpus
@pytest.mark.parametrize("bank_dir", [gen_tier2.SINGLE_BANK_DIR, gen_tier2.HANDOFF_BANK_DIR])
def test_repo_class_bank_full_solves(bank_dir):
    """Corpus sweep: every emitted pair re-loads, decodes, applies and solves from disk."""
    for split in ("train", "eval"):
        for oml in sorted((bank_dir / split).glob("v*.oml")):
            entry = LevelEntry(
                source=oml,
                cost_max=gen_tier2.MANIFEST_COST_MAX,
                name=oml.stem,
                solutions=(oml.with_suffix(".oms"),),
            )
            assert load_solutions(entry)[0], oml.name
