"""Tier-3 bank generator: stratified config split, deterministic emission, on-disk validation, provenance."""

import tomllib
from pathlib import Path

import pytest
from opus_magnum_sym import Level

from opus_magnum_blackbox import gen_tier3
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.families import (
    LONG_BLUEPRINTS,
    SHORT_BLUEPRINTS,
    placed_config,
    placed_configs,
)
from opus_magnum_blackbox.solutions import load_solutions


def _keys(configs):
    return {(config.atom.name, config.target, config.long_arms) for config in configs}


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    return gen_tier3.generate_bank(tmp_path_factory.mktemp("tier3_bank"))


# --- config split ---


def test_split_sizes_disjoint_axis_coverage():
    train, eval_ = gen_tier3.split_configs()
    assert len(train) == gen_tier3.TRAIN_COUNT
    assert len(eval_) == gen_tier3.EVAL_COUNT
    assert not _keys(train) & _keys(eval_)
    assert _keys(train) | _keys(eval_) == _keys(placed_configs())
    # the stratification constraint: every axis value keeps train coverage
    assert {config.atom for config in train} == {config.atom for config in placed_configs()}
    assert {config.target for config in train} == {1, 2}
    assert {config.long_arms for config in train} == {False, True}
    # one eval pick per atom stratum
    assert sorted(config.atom.name for config in eval_) == ["AIR", "EARTH", "FIRE", "WATER"]


def test_split_configs_deterministic():
    assert gen_tier3.split_configs(seed=5) == gen_tier3.split_configs(seed=5)


# --- bank emission ---


def test_generation_is_deterministic(tmp_path):
    params = gen_tier3.BankParams(seed=7)
    first = gen_tier3.generate_bank(tmp_path / "one", params=params)
    second = gen_tier3.generate_bank(tmp_path / "two", params=params)
    names_one = sorted(p.relative_to(first.root).as_posix() for p in first.root.rglob("*") if p.is_file())
    names_two = sorted(p.relative_to(second.root).as_posix() for p in second.root.rglob("*") if p.is_file())
    assert names_one == names_two
    for name in names_one:
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes(), name


def test_regeneration_removes_stale_files(tmp_path):
    root = tmp_path / "bank"
    gen_tier3.generate_bank(root)
    (root / "train" / "v099.oml").write_text("stale")
    bank = gen_tier3.generate_bank(root)
    assert not (root / "train" / "v099.oml").exists()
    assert sorted(variant.oml.name for variant in bank.train) == [f"v{index:03d}.oml" for index in range(gen_tier3.TRAIN_COUNT)]
    assert sorted(variant.oml.name for variant in bank.eval) == [f"v{index:03d}.oml" for index in range(gen_tier3.EVAL_COUNT)]


def test_bank_splits_are_config_disjoint(bank):
    train_keys = {variant.key for variant in bank.train}
    eval_keys = {variant.key for variant in bank.eval}
    assert len(bank.train) == gen_tier3.TRAIN_COUNT
    assert len(bank.eval) == gen_tier3.EVAL_COUNT
    assert not train_keys & eval_keys


@pytest.mark.parametrize("index", range(gen_tier3.TRAIN_COUNT + gen_tier3.EVAL_COUNT))
def test_generated_variant_loads_and_solves(bank, index):
    split, ordinal = divmod(index, gen_tier3.TRAIN_COUNT)
    variants = bank.train if split == 0 else bank.eval
    variant = variants[ordinal]
    entry = LevelEntry(
        source=variant.oml,
        cost_max=gen_tier3.MANIFEST_COST_MAX,
        name=variant.oml.stem,
        solutions=(variant.oms,),
    )
    tokens = load_solutions(entry)
    assert len(tokens) == 1
    assert tokens[0]


def test_generated_specs_match_family_configs(bank):
    for variant in bank.variants:
        spec = Level.from_file(variant.oml).spec
        assert spec == variant.spec
        config = placed_config(spec)
        assert (config.atom.name, config.target, config.long_arms) == variant.key


# --- manifest consistency with files ---


def test_manifest_header_matches_bank(bank):
    manifest = tomllib.loads(bank.manifest.read_text())
    assert manifest["cost_max"] == gen_tier3.MANIFEST_COST_MAX
    assert manifest["weight"] == gen_tier3.MANIFEST_WEIGHT
    provenance = manifest["provenance"]
    assert provenance["mode"] == "enumerate"
    assert provenance["seed"] == 0
    assert provenance["train_count"] == len(bank.train)
    assert provenance["eval_count"] == len(bank.eval)
    assert provenance["steps"] == 30
    assert provenance["size"] == [6, 6]
    assert len(provenance["configs"]) == len(bank.variants)


def _oml_fields(oml: Path):
    """(atom, target, allow-list, entrypoints_fixed) of one variant, read back from disk."""
    spec = Level.from_file(oml).spec
    atom = next(cell.atom_type for shape in spec.shapes for cell in shape.cells if cell.atom_type)
    return atom, spec.target, spec.allowed_blueprints, spec.entrypoints_fixed


def _oms_placements(oms: Path) -> list[tuple[str, int, int, int]]:
    """(name, r, s, rotation) of every `place` line of one emitted solution."""
    placements = []
    for line in oms.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] != "place":
            continue
        rotation = 0
        for arg in parts[4:]:
            key, _, value = arg.partition("=")
            assert key == "rotation", arg
            rotation = int(value)
        placements.append((parts[1], int(parts[2]), int(parts[3]), rotation))
    return placements


def test_manifest_per_config_provenance_matches_files(bank):
    """Every `[[provenance.configs]]` record must be re-derivable from the emitted files themselves."""
    records = {record["variant"]: record for record in tomllib.loads(bank.manifest.read_text())["provenance"]["configs"]}
    assert len(records) == len(bank.variants)
    for variant in bank.variants:
        record = records[f"{variant.oml.parent.name}/{variant.oml.stem}"]
        atom, target, allow, fixed = _oml_fields(variant.oml)
        assert fixed is False  # the whole family point: entrypoints are solution-placed
        assert atom.name.lower() == record["atom"]
        assert target == record["target"]
        assert allow == (LONG_BLUEPRINTS if record["long_arms"] else SHORT_BLUEPRINTS)
        entrance, exit_, arm = _oms_placements(variant.oms)
        assert (entrance[1], entrance[2]) == tuple(record["entrance"])
        assert (exit_[1], exit_[2]) == tuple(record["exit"])
        assert arm[0] == record["arm"]
        assert (arm[1], arm[2]) == tuple(record["base"])
        assert arm[3] == record["rotation"]
        assert (entrance[1], entrance[2]) != (exit_[1], exit_[2])


# --- the checked-in repo bank ---


def test_repo_bank_matches_manifest():
    manifest = tomllib.loads((gen_tier3.BANK_DIR / "manifest.toml").read_text())
    provenance = manifest["provenance"]
    assert provenance["train_count"] == gen_tier3.TRAIN_COUNT
    assert provenance["eval_count"] == gen_tier3.EVAL_COUNT
    for split, count in (("train", provenance["train_count"]), ("eval", provenance["eval_count"])):
        omls = sorted((gen_tier3.BANK_DIR / split).glob("v*.oml"))
        assert len(omls) == count, split
        assert all(oml.with_suffix(".oms").exists() for oml in omls), split


def test_repo_bank_full_solves():
    """Corpus sweep at bank scale (16 variants): every emitted pair re-loads, decodes, applies and solves from disk."""
    for split in ("train", "eval"):
        for oml in sorted((gen_tier3.BANK_DIR / split).glob("v*.oml")):
            entry = LevelEntry(
                source=oml,
                cost_max=gen_tier3.MANIFEST_COST_MAX,
                name=oml.stem,
                solutions=(oml.with_suffix(".oms"),),
            )
            assert load_solutions(entry)[0], oml.name
