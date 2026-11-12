"""Corpus scanner: recursive discovery, manifests, single/group resolution."""

import tomllib

import pytest

from opus_magnum_blackbox import gen_tier1
from opus_magnum_blackbox.config import RewardConfig
from opus_magnum_blackbox.corpus import LEVELS_DIR, TIER0_LEVELS, resolve_entries, scan_levels


def _bank_provenance() -> dict:
    return tomllib.loads((gen_tier1.BANK_DIR / "manifest.toml").read_text())["provenance"]


@pytest.fixture
def tree(tmp_path):
    """
    A small corpus tree: one family with a manifest, a manifest-less top-level
    level, and (mutable) spots for the error paths.
    """
    root = tmp_path / "levels"
    (root / "family/train").mkdir(parents=True)
    (root / "family/eval").mkdir()
    (root / "family/manifest.toml").write_text("cost_max = 7\nweight = 2.5\n")
    (root / "family/train/v00.oml").write_text("size 6 6\n")
    (root / "family/train/v01.oml").write_text("size 6 6\n")
    (root / "family/train/v00.oms").write_text("place arm 3 1 rotation=0\n")
    (root / "family/eval/v00.oml").write_text("size 6 6\n")
    (root / "toplevel.oml").write_text("size 6 6\n")
    return root


# --- the real repo tree ---


def test_tier0_resolves_unchanged():
    [entry] = resolve_entries(["tier0_typed_delivery"])
    [explicit] = TIER0_LEVELS
    assert entry == explicit
    assert entry.name == "tier0_typed_delivery"
    assert entry.cost_max == 50
    assert entry.source == LEVELS_DIR / "tier0_typed_delivery.oml"
    assert entry.solutions == (LEVELS_DIR / "tier0_typed_delivery.oms",)


def test_scan_registry_contains_tier0_and_bank():
    registry = scan_levels()
    assert registry["tier0_typed_delivery"] == TIER0_LEVELS[0]
    bank_names = {name for name in registry if name.startswith(gen_tier1.FAMILY_NAME)}
    provenance = _bank_provenance()
    assert len(bank_names) == provenance["train_count"] + provenance["eval_count"]
    assert "tier1_simple_arms/train/v000" in registry


def test_bank_levels_use_family_manifest():
    [train, eval_] = resolve_entries(["tier1_simple_arms/train/v000", "tier1_simple_arms/eval/v085"])
    for entry in (train, eval_):
        assert entry.cost_max == gen_tier1.MANIFEST_COST_MAX
        assert entry.weight == gen_tier1.MANIFEST_WEIGHT
    assert train.solutions == (LEVELS_DIR / "tier1_simple_arms/train/v000.oms",)
    assert eval_.solutions == (LEVELS_DIR / "tier1_simple_arms/eval/v085.oms",)


def test_group_resolves_sorted_and_unique():
    train = resolve_entries(["tier1_simple_arms/train"])
    names = [entry.name for entry in train]
    train_count = _bank_provenance()["train_count"]
    assert len(train) == train_count
    assert names == sorted(set(names))
    assert names[0] == "tier1_simple_arms/train/v000"
    assert names[-1] == f"tier1_simple_arms/train/v{train_count - 1:03d}"


def test_family_group_spans_subdirs():
    names = [entry.name for entry in resolve_entries(["tier1_simple_arms"])]
    provenance = _bank_provenance()
    assert len(names) == provenance["train_count"] + provenance["eval_count"]
    assert names == sorted(set(names))
    assert sum(name.startswith("tier1_simple_arms/train/") for name in names) == provenance["train_count"]
    assert sum(name.startswith("tier1_simple_arms/eval/") for name in names) == provenance["eval_count"]


def test_mixed_names_keep_order():
    entries = resolve_entries(["tier1_simple_arms/eval/v000", "tier0_typed_delivery"])
    assert [entry.name for entry in entries] == ["tier1_simple_arms/eval/v000", "tier0_typed_delivery"]


# --- tmp tree: manifests, fallback, error paths ---


def test_manifest_applies_to_family_subdirs(tree):
    [train, eval_] = resolve_entries(["family/train/v00", "family/eval/v00"], levels_dir=tree)
    assert train.cost_max == 7
    assert train.weight == 2.5
    assert train.source == tree / "family/train/v00.oml"
    assert train.solutions == (tree / "family/train/v00.oms",)
    assert eval_.cost_max == 7
    assert eval_.weight == 2.5
    assert eval_.solutions == ()


def test_top_level_without_manifest_uses_default(tree):
    [entry] = resolve_entries(["toplevel"], levels_dir=tree)
    assert entry.cost_max == RewardConfig().cost_max
    assert entry.weight == 1.0


def test_missing_manifest_in_subdir_raises(tree):
    (tree / "family/manifest.toml").unlink()
    with pytest.raises(ValueError, match="manifest"):
        resolve_entries(["family/train/v00"], levels_dir=tree)


def test_manifest_without_cost_max_raises(tree):
    (tree / "family/manifest.toml").write_text("weight = 2.0\n")
    with pytest.raises(TypeError, match="cost_max"):
        resolve_entries(["family/train/v00"], levels_dir=tree)


def test_unknown_name_raises_with_listing(tree):
    with pytest.raises(ValueError, match=r"unknown levels: \['nope'\]; corpus has:"):
        resolve_entries(["nope"], levels_dir=tree)


def test_empty_group_raises(tree):
    (tree / "empty").mkdir()
    with pytest.raises(ValueError, match=r"no level \.oml files"):
        resolve_entries(["empty"], levels_dir=tree)
