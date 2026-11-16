"""Shared fixtures: an in-Python tier-0 spec and its known-good solution tokens."""

import pytest
from opus_magnum_sym import AtomType, ComponentType, Hex, LevelSpec, ShapeCell, ShapeTemplate, mkentr, mkexit

from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.masking import ActionMasker
from opus_magnum_blackbox.vocab import Vocab


def _straight(name: str, kind: ComponentType, origin_type: AtomType, beam_type: AtomType) -> ShapeTemplate:
    return ShapeTemplate(
        name=name,
        comp_type=kind,
        cells=(
            ShapeCell(offset=Hex(0, 0), atom_type=origin_type),
            ShapeCell(offset=Hex(-1, 1), atom_type=beam_type),
        ),
        links=((0, 1),),
    )


def make_spec(**overrides) -> LevelSpec:
    kwargs = {
        "size": (6, 6),
        "shapes": (
            _straight("entrance_typed", ComponentType.ENTRANCE, AtomType.WATER, AtomType.SALT),
            _straight("exit_typed", ComponentType.EXIT, AtomType.SALT, AtomType.WATER),
        ),
        "entrypoints": (mkentr("entrance_typed"), mkexit("exit_typed")),
        "allowed_blueprints": frozenset({"arm"}),
        "steps": 30,
        "target": 1,
        "cycling": True,
    }
    kwargs.update(overrides)
    return LevelSpec(**kwargs)


@pytest.fixture
def spec() -> LevelSpec:
    return make_spec()


@pytest.fixture
def vocab() -> Vocab:
    return Vocab()


@pytest.fixture
def tokenizer() -> OpusTokenizer:
    return OpusTokenizer()


@pytest.fixture
def masker(vocab) -> ActionMasker:
    return ActionMasker(vocab)


def known_solution_tokens(vocab: Vocab) -> list[int]:
    """Hand-translation of the tier-0 solution (mirrors `27_level_typed.oms`)."""
    tokens = [
        vocab.special_id("[START_LAYOUT]"),
        # entrance_typed at Hex(r=2, s=3), rotation 1
        vocab.value_id("entry_slot", 0),
        vocab.value_id("rotation", 1),
        vocab.value_id("q", 3),
        vocab.value_id("r", 2),
        # exit_typed at Hex(r=3, s=2), rotation 2
        vocab.value_id("entry_slot", 1),
        vocab.value_id("rotation", 2),
        vocab.value_id("q", 2),
        vocab.value_id("r", 3),
        # arm at Hex(r=3, s=3), rotation 0
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
        vocab.value_id("q", 3),
        vocab.value_id("r", 3),
        vocab.special_id("[END_LAYOUT]"),
        vocab.special_id("[START_PROGRAM]"),
        vocab.value_id("handle", 0),
        vocab.value_id("tick", 1),
        vocab.id("ACT_GRAB"),
    ]
    for tick in (2, 3, 4, 5):
        tokens += [vocab.value_id("handle", 0), vocab.value_id("tick", tick), vocab.id("ACT_ROTATE_CLW")]
    tokens += [
        vocab.value_id("handle", 0),
        vocab.value_id("tick", 6),
        vocab.id("ACT_RELEASE"),
        vocab.special_id("[SUBMIT]"),
    ]
    return tokens


def pytest_addoption(parser):
    parser.addoption("--corpus", action="store_true", default=False, help="include corpus-scale sweep tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "corpus: full-corpus sweeps; skipped unless --corpus is passed")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--corpus"):
        return
    skip = pytest.mark.skip(reason="corpus sweep; pass --corpus to include")
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)
