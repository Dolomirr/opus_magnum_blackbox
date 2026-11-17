"""Tier-3 placed family: solution-placed entrypoints over the 16-config space, simulator-verified."""

from dataclasses import replace

import pytest
from opus_magnum_sym import AtomType, ComponentType, Hex, Level, StepActionType
from opus_magnum_sym.levels.spec import EntrypointSpec

from opus_magnum_blackbox.codec import OpusTokenizer, apply_solution, decode_solution, solution_to_oms
from opus_magnum_blackbox.families import (
    ANCHOR_CELLS,
    LONG_BLUEPRINTS,
    RING_OFFSETS,
    SHORT_BLUEPRINTS,
    FamilyError,
    PlacedConfig,
    placed_configs,
    placed_plan,
    placed_reference_oms,
    placed_reference_solution,
    placed_reference_tokens,
    placed_spec,
    placed_spec_plan,
    verify,
)
from opus_magnum_blackbox.families._common import ENTRANCE_SHAPE, EXIT_SHAPE, _cell_shape
from opus_magnum_blackbox.grammar import GrammarError
from opus_magnum_blackbox.masking import ActionMasker

CONFIGS = placed_configs()
ATOMS = (AtomType.WATER, AtomType.FIRE, AtomType.EARTH, AtomType.AIR)
RADIAL_ACTIONS = (StepActionType.EXTEND, StepActionType.RETRACT)


def _fingerprint(sol):
    placements = [(p.slot, p.builtin.name if p.builtin else None, p.rotation, p.q, p.r) for p in sol.placements]
    actions = sorted(((a.handle, a.tick, a.action) for a in sol.actions), key=lambda item: item[:2])
    return placements, actions


def test_config_space_is_sixteen():
    assert len(CONFIGS) == 16
    assert {(config.atom, config.target, config.long_arms) for config in CONFIGS} == {
        (atom, target, long_arms) for atom in ATOMS for target in (1, 2) for long_arms in (False, True)
    }


@pytest.mark.parametrize("config", CONFIGS)
def test_reference_solution_solves(config):
    spec = placed_spec(config)
    assert verify(spec, placed_reference_solution(spec)) > 0


@pytest.mark.parametrize("config", CONFIGS)
def test_spec_shape_contract(config):
    spec = placed_spec(config)
    assert spec.size == (6, 6)
    assert spec.entrypoints_fixed is False
    assert spec.cycling is False
    assert spec.steps == 30
    assert spec.target == config.target
    assert spec.allowed_blueprints == (LONG_BLUEPRINTS if config.long_arms else SHORT_BLUEPRINTS)
    assert [entrypoint.pos for entrypoint in spec.entrypoints] == [None, None]
    assert [entrypoint.kind for entrypoint in spec.entrypoints] == [ComponentType.ENTRANCE, ComponentType.EXIT]
    for shape in spec.shapes:
        assert len(shape.cells) == 1
        assert not shape.links
        assert shape.cells[0].atom_type is config.atom


@pytest.mark.parametrize("config", CONFIGS)
def test_tokens_round_trip(config):
    spec = placed_spec(config)
    ir = placed_reference_solution(spec)
    decoded = decode_solution(placed_reference_tokens(spec), spec, OpusTokenizer().vocab)
    assert not isinstance(decoded, GrammarError)
    assert decoded.submitted
    assert _fingerprint(decoded) == _fingerprint(ir)


@pytest.mark.parametrize("config", CONFIGS)
def test_tokens_are_mask_valid_at_every_prefix(config):
    spec = placed_spec(config)
    tokens = placed_reference_tokens(spec)
    masker = ActionMasker(OpusTokenizer().vocab)
    for position, token in enumerate(tokens):
        assert masker.mask(tokens[:position], spec)[token], f"token {position} invalid at its prefix"


@pytest.mark.parametrize("config", CONFIGS)
def test_reference_tokens_solve_through_decode(config):
    """End-to-end on the unfixed spec: tokens -> decode -> apply (entry slots route to place_entrypoint) -> run."""
    spec = placed_spec(config)
    level = Level(spec)
    report = OpusTokenizer().decode_solution_to_level(placed_reference_tokens(spec), spec, level)
    assert report.ok, report.error
    result = level.run()
    assert result.solved
    level.reset()


@pytest.mark.parametrize("config", CONFIGS)
def test_single_static_arm_pure_rotation(config):
    """The tier-2 single-arm leg exactly: three placements, no radial actions."""
    sol = placed_reference_solution(placed_spec(config))
    assert [placement.slot for placement in sol.placements[:2]] == [0, 1]
    assert sol.placements[2].builtin.name in ("arm", "arm_l2", "arm_l3")
    actions = [action.action for action in sol.actions]
    assert not any(action in RADIAL_ACTIONS for action in actions)
    assert len(actions) == 8 * config.target


@pytest.mark.parametrize("config", CONFIGS)
def test_oms_places_entrypoints(config):
    spec = placed_spec(config)
    ir = placed_reference_solution(spec)
    level = Level(spec)
    report = apply_solution(ir, spec, level)
    assert report.ok
    text = solution_to_oms(ir, report.handles, spec)
    assert text.count("place ") == 3
    assert "place entrance_1" in text
    assert "place exit_1" in text
    assert placed_reference_oms(spec) == text


def test_plan_is_deterministic():
    spec = placed_spec(CONFIGS[0])
    assert placed_plan(spec) == placed_plan(spec)
    assert placed_reference_tokens(spec) == placed_reference_tokens(spec)


def test_allow_list_axis_changes_layout():
    """Go condition: the arm-only winner must differ from the long-allow-list winner on every (atom, target)."""
    for atom in ATOMS:
        for target in (1, 2):
            short = placed_spec_plan(PlacedConfig(atom=atom, target=target, long_arms=False))[1]
            long_ = placed_spec_plan(PlacedConfig(atom=atom, target=target, long_arms=True))[1]
            assert (short.entrance, short.exit, short.arm.base) != (long_.entrance, long_.exit, long_.arm.base)
            assert short.arm.blueprint == "arm"
            assert long_.arm.blueprint != "arm"


def test_target_axis_changes_program():
    """Go condition: target 1 vs 2 differ in program length (8 vs 16 actions) at equal layout."""
    for atom in ATOMS:
        for long_arms in (False, True):
            one = placed_spec_plan(PlacedConfig(atom=atom, target=1, long_arms=long_arms))[1]
            two = placed_spec_plan(PlacedConfig(atom=atom, target=2, long_arms=long_arms))[1]
            assert (one.entrance, one.exit, one.arm.base) == (two.entrance, two.exit, two.arm.base)
            assert len(one.arm.ticks) == 8
            assert len(two.arm.ticks) == 16


def test_atom_axis_changes_layout():
    """The per-atom anchors keep the four canonical layouts apart."""
    winners = [placed_spec_plan(PlacedConfig(atom=atom, target=2, long_arms=True))[1] for atom in ATOMS]
    assert len({(plan.entrance.r, plan.entrance.s) for plan in winners}) == 4


def test_anchor_reach_supports_the_fall_through():
    """Every anchor keeps a separation-3 candidate on board along direction 0 — the allow-list axis's variety floor."""
    for anchor in ANCHOR_CELLS.values():
        far = Hex(anchor.r + RING_OFFSETS[0][0] * 3, anchor.s + RING_OFFSETS[0][1] * 3)
        assert 0 <= far.r < 6
        assert 0 <= far.s < 6


def test_configs_outside_the_family_raise():
    with pytest.raises(FamilyError):
        placed_spec(PlacedConfig(atom=AtomType.SALT, target=1, long_arms=False))
    with pytest.raises(FamilyError):
        placed_spec(PlacedConfig(atom=AtomType.WATER, target=3, long_arms=False))
    with pytest.raises(FamilyError):
        placed_spec(PlacedConfig(atom=AtomType.WATER, target=1, long_arms=False), steps=65)
    with pytest.raises(FamilyError):
        placed_spec(PlacedConfig(atom=AtomType.WATER, target=1, long_arms=False), size=(8, 8))


def test_specs_outside_the_family_raise():
    spec = placed_spec(CONFIGS[0])
    fixed = replace(
        spec,
        entrypoints=tuple(
            EntrypointSpec(kind=entrypoint.kind, shape=entrypoint.shape, pos=Hex(0, index))
            for index, entrypoint in enumerate(spec.entrypoints)
        ),
        entrypoints_fixed=True,
    )
    alien = replace(spec, allowed_blueprints=frozenset({"arm", "arm_piston"}))
    mixed = replace(
        spec,
        shapes=(
            _cell_shape(ENTRANCE_SHAPE, ComponentType.ENTRANCE, AtomType.WATER),
            _cell_shape(EXIT_SHAPE, ComponentType.EXIT, AtomType.FIRE),
        ),
    )
    for outsider in (fixed, alien, mixed):
        with pytest.raises(FamilyError):
            placed_plan(outsider)
        with pytest.raises(FamilyError):
            placed_reference_solution(outsider)
