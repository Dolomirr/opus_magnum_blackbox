"""Tier-2 anywhere family: piston-length singles and two-arm handoffs, simulator-verified."""

import random

import pytest
from opus_magnum_sym import AtomType, Hex, Level, StepActionType

from opus_magnum_blackbox.codec import OpusTokenizer, apply_solution, decode_solution, solution_to_oms
from opus_magnum_blackbox.families import (
    FamilyError,
    anywhere_plan,
    anywhere_reference_oms,
    anywhere_reference_solution,
    anywhere_reference_tokens,
    anywhere_spec,
    hex_distance,
    verify,
)
from opus_magnum_blackbox.families.tier2_anywhere import _anywhere_level_spec, _handoff_candidates, _plan_solves
from opus_magnum_blackbox.grammar import GrammarError
from opus_magnum_blackbox.masking import ActionMasker

# One case per class: single rings 1/2/3 (including the k_clw > 3 geometries whose
# short path is CCLW) and handoffs at d5, d7, d10. Distance >= 5 guarantees no
# single-arm plan exists (one ring reaches at most d6 only at ring 3's 120-degree
# geometry), so those winners must be two-arm handoffs.
CASES = [
    ("single_ring1", Hex(0, 0), Hex(0, 1), 1),
    ("single_ring1_cclw", Hex(0, 0), Hex(1, 0), 1),
    ("single_ring2_d4", Hex(2, 1), Hex(0, 5), 1),
    ("single_ring3_d3", Hex(0, 0), Hex(0, 3), 1),
    ("single_ring3_straight", Hex(0, 3), Hex(0, 0), 1),
    ("single_ring3_d6", Hex(0, 0), Hex(3, 3), 1),
    ("handoff_d5", Hex(1, 1), Hex(2, 5), 2),
    ("handoff_d7", Hex(0, 0), Hex(4, 3), 2),
    ("handoff_d10", Hex(0, 0), Hex(5, 5), 2),
]


@pytest.mark.parametrize(("name", "entrance", "exit_", "arms"), CASES)
def test_reference_solution_solves(name, entrance, exit_, arms):
    spec = anywhere_spec(entrance, exit_)
    assert verify(spec, anywhere_reference_solution(spec)) > 0


@pytest.mark.parametrize(("name", "entrance", "exit_", "arms"), CASES)
def test_class_rule_targets(name, entrance, exit_, arms):
    spec = anywhere_spec(entrance, exit_)
    assert spec.target == (2 if arms == 1 else 1)


@pytest.mark.parametrize(("name", "entrance", "exit_", "arms"), CASES)
def test_plan_shape(name, entrance, exit_, arms):
    spec = anywhere_spec(entrance, exit_)
    plan = anywhere_plan(spec)
    assert plan is not None
    assert plan.target == spec.target
    assert len(plan.arms) == arms
    blueprints = {arm.blueprint for arm in plan.arms}
    if name.startswith("single_ring1"):
        assert blueprints == {"arm"}
    else:
        assert "arm_piston" in blueprints


@pytest.mark.parametrize(("name", "entrance", "exit_", "arms"), CASES)
def test_tokens_round_trip(name, entrance, exit_, arms):
    spec = anywhere_spec(entrance, exit_)
    ir = anywhere_reference_solution(spec)
    decoded = decode_solution(anywhere_reference_tokens(spec), spec, OpusTokenizer().vocab)
    assert not isinstance(decoded, GrammarError)
    assert decoded.submitted
    assert [(p.builtin.name, p.rotation, p.q, p.r) for p in decoded.placements] == [
        (p.builtin.name, p.rotation, p.q, p.r) for p in ir.placements
    ]
    assert [(a.handle, a.tick, a.action) for a in decoded.actions] == [(a.handle, a.tick, a.action) for a in ir.actions]


@pytest.mark.parametrize(("name", "entrance", "exit_", "arms"), CASES)
def test_tokens_are_mask_valid_at_every_prefix(name, entrance, exit_, arms):
    spec = anywhere_spec(entrance, exit_)
    tokens = anywhere_reference_tokens(spec)
    masker = ActionMasker(OpusTokenizer().vocab)
    for position, token in enumerate(tokens):
        assert masker.mask(tokens[:position], spec)[token], f"token {position} invalid at its prefix"


def test_oms_renders_both_arms():
    spec = anywhere_spec(Hex(0, 0), Hex(4, 3))
    ir = anywhere_reference_solution(spec)
    level = Level(spec)
    report = apply_solution(ir, spec, level)
    assert report.ok
    text = solution_to_oms(ir, report.handles, spec)
    assert text.count("place ") == 2
    assert "place arm_piston" in text
    assert text.count("grab") == 2
    assert text.count("release") == 2
    assert anywhere_reference_oms(spec) == text


def test_ring1_cycle_matches_tier1_shape():
    """The ring-1 anywhere cycle is the tier-1 cycle: GRAB, k turns, RELEASE, (6-k) turns."""
    spec = anywhere_spec(Hex(0, 0), Hex(0, 1))
    actions = [action for _, action in anywhere_plan(spec).arms[0].ticks]
    clw = StepActionType.ROTATE_CLW
    assert actions == [
        StepActionType.GRAB,
        clw,
        StepActionType.RELEASE,
        clw,
        clw,
        clw,
        clw,
        clw,
    ] * 2


def test_long_carry_geometry_picks_short_cclw_swing():
    """Regression (stage 1.5): (0,0)->(1,0) has k_clw=5; the winner must be tier-1's CCLW program."""
    spec = anywhere_spec(Hex(0, 0), Hex(1, 0))
    plan = anywhere_plan(spec)
    assert plan is not None
    assert len(plan.arms) == 1
    assert spec.target == 2
    cclw = StepActionType.ROTATE_CCLW
    assert [action for _, action in plan.arms[0].ticks] == [
        StepActionType.GRAB,
        cclw,
        StepActionType.RELEASE,
        cclw,
        cclw,
        cclw,
        cclw,
        cclw,
    ] * 2
    assert verify(spec, anywhere_reference_solution(spec)) > 0


def test_straight_line_d3_stays_single_arm():
    """Regression (stage 1.5): straight-line d3 (k_clw=5 at ring 3) must not fall back to a handoff."""
    spec = anywhere_spec(Hex(0, 3), Hex(0, 0))
    plan = anywhere_plan(spec)
    assert plan is not None
    assert len(plan.arms) == 1
    assert spec.target == 2
    actions = [action for _, action in plan.arms[0].ticks]
    assert len(actions) == 24  # 12-action ring-3 cycle, target 2
    assert verify(spec, anywhere_reference_solution(spec)) > 0


def test_single_piston_cycle_books_extends_at_cycle_ends():
    """Ring 2 single: each cycle opens with EXTEND and closes with RETRACT around the turn core."""
    spec = anywhere_spec(Hex(2, 1), Hex(0, 5))
    actions = [action for _, action in anywhere_plan(spec).arms[0].ticks]
    assert len(actions) == 20  # 10-action cycle, target 2
    for start in (0, 10):
        cycle = actions[start : start + 10]
        assert cycle[0] is StepActionType.EXTEND
        assert cycle[1] is StepActionType.GRAB
        assert cycle[4] is StepActionType.RELEASE
        assert cycle[-1] is StepActionType.RETRACT


def test_handoff_grab_follows_release():
    """The handover atom persists on the board, so B grabs at least one tick after A releases."""
    spec = anywhere_spec(Hex(0, 0), Hex(4, 3))
    ir = anywhere_reference_solution(spec)
    release_a = min(action.tick for action in ir.actions if action.handle == 0 and action.action is StepActionType.RELEASE)
    grab_b = min(action.tick for action in ir.actions if action.handle == 1 and action.action is StepActionType.GRAB)
    assert grab_b >= release_a + 1
    assert ir.actions == sorted(ir.actions, key=lambda action: (action.handle, action.tick))


def test_plan_is_deterministic():
    spec = anywhere_spec(Hex(1, 1), Hex(2, 5))
    assert anywhere_plan(spec) == anywhere_plan(spec)
    assert anywhere_reference_tokens(spec) == anywhere_reference_tokens(spec)


def test_d3_pair_solves():
    """A d3 pair has both single (ring 3) and handoff plans; the ranked winner must still solve."""
    spec = anywhere_spec(Hex(2, 1), Hex(2, 4))
    assert hex_distance(Hex(2, 1), Hex(2, 4)) == 3
    assert verify(spec, anywhere_reference_solution(spec)) > 0


def test_length1_handoff_solves_d3():
    """White-box pin of the spike: two plain arms handing over at a shared ring cell solve d3."""
    entrance, exit_ = Hex(2, 1), Hex(2, 4)
    spec = _anywhere_level_spec(entrance, exit_, 1, 30, (6, 6), AtomType.WATER)
    candidates = _handoff_candidates(entrance, exit_, 30, (6, 6))
    assert candidates
    plains = [plan for _, plan in candidates if all(arm.blueprint == "arm" for arm in plan.arms)]
    assert plains
    assert any(_plan_solves(spec, plan) for plan in plains)


@pytest.mark.parametrize(("name", "entrance", "exit_", "arms"), CASES)
def test_wire_caps(name, entrance, exit_, arms):
    spec = anywhere_spec(entrance, exit_)
    ir = anywhere_reference_solution(spec)
    assert len(ir.placements) <= 16
    assert len(ir.actions) <= 32
    assert max(action.tick for action in ir.actions) < spec.steps
    assert len(anywhere_reference_tokens(spec)) <= OpusTokenizer().config.max_decoder_len


def test_explicit_target_is_honoured():
    spec = anywhere_spec(Hex(0, 0), Hex(0, 1), target=3)
    assert spec.target == 3
    assert verify(spec, anywhere_reference_solution(spec)) > 0


def test_spec_shape_contract():
    spec = anywhere_spec(Hex(0, 0), Hex(0, 3))
    assert spec.size == (6, 6)
    assert spec.entrypoints_fixed is True
    assert spec.cycling is False
    assert spec.allowed_blueprints == frozenset({"arm", "arm_piston"})
    assert [entrypoint.pos for entrypoint in spec.entrypoints] == [Hex(0, 0), Hex(0, 3)]
    assert all(len(shape.cells) == 1 and not shape.links for shape in spec.shapes)


@pytest.mark.parametrize(
    ("entrance", "exit_", "target"),
    [
        (Hex(0, 0), Hex(0, 0), None),  # same cell
        (Hex(0, 0), Hex(0, 6), None),  # out of board
        (Hex(-1, 2), Hex(0, 3), None),  # out of board
        (Hex(0, 0), Hex(4, 3), 2),  # d7: only handoffs reach it, and they deliver exactly 1
        (Hex(0, 0), Hex(5, 5), 3),  # d10: no plan fits the 32-triplet cap at target 3
    ],
)
def test_invalid_requests_raise(entrance, exit_, target):
    with pytest.raises(FamilyError):
        anywhere_spec(entrance, exit_, target=target)


# --- stage-1.5 acceptance sweep ---------------------------------------------------

ALL_CELLS = [Hex(row, col) for row in range(6) for col in range(6)]
ALL_PAIRS = [(entrance, exit_) for entrance in ALL_CELLS for exit_ in ALL_CELLS if entrance != exit_]


def _classify_pair(entrance: Hex, exit_: Hex) -> tuple[int, int]:
    spec = anywhere_spec(entrance, exit_)
    plan = anywhere_plan(spec)
    assert plan is not None  # a verified build must always yield its plan back
    return len(plan.arms), spec.target


def test_sweep_sampled_pair_classes():
    """Default-run fallback for the full sweep: a seeded sample obeys the class/target rule."""
    for entrance, exit_ in random.Random(0).sample(ALL_PAIRS, 60):
        arms, target = _classify_pair(entrance, exit_)
        assert (arms, target) in {(1, 2), (2, 1)}


@pytest.mark.corpus
def test_sweep_full_pair_classes():
    """
    All 1260 ordered pairs split exactly into 656 single-arm (target 2) + 604 handoff (target 1).

    Any FamilyError, plan-less spec, or single-arm-at-target-1 build shows up as a
    count off this table (or a raised error), pinning the stage-1.5 CCLW fix.
    """
    counts: dict[tuple[int, int], int] = {}
    for entrance, exit_ in ALL_PAIRS:
        arms, target = _classify_pair(entrance, exit_)
        counts[(arms, target)] = counts.get((arms, target), 0) + 1
    assert counts == {(1, 2): 656, (2, 1): 604}
