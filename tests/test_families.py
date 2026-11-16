"""Tier-1 family module: spec builder boundary and simulator-verified reference solutions."""

import pytest
from opus_magnum_sym import Hex, Level, LevelSpec, mkentr, mkexit
from opus_magnum_sym.levels.spec import EntrypointSpec

from opus_magnum_blackbox.codec import OpusTokenizer, apply_solution, decode_solution, solution_to_oms
from opus_magnum_blackbox.families import (
    RING_OFFSETS,
    FamilyError,
    arm_plan,
    hex_distance,
    reference_oms,
    reference_solution,
    reference_tokens,
    single_atom_spec,
    verify,
)
from opus_magnum_blackbox.grammar import GrammarError
from opus_magnum_blackbox.masking import ActionMasker

# entrance/exit spread over distance 1 and 2, corners, edges and the middle
PAIRS = [
    (Hex(0, 0), Hex(0, 1)),
    (Hex(0, 0), Hex(0, 2)),  # base sits between the two cells
    (Hex(0, 2), Hex(2, 0)),  # opposite sides of one base
    (Hex(1, 3), Hex(2, 4)),
    (Hex(2, 2), Hex(3, 3)),
    (Hex(3, 3), Hex(3, 4)),
    (Hex(5, 5), Hex(5, 4)),
    (Hex(5, 5), Hex(4, 4)),
    (Hex(4, 0), Hex(5, 1)),
]


@pytest.mark.parametrize(("entrance", "exit_"), PAIRS)
def test_reference_solution_solves(entrance, exit_):
    spec = single_atom_spec(entrance, exit_)
    assert verify(spec) > 0


@pytest.mark.parametrize(("entrance", "exit_"), PAIRS)
def test_tokens_are_mask_valid_at_every_prefix(entrance, exit_):
    spec = single_atom_spec(entrance, exit_)
    tokens = reference_tokens(spec)
    masker = ActionMasker(OpusTokenizer().vocab)
    for position, token in enumerate(tokens):
        assert masker.mask(tokens[:position], spec)[token], f"token {position} invalid at its prefix"


@pytest.mark.parametrize(("entrance", "exit_"), PAIRS)
def test_tokens_decode_back_to_reference_ir(entrance, exit_):
    spec = single_atom_spec(entrance, exit_)
    ir = reference_solution(spec)
    decoded = decode_solution(reference_tokens(spec), spec, OpusTokenizer().vocab)
    assert not isinstance(decoded, GrammarError)
    assert decoded.submitted
    assert [(p.builtin.name, p.rotation, p.q, p.r) for p in decoded.placements] == [
        (p.builtin.name, p.rotation, p.q, p.r) for p in ir.placements
    ]
    assert [(a.handle, a.tick, a.action) for a in decoded.actions] == [(a.handle, a.tick, a.action) for a in ir.actions]


@pytest.mark.parametrize(("entrance", "exit_"), PAIRS)
def test_oms_round_trips_through_placement(entrance, exit_):
    spec = single_atom_spec(entrance, exit_)
    ir = reference_solution(spec)
    level = Level(spec)
    report = apply_solution(ir, spec, level)
    assert report.ok
    text = solution_to_oms(ir, report.handles, spec)
    assert f"place arm {ir.placements[0].r} {ir.placements[0].q}" in text
    assert text.count("grab") == 2
    assert text.count("release") == 2
    assert reference_oms(spec) == text


def test_hand_ring_matches_simulator():
    """Rotation 0 is (-1, +1) from the base and CLW is +1; pin the table against a real arm."""
    spec = single_atom_spec(Hex(0, 2), Hex(2, 0))
    base = Hex(1, 1)
    for rotation in range(6):
        level = Level(spec)
        arm = level.place("arm", base, rotation=rotation)
        hand = level.sim.components[arm.comp_id].get_hand_pos
        assert (hand.r - base.r, hand.s - base.s) == RING_OFFSETS[rotation]


def test_spec_shape_contract():
    spec = single_atom_spec(Hex(0, 0), Hex(0, 2))
    assert spec.size == (6, 6)
    assert spec.entrypoints_fixed is True
    assert spec.target == 2
    assert spec.cycling is False
    assert spec.allowed_blueprints == frozenset({"arm"})
    assert [ep.pos for ep in spec.entrypoints] == [Hex(0, 0), Hex(0, 2)]
    assert all(len(shape.cells) == 1 and not shape.links for shape in spec.shapes)


def test_spec_params_are_honoured():
    spec = single_atom_spec(Hex(0, 0), Hex(0, 2), target=3, steps=40)
    assert spec.target == 3
    assert spec.steps == 40
    assert verify(spec) > 0


@pytest.mark.parametrize(
    ("entrance", "exit_"),
    [
        (Hex(0, 0), Hex(0, 0)),  # same cell
        (Hex(0, 0), Hex(0, 3)),  # too far
        (Hex(0, 0), Hex(2, 2)),  # too far diagonally
        (Hex(0, 0), Hex(0, 6)),  # out of board
        (Hex(-1, 0), Hex(0, 2)),  # out of board
    ],
)
def test_invalid_pairs_raise(entrance, exit_):
    with pytest.raises(FamilyError):
        single_atom_spec(entrance, exit_)


def test_arm_plan_is_deterministic():
    spec = single_atom_spec(Hex(1, 3), Hex(2, 4))
    first, second = arm_plan(spec), arm_plan(spec)
    assert first == second
    assert first is not None
    assert hex_distance(first.base, Hex(1, 3)) == 1
    assert hex_distance(first.base, Hex(2, 4)) == 1


def test_encoder_distinguishes_entry_positions():
    tokenizer = OpusTokenizer()
    first = single_atom_spec(Hex(0, 0), Hex(0, 2))
    same = single_atom_spec(Hex(0, 0), Hex(0, 2))
    other = single_atom_spec(Hex(2, 2), Hex(2, 4))
    a, a_len = tokenizer.encode_level_spec(first)
    b, b_len = tokenizer.encode_level_spec(same)
    c, c_len = tokenizer.encode_level_spec(other)
    assert a.tolist() == b.tolist()
    assert a_len == b_len == c_len
    assert a.tolist() != c.tolist()
    # header mode token mirrors `spec.entrypoints_fixed`; names follow the flag
    vocab = tokenizer.vocab
    assert a[0] == vocab.value_id("mode", int(first.entrypoints_fixed))


def test_verify_rejects_unfixed_spec():
    fixed = single_atom_spec(Hex(0, 0), Hex(0, 2))
    unfixed = LevelSpec(
        size=fixed.size,
        shapes=fixed.shapes,
        entrypoints=tuple(EntrypointSpec(kind=ep.kind, shape=ep.shape) for ep in fixed.entrypoints),
        allowed_blueprints=fixed.allowed_blueprints,
        steps=fixed.steps,
        target=fixed.target,
    )
    assert not unfixed.entrypoints_fixed
    with pytest.raises(FamilyError):
        reference_oms(unfixed)


# --- corpus sweep (marked, skipped unless `pytest --corpus`) ----------------------


@pytest.mark.corpus
def test_sweep_tier1_short_pairs_solve():
    """
    Every ordered pair in the family's domain (hex-distance 1-2) builds and verifies.

    Pins the tier-1 domain against the board geometry: 170 distance-1 + 258
    distance-2 pairs, target 2, no FamilyError anywhere.
    """
    cells = [Hex(row, col) for row in range(6) for col in range(6)]
    counts: dict[int, int] = {}
    for entrance in cells:
        for exit_ in cells:
            distance = hex_distance(entrance, exit_)
            if entrance == exit_ or distance > 2:
                continue
            spec = single_atom_spec(entrance, exit_)
            assert spec.target == 2
            assert verify(spec) > 0
            counts[distance] = counts.get(distance, 0) + 1
    assert counts == {1: 170, 2: 258}
