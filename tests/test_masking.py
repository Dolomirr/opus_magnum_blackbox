"""ActionMasker: per-phase mask rules."""

import numpy as np
from conftest import known_solution_tokens, make_spec
from opus_magnum_sym import Hex, mkentr, mkexit

from opus_magnum_blackbox.masking import ActionMasker


def masked_names(masker, history, spec):
    mask = masker.mask(history, spec)
    return {masker.vocab.name(i) for i in np.flatnonzero(mask)}


def test_start_phase_allows_only_start_layout(masker, spec):
    assert masked_names(masker, [], spec) == {"[START_LAYOUT]"}


def test_layout_name_position(masker, spec):
    names = masked_names(masker, [masker.vocab.special_id("[START_LAYOUT]")], spec)
    assert "BP_arm" in names
    assert {"ESHAPE_0", "ESHAPE_1"} <= names
    assert "[END_LAYOUT]" not in names  # entrypoints unsatisfied
    assert "[SUBMIT]" not in names


def test_fixed_mode_masks_entrypoint_slots(masker, vocab):
    spec = make_spec(
        entrypoints=(mkentr("entrance_typed", Hex(2, 3), rotation=1), mkexit("exit_typed", Hex(3, 2), rotation=2)),
        entrypoints_fixed=True,
    )
    names = masked_names(masker, [vocab.special_id("[START_LAYOUT]")], spec)
    assert "BP_arm" in names
    assert not any(name.startswith("ESHAPE_") for name in names)
    assert "[END_LAYOUT]" in names  # fixed entrypoints are pre-placed


def test_entrypoint_completeness_gates_end_layout(masker, spec, vocab):
    history = [
        vocab.special_id("[START_LAYOUT]"),
        vocab.value_id("entry_slot", 0),
        vocab.value_id("rotation", 1),
        vocab.value_id("q", 3),
        vocab.value_id("r", 2),
    ]
    names = masked_names(masker, history, spec)
    assert "ESHAPE_0" not in names  # satisfied slot masked off
    assert "ESHAPE_1" in names
    assert "[END_LAYOUT]" not in names

    history += [
        vocab.value_id("entry_slot", 1),
        vocab.value_id("rotation", 2),
        vocab.value_id("q", 2),
        vocab.value_id("r", 3),
    ]
    names = masked_names(masker, history, spec)
    assert not any(name.startswith("ESHAPE_") for name in names)
    assert "[END_LAYOUT]" in names


def test_rotation_restrictions(masker, vocab):
    spec = make_spec(allowed_blueprints=frozenset({"arm", "connector_tri_odd"}))
    history = [vocab.special_id("[START_LAYOUT]"), vocab.builtin_id("connector_tri_odd")]
    names = masked_names(masker, history, spec)
    assert {name for name in names if name.startswith("ROT_")} == {"ROT_1", "ROT_3", "ROT_5"}

    history = [vocab.special_id("[START_LAYOUT]"), vocab.builtin_id("arm")]
    names = masked_names(masker, history, spec)
    assert {name for name in names if name.startswith("ROT_")} == {f"ROT_{i}" for i in range(6)}


def test_coordinate_masks_out_of_bounds_and_overlap(masker, spec, vocab):
    # Arm base occupies (r=3, s=3); a second arm with Q_3 must not take R_3.
    history = [
        vocab.special_id("[START_LAYOUT]"),
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
        vocab.value_id("q", 3),
        vocab.value_id("r", 3),
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
        vocab.value_id("q", 3),
    ]
    names = masked_names(masker, history, spec)
    assert "R_3" not in names
    assert "R_4" in names
    assert not any(name.startswith("R_") and int(name[2:]) >= 6 for name in names)


def test_coordinate_masks_out_of_board_columns(masker, spec, vocab):
    history = [
        vocab.special_id("[START_LAYOUT]"),
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
    ]
    names = masked_names(masker, history, spec)
    assert {name for name in names if name.startswith("Q_")} == {f"Q_{q}" for q in range(6)}


def test_beam_shape_q_mask_requires_some_valid_r(masker, spec, vocab):
    # Straight-2 shape cells: origin + beam; rot 0 beam is at (r, s+1)... at column edge no row fits.
    history = [
        vocab.special_id("[START_LAYOUT]"),
        vocab.value_id("entry_slot", 0),
        vocab.value_id("rotation", 1),
    ]
    names = masked_names(masker, history, spec)
    # rot 1 puts the beam at (r, s+1), so the rightmost column fits no row.
    q_allowed = {int(name[2:]) for name in names if name.startswith("Q_")}
    assert q_allowed == {0, 1, 2, 3, 4}


def test_program_masks(masker, spec, vocab):
    history = known_solution_tokens(vocab)
    # Drop the trailing SUBMIT: next position is a handle position.
    names = masked_names(masker, history[:-1], spec)
    assert names == {"H_0", "[SUBMIT]"}

    # Tick position: ticks already used for the current handle are masked off.
    history = history[:-4]  # cut the release triplet and SUBMIT
    names = masked_names(masker, history, spec)
    assert names == {"H_0", "[SUBMIT]"}
    names = masked_names(masker, [*history, vocab.value_id("handle", 0)], spec)
    tick_names = {name for name in names if name.startswith("TICK_")}
    assert "TICK_6" in tick_names
    assert not {"TICK_1", "TICK_2", "TICK_3", "TICK_4", "TICK_5"} & tick_names
    assert "TICK_30" not in tick_names  # beyond spec.steps


def test_exhausted_handle_not_offered(masker, spec, vocab):
    """
    A handle with every tick burned must not be offered at a group start:
    its triplet could never complete, leaving the next mask empty.
    """
    spec = make_spec(steps=4)
    history = [
        vocab.special_id("[START_LAYOUT]"),
        vocab.value_id("entry_slot", 0),
        vocab.value_id("rotation", 1),
        vocab.value_id("q", 3),
        vocab.value_id("r", 2),
        vocab.value_id("entry_slot", 1),
        vocab.value_id("rotation", 2),
        vocab.value_id("q", 2),
        vocab.value_id("r", 3),
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
        vocab.value_id("q", 3),
        vocab.value_id("r", 3),
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
        vocab.value_id("q", 4),
        vocab.value_id("r", 4),
        vocab.special_id("[END_LAYOUT]"),
        vocab.special_id("[START_PROGRAM]"),
    ]
    names = masked_names(masker, history, spec)
    assert names == {"H_0", "H_1", "[SUBMIT]"}

    for tick in range(4):  # burn every tick of handle 0
        history += [vocab.value_id("handle", 0), vocab.value_id("tick", tick), vocab.id("ACT_ROTATE_CLW")]
    names = masked_names(masker, history, spec)
    assert "H_0" not in names
    assert "H_1" in names
    assert "[SUBMIT]" in names


def test_program_canonical_handle_order(masker, spec, vocab):
    spec2 = make_spec(allowed_blueprints=frozenset({"arm"}))
    history = [
        vocab.special_id("[START_LAYOUT]"),
        vocab.value_id("entry_slot", 0),
        vocab.value_id("rotation", 1),
        vocab.value_id("q", 3),
        vocab.value_id("r", 2),
        vocab.value_id("entry_slot", 1),
        vocab.value_id("rotation", 2),
        vocab.value_id("q", 2),
        vocab.value_id("r", 3),
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
        vocab.value_id("q", 3),
        vocab.value_id("r", 3),
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
        vocab.value_id("q", 4),
        vocab.value_id("r", 4),
        vocab.special_id("[END_LAYOUT]"),
        vocab.special_id("[START_PROGRAM]"),
        vocab.value_id("handle", 1),
        vocab.value_id("tick", 0),
        vocab.id("ACT_GRAB"),
    ]
    names = masked_names(masker, history, spec2)
    assert "H_1" in names
    assert "H_0" not in names  # earlier handles masked once a higher one is used


def test_extend_retract_only_for_piston(masker, vocab):
    spec = make_spec(allowed_blueprints=frozenset({"arm", "arm_piston"}))

    def action_names(blueprint):
        history = [
            vocab.special_id("[START_LAYOUT]"),
            vocab.value_id("entry_slot", 0),
            vocab.value_id("rotation", 1),
            vocab.value_id("q", 3),
            vocab.value_id("r", 2),
            vocab.value_id("entry_slot", 1),
            vocab.value_id("rotation", 2),
            vocab.value_id("q", 2),
            vocab.value_id("r", 3),
            vocab.builtin_id(blueprint),
            vocab.value_id("rotation", 0),
            vocab.value_id("q", 3),
            vocab.value_id("r", 3),
            vocab.special_id("[END_LAYOUT]"),
            vocab.special_id("[START_PROGRAM]"),
            vocab.value_id("handle", 0),
            vocab.value_id("tick", 1),
        ]
        return masked_names(masker, history, spec)

    plain = action_names("arm")
    assert "ACT_EXTEND" not in plain
    assert "ACT_RETRACT" not in plain
    assert "ACT_GRAB" in plain
    assert "ACT_NOP" in plain

    piston = action_names("arm_piston")
    assert "ACT_EXTEND" in piston
    assert "ACT_RETRACT" in piston


def test_done_phase_masks_all_but_pad(masker, spec, vocab):
    names = masked_names(masker, known_solution_tokens(vocab), spec)
    assert names == {"[PAD]"}


def test_broken_history_yields_pad_only(masker, spec, vocab):
    names = masked_names(masker, [vocab.value_id("tick", 0), vocab.value_id("q", 1)], spec)
    assert names == {"[PAD]"}
