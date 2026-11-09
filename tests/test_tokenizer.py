"""Tokenizer: round-trip acceptance, coordinate mapping, entrypoint routing."""

from conftest import known_solution_tokens
from opus_magnum_sym import Hex, Level
from opus_magnum_sym.levels.result import LevelStatus

from opus_magnum_blackbox.codec import apply_solution, decode_solution, solution_to_oms
from opus_magnum_blackbox.grammar import Solution


def test_encode_spec_pads_to_fixed_length(spec, tokenizer):
    encoded, length = tokenizer.encode_level_spec(spec)
    assert encoded.shape == (tokenizer.config.max_encoder_len,)
    assert 0 < length <= len(encoded)
    assert (encoded[length:] == tokenizer.vocab.pad_id).all()


def test_round_trip_known_solution_solves(spec, tokenizer):
    level = Level(spec)
    report = tokenizer.decode_solution_to_level(known_solution_tokens(tokenizer.vocab), spec, level)
    assert report.ok, report.error
    result = level.run()
    assert result.status is LevelStatus.COMPLETED
    assert result.solved
    assert result.total_finished == 1


def test_decode_solution_is_pure(spec, tokenizer):
    sol = decode_solution(known_solution_tokens(tokenizer.vocab), spec, tokenizer.vocab)
    assert isinstance(sol, Solution)
    assert sol.submitted
    assert len(sol.placements) == 3
    assert len(sol.actions) == 6
    assert sol.actions[0].handle == 0
    assert sol.actions[0].tick == 1


def test_coordinate_mapping_q_is_column_r_is_row(spec, tokenizer):
    vocab = tokenizer.vocab
    tokens = [
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
        vocab.value_id("q", 4),
        vocab.value_id("r", 0),
        vocab.special_id("[END_LAYOUT]"),
        vocab.special_id("[START_PROGRAM]"),
        vocab.special_id("[SUBMIT]"),
    ]
    level = Level(spec)
    report = tokenizer.decode_solution_to_level(tokens, spec, level)
    assert report.ok, report.error
    arm_record = level.sim.placements[-1]
    assert arm_record.blueprint == "arm"
    assert arm_record.pos == Hex(r=0, s=4)


def test_entrypoint_slots_route_through_place_entrypoint(spec, tokenizer):
    level = Level(spec)
    report = tokenizer.decode_solution_to_level(known_solution_tokens(tokenizer.vocab), spec, level)
    assert report.ok, report.error
    placed = {record.blueprint for record in level.sim.placements}
    assert {"entrance_typed", "exit_typed", "arm"} <= placed
    # Correct routing means the unfixed entrypoint check passes at run time.
    assert level.run().status is LevelStatus.COMPLETED


def test_decode_static_failure_on_garbage(spec, tokenizer):
    level = Level(spec)
    report = tokenizer.decode_solution_to_level([tokenizer.vocab.value_id("tick", 3)] * 5, spec, level)
    assert not report.ok
    assert report.error_token_pos == 0


def test_decode_static_failure_on_illegal_placement(spec, tokenizer):
    vocab = tokenizer.vocab
    tokens = [
        vocab.special_id("[START_LAYOUT]"),
        vocab.value_id("entry_slot", 0),
        vocab.value_id("rotation", 1),
        vocab.value_id("q", 3),
        vocab.value_id("r", 2),
        vocab.value_id("entry_slot", 1),
        vocab.value_id("rotation", 2),
        vocab.value_id("q", 2),
        vocab.value_id("r", 3),
        # out of bounds for a 6x6 board
        vocab.builtin_id("arm"),
        vocab.value_id("rotation", 0),
        vocab.value_id("q", 9),
        vocab.value_id("r", 9),
        vocab.special_id("[END_LAYOUT]"),
        vocab.special_id("[START_PROGRAM]"),
        vocab.special_id("[SUBMIT]"),
    ]
    level = Level(spec)
    report = tokenizer.decode_solution_to_level(tokens, spec, level)
    assert not report.ok
    assert report.error_token_pos == 9


KNOWN_SOLUTION_OMS = """\
place entrance_typed 2 3 rotation=1
place exit_typed 3 2 rotation=2
place arm 3 3

action 4 1 grab
action 4 2 rotate_clw
action 4 3 rotate_clw
action 4 4 rotate_clw
action 4 5 rotate_clw
action 4 6 release
"""


def test_solution_to_oms_renders_reference_text(spec, tokenizer):
    sol = decode_solution(known_solution_tokens(tokenizer.vocab), spec, tokenizer.vocab)
    level = Level(spec)
    report = apply_solution(sol, spec, level)
    assert report.ok, report.error
    assert solution_to_oms(sol, report.handles, spec) == KNOWN_SOLUTION_OMS


def test_solution_to_oms_round_trips_through_load_solution(spec, tokenizer, tmp_path):
    sol = decode_solution(known_solution_tokens(tokenizer.vocab), spec, tokenizer.vocab)
    level = Level(spec)
    report = apply_solution(sol, spec, level)
    assert report.ok, report.error

    path = tmp_path / "generated.oms"
    path.write_text(solution_to_oms(sol, report.handles, spec))
    fresh = Level(spec)
    fresh.load_solution(path)
    result = fresh.run()
    assert result.status is LevelStatus.COMPLETED
    assert result.solved
