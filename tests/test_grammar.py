"""Grammar parser: phase transitions, group validation, mask/decoder agreement."""

import numpy as np
from conftest import known_solution_tokens, make_spec

from opus_magnum_blackbox.grammar import GrammarError, ParseState, Phase, advance, parse


def start(vocab):
    return [vocab.special_id("[START_LAYOUT]")]


def test_empty_history_is_start_phase(spec, vocab):
    state = parse([], spec, vocab)
    assert isinstance(state, ParseState)
    assert state.phase is Phase.START


def test_missing_start_layout_is_rejected(spec, vocab):
    error = parse([vocab.value_id("q", 1)], spec, vocab)
    assert error == GrammarError(0, "sequence must start with [START_LAYOUT]")


def test_partial_layout_group_stays_pending(spec, vocab):
    history = [*start(vocab), vocab.builtin_id("arm"), vocab.value_id("rotation", 0)]
    state = parse(history, spec, vocab)
    assert isinstance(state, ParseState)
    assert state.phase is Phase.LAYOUT
    assert len(state.placements) == 0
    assert len(state.pending) == 2


def test_end_layout_gated_on_entrypoint_completeness(spec, vocab):
    error = parse([*start(vocab), vocab.special_id("[END_LAYOUT]")], spec, vocab)
    assert isinstance(error, GrammarError)
    assert error.pos == 1


def test_out_of_range_entrypoint_slot_is_rejected(spec, vocab):
    error = parse([*start(vocab), vocab.value_id("entry_slot", 5)], spec, vocab)
    assert isinstance(error, GrammarError)
    assert error.pos == 1


def test_canonical_handle_order_is_enforced(masker, vocab):
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
        vocab.value_id("handle", 0),
        vocab.value_id("tick", 1),
        vocab.id("ACT_GRAB"),
    ]
    error = parse(history, spec2, vocab)
    assert isinstance(error, GrammarError)
    assert "canonical order" in error.message


def test_unplaced_handle_is_rejected(spec, vocab):
    tokens = known_solution_tokens(vocab)[:-4]
    tokens += [vocab.value_id("handle", 3), vocab.value_id("tick", 7), vocab.id("ACT_GRAB")]
    error = parse(tokens, spec, vocab)
    assert isinstance(error, GrammarError)
    assert "unplaced arm handle 3" in error.message


def test_reused_tick_is_rejected(spec, vocab):
    tokens = known_solution_tokens(vocab)[:-4]
    tokens += [vocab.value_id("handle", 0), vocab.value_id("tick", 1), vocab.id("ACT_GRAB")]
    error = parse(tokens, spec, vocab)
    assert isinstance(error, GrammarError)
    assert "already used" in error.message


def test_submit_ends_sequence(spec, vocab):
    state = parse(known_solution_tokens(vocab), spec, vocab)
    assert isinstance(state, ParseState)
    assert state.phase is Phase.DONE
    assert advance(state, vocab.pad_id, spec, vocab).message == "sequence already terminated with [SUBMIT]"


def test_truncated_trailing_group_is_ignored(spec, vocab):
    state = parse(known_solution_tokens(vocab)[:-2], spec, vocab)
    assert isinstance(state, ParseState)
    assert len(state.pending) == 2
    # A truncated sequence still yields the actions of its complete triplets.
    assert len(state.actions) == 5


def test_masked_tokens_are_accepted_by_advance(spec, masker, vocab):
    for length in range(1, len(known_solution_tokens(vocab)) + 1):
        history = known_solution_tokens(vocab)[:length]
        state = parse(history, spec, vocab)
        assert not isinstance(state, GrammarError), state
        mask = masker.mask(history, spec)
        if state.phase is Phase.DONE:
            continue
        for token in np.flatnonzero(mask):
            assert not isinstance(advance(state, int(token), spec, vocab), GrammarError)
