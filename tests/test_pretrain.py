"""Stage-1 data path: `.oms` solution files to validated token sequences."""

import pytest
from conftest import known_solution_tokens, make_spec
from opus_magnum_sym import Level, StepActionType

from opus_magnum_blackbox.codec import apply_solution, decode_solution
from opus_magnum_blackbox.corpus import LEVELS_DIR
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.grammar import Action, GrammarError, Placement, spec_model
from opus_magnum_blackbox.solutions import SolutionLoadError, augment, load_solutions

TIER0_OML = LEVELS_DIR / "tier0_typed_delivery.oml"
TIER0_OMS = LEVELS_DIR / "tier0_typed_delivery.oms"


def tier0_entry() -> LevelEntry:
    return LevelEntry(source=TIER0_OML, cost_max=50, name="tier0_typed_delivery", solutions=(TIER0_OMS,))


def test_encode_solution_matches_known_tokens(tokenizer, vocab, spec):
    model = spec_model(spec)
    arm = vocab.builtin(vocab.builtin_id("arm"))
    placements = [
        Placement(
            builtin=None,
            slot=model.slots.index("entrance_typed"),
            template=model.templates[0],
            rotation=1,
            q=3,
            r=2,
            pos=0,
        ),
        Placement(
            builtin=None, slot=model.slots.index("exit_typed"), template=model.templates[1], rotation=2, q=2, r=3, pos=0
        ),
        Placement(builtin=arm, slot=None, template=arm.template, rotation=0, q=3, r=3, pos=0),
    ]
    actions = [
        Action(handle=0, tick=tick, action=step, pos=0)
        for tick, step in [
            (1, StepActionType.GRAB),
            (2, StepActionType.ROTATE_CLW),
            (3, StepActionType.ROTATE_CLW),
            (4, StepActionType.ROTATE_CLW),
            (5, StepActionType.ROTATE_CLW),
            (6, StepActionType.RELEASE),
        ]
    ]
    assert tokenizer.encode_solution(placements, actions, spec) == known_solution_tokens(vocab)


def test_loaded_solution_is_known_and_solves(vocab):
    (tokens,) = load_solutions(tier0_entry())
    assert tokens == known_solution_tokens(vocab)

    level = Level.from_file(TIER0_OML)
    decoded = decode_solution(tokens, level.spec, vocab)
    assert not isinstance(decoded, GrammarError)
    assert apply_solution(decoded, level.spec, level).ok
    assert level.run().solved


def test_loaded_tokens_mask_valid_at_prefixes(masker):
    (tokens,) = load_solutions(tier0_entry())
    spec = Level.from_file(TIER0_OML).spec
    for position, token in enumerate(tokens):
        assert masker.mask(tokens[:position], spec)[token]


def test_augmented_variants_solve_and_are_deterministic(vocab, spec):
    tokens = known_solution_tokens(vocab)
    variants = augment(tokens, spec, seed=0, max_variants=6)
    assert variants
    assert len(variants) <= 6
    assert all(variant != tokens for variant in variants)
    for variant in variants:
        decoded = decode_solution(variant, spec, vocab)
        assert not isinstance(decoded, GrammarError)
        level = Level(spec)
        assert apply_solution(decoded, spec, level).ok
        assert level.run().solved
    assert augment(tokens, spec, seed=0, max_variants=6) == variants


def test_unsupported_command_raises(tmp_path):
    path = tmp_path / "unsupported.oms"
    path.write_text("# only place/action are supported\ncomp 1 2 ARM\n")
    entry = LevelEntry(source=make_spec(), cost_max=50, name="unsupported", solutions=(path,))
    with pytest.raises(SolutionLoadError, match="unsupported command 'comp'"):
        load_solutions(entry)
