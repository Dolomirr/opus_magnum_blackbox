"""
Wire guard for the arm_l2..arm_l6 insertion: pure insertion, documented shift.

The builtin category sits mid-vocab, so registering the five static-length arm
blueprints shifted every later category by +5. These tests pin that event: the
inserted block is contiguous right after `BP_arm_piston`, no pre-existing token
changed meaning, and ids move only by the exact documented shift.
"""

from opus_magnum_blackbox import vocab as vocab_module
from opus_magnum_blackbox.vocab import INSERTED_BUILTINS, Builtin, Vocab

# Builtin names in `BPN` order as they were before arm_l2..arm_l6 were registered.
PRE_INSERTION_BUILTINS = (
    "arm",
    "arm_piston",
    "connector_straight_2",
    "connector_tri",
    "connector_tri_odd",
    "salt_transmuter",
    "glyph_duplication",
)


def _pre_insertion_vocab(monkeypatch):
    """Vocab as it looked before the insertion, via the frozen builtin snapshot."""
    by_name = {b.name: b for b in vocab_module.BUILTINS}
    monkeypatch.setattr(vocab_module, "BUILTINS", tuple(by_name[name] for name in PRE_INSERTION_BUILTINS))
    return Vocab()


def test_inserted_block_sits_after_arm_piston(vocab):
    order = tuple(b.name for b in vocab_module.BUILTINS)
    assert order == (*PRE_INSERTION_BUILTINS[:2], *INSERTED_BUILTINS, *PRE_INSERTION_BUILTINS[2:])
    first = vocab.builtin_id("arm_piston") + 1
    assert [vocab.builtin_id(name) for name in INSERTED_BUILTINS] == list(
        range(first, first + len(INSERTED_BUILTINS))
    )


def test_static_arms_are_non_extendable_manipulators(vocab):
    for name in INSERTED_BUILTINS:
        assert vocab.builtin(vocab.builtin_id(name)) == Builtin(
            name=name, manipulator=True, extendable=False, template=None
        )


def test_insertion_is_pure(monkeypatch):
    new = Vocab()  # real registry; _pre_insertion_vocab patches BUILTINS for the rest of the test
    old = _pre_insertion_vocab(monkeypatch)
    inserted = frozenset(INSERTED_BUILTINS)

    assert len(new) == len(old) + len(inserted)
    assert tuple(old.ranges) == tuple(new.ranges)  # category order unchanged

    # Every pre-existing named token survives exactly once and moves by exactly
    # the number of inserted builtins placed before it.
    for token_id, name in enumerate(old._names):
        if name is None:
            continue
        assert new._names.count(name) == 1, name
        shift = sum(1 for other in inserted if new.id(other) < new.id(name))
        assert new.id(name) == token_id + shift, name

    # Category ranges follow the same shift rule on their start ids.
    for category, rng in old.ranges.items():
        shift = sum(1 for other in inserted if new.ranges[category].start > new.id(other))
        assert new.ranges[category].start == rng.start + shift, category
