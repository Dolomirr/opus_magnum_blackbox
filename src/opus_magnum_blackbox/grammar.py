"""
Incremental DSL parser shared for masker and codec.

Grammar (in order in which tokens should be produced):

| Token / stage        |                        |
|----------------------|------------------------|
| [START_LAYOUT]       |                        |
| <layout quadruplets> | (name, rot, q, r)      |
| [END_LAYOUT]         |                        |
| [START_PROGRAM]      |                        |
| <program triplets>   | (handle, tick, action) |
| [SUBMIT]             |                        |

Trailing/incomplete group is ignored and already existing group is evaluated.

`advance` is the only code that knows group structure and phase rules;
both `ActionMasker` and `decode_solution` runs only on token history.
"""

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from functools import cache

from opus_magnum_sym import LevelSpec, StepActionType
from opus_magnum_sym.elements.components.blueprints import ShapeTemplate

from opus_magnum_blackbox.vocab import Builtin, Vocab

LAYOUT_GROUP = 4
PROGRAM_GROUP = 3


class Phase(Enum):
    # fmt: off
    START       = auto()
    LAYOUT      = auto()
    LAYOUT_DONE = auto()
    PROGRAM     = auto()
    DONE        = auto()
    # fmt: on


@dataclass(frozen=True)
class Placement:
    """
    Built-in blueprint or entrypoint slot.
    """

    # fmt: off
    builtin: Builtin | None        # None for entrypoint slots
    slot: int | None               # entrypoint shape slot, None for built-ins
    template: ShapeTemplate | None # footprint geometry (entrypoint or built-in)
    rotation: int
    q: int
    r: int
    pos: int                       # token position of the quadriplet first token
    # fmt: on


@dataclass(frozen=True)
class Action:
    """
    Real (non-NOP) assigns action.
    """

    # fmt: off
    handle : int
    tick   : int
    action : StepActionType
    pos    : int            # token position of triplet's first token
    # fmt: on


@dataclass(frozen=True)
class Solution:
    # fmt: off
    placements : list[Placement]
    actions    : list[Action]
    submitted  : bool            # `[SUBMIT]` reached
    length     : int             # tokens consumed by parser
    # fmt: on


@dataclass
class ParseState:
    # fmt: off
    phase         : Phase           = Phase.START
    length        : int             = 0                           # tokens consumed so far
    group_start   : int             = 0                           # token position of the first token of the current group
    pending       : list[int]       = field(default_factory=list) # tokens of current incomplete group
    placements    : list[Placement] = field(default_factory=list)
    slot_use      : dict[int, int]  = field(default_factory=dict)
    arms          : list[Builtin]   = field(default_factory=list) # per program handle
    actions       : list[Action]    = field(default_factory=list)
    used_ticks    : list[set[int]]  = field(default_factory=list) # per handle
    current_handle: int             = 0
    # fmt: on


@dataclass(frozen=True)
class GrammarError:
    """
    Note this is not exception, but dataclass helding the position and reason.
    """

    pos: int
    message: str


def solution(state: ParseState) -> Solution:
    return Solution(
        placements=list(state.placements),
        actions=list(state.actions),
        submitted=state.phase is Phase.DONE,
        length=state.length,
    )


@dataclass(frozen=True)
class SpecModel:
    """
    Per-spec view of layout allow-lists, shared by masker and codec.
    """

    # fmt: off
    slots       : tuple[str, ...]                  # unique entrypoint shape names in first-occurrence order
    slot_indices: tuple[tuple[int, ...], ...]      # spec entrypoint indices per slot
    required    : tuple[int, ...]                  # placements needed per slot (0 when entrypoints are fixed)
    builtins    : tuple[str, ...]                  # allowed built-in blueprint names
    templates   : tuple[ShapeTemplate | None, ...] # template per entrypoint slot
    # fmt: on


def spec_model(spec: LevelSpec) -> SpecModel:
    # LevelSpec hashes in frozenset, these hashes are randomized per process;
    # keying on its scalar fields keeps the cache effective across runs.
    return _spec_model(spec, spec.steps, spec.target, spec.entrypoints_fixed)


@cache
def _spec_model(spec: LevelSpec, steps: int, target: int | None, entrypoints_fixed: bool) -> SpecModel:
    shapes = {shape.name: shape for shape in spec.shapes}
    slots: list[str] = []
    for ep in spec.entrypoints:
        if ep.shape not in slots:
            slots.append(ep.shape)
    slot_indices = tuple(tuple(i for i, ep in enumerate(spec.entrypoints) if ep.shape == name) for name in slots)
    required = (0,) * len(slots) if spec.entrypoints_fixed else tuple(len(i) for i in slot_indices)
    return SpecModel(
        slots=tuple(slots),
        slot_indices=slot_indices,
        required=required,
        builtins=tuple(name for name in spec.allowed_blueprints if name not in shapes),
        templates=tuple(shapes.get(name) for name in slots),
    )


def entrypoint_index(model: SpecModel, slot: int, use: int) -> int:
    return model.slot_indices[slot][use]


def advance(state: ParseState, token: int, spec: LevelSpec, vocab: Vocab) -> ParseState | GrammarError:
    """
    Feeds one token to parser.

    Returns the updated state,
    or `GrammarError` at offending position (the token itself).

    Never mutates state.
    """
    if state.phase is Phase.DONE:
        return GrammarError(state.length, "sequence already terminated with [SUBMIT]")

    if not state.pending:
        boundary = _boundary(state, token, spec, vocab)
        if isinstance(boundary, GrammarError):
            return boundary
        if boundary is not None:
            return replace(boundary, length=state.length + 1)

    group = [*state.pending, token]
    group_start = state.group_start if state.pending else state.length

    if state.phase is Phase.LAYOUT:
        updated = _layout_group(group, group_start, state, spec, vocab)
    else:
        updated = _program_group(group, group_start, state, vocab)
    if isinstance(updated, GrammarError):
        return updated

    group_length = LAYOUT_GROUP if state.phase is Phase.LAYOUT else PROGRAM_GROUP
    pending = group if len(group) < group_length else []

    return replace(
        updated,
        pending=pending,
        group_start=group_start,
        length=state.length + 1,
    )


def parse(tokens, spec: LevelSpec, vocab: Vocab) -> ParseState | GrammarError:
    """
    Process `advance` over full token history.
    """
    state = ParseState()
    for token in tokens:
        state = advance(state, int(token), spec, vocab)
        if isinstance(state, GrammarError):
            return state
    return state


def _boundary(  # noqa: PLR0911
    state: ParseState,
    token: int,
    spec: LevelSpec,
    vocab: Vocab,
) -> ParseState | GrammarError | None:
    """
    Handle `token` when it opens a new group.

    Returns the next state, an error, or None when the token opens a layout
    quadruplet (which the caller appends to the group).
    """
    # TODO: try to make this less of abomination
    pos = state.length
    if state.phase is Phase.START:
        if token != vocab.special_id("[START_LAYOUT]"):
            return GrammarError(pos, "sequence must start with [START_LAYOUT]")
        return replace(state, phase=Phase.LAYOUT)
    if state.phase is Phase.LAYOUT_DONE:
        if token != vocab.special_id("[START_PROGRAM]"):
            return GrammarError(pos, "program segment must start with [START_PROGRAM]")
        return replace(state, phase=Phase.PROGRAM)
    if state.phase is Phase.PROGRAM:
        if token == vocab.special_id("[SUBMIT]"):
            return replace(state, phase=Phase.DONE)
        if not vocab.is_in(token, "handle"):
            return GrammarError(pos, f"expected program triplet, got {vocab.name(token)}")
        handle = token - vocab.base("handle")
        if handle >= len(state.arms):
            return GrammarError(pos, f"program references unplaced arm handle {handle}")
        return None
    if token == vocab.special_id("[END_LAYOUT]"):
        model = spec_model(spec)
        if any(state.slot_use.get(slot, 0) < required for slot, required in enumerate(model.required)):
            return GrammarError(pos, "[END_LAYOUT] before every entrypoint shape was placed")
        return replace(state, phase=Phase.LAYOUT_DONE)
    if vocab.is_in(token, "entry_slot"):
        if token - vocab.base("entry_slot") >= len(spec_model(spec).slots):
            return GrammarError(pos, f"entrypoint slot token {vocab.name(token)} out of range for the spec")
        return None
    if not vocab.is_in(token, "builtin"):
        return GrammarError(pos, f"expected layout quadruplet, got {vocab.name(token)}")
    return None


def _layout_group(
    group: list[int],
    group_start: int,
    state: ParseState,
    spec: LevelSpec,
    vocab: Vocab,
) -> ParseState | GrammarError:
    """
    Validate in-group token at `len(group) - 1`;
    append the placement when the group closes.
    """
    expected = (("builtin", "entry_slot"), ("rotation",), ("q",), ("r",))
    token = group[-1]
    if not any(vocab.is_in(token, category) for category in expected[len(group) - 1]):
        return GrammarError(group_start, f"expected layout quadruplet, got {[vocab.name(t) for t in group]}")
    if len(group) < LAYOUT_GROUP:
        return state

    slot = group[0] - vocab.base("entry_slot") if vocab.is_in(group[0], "entry_slot") else None
    if slot is not None:
        model = spec_model(spec)
        builtin = None
        template = model.templates[slot]
    else:
        builtin = vocab.builtin(group[0])
        template = builtin.template
    # fmt: off
    placement = Placement(
        builtin  = builtin,
        slot     = slot,
        template = template,
        rotation = group[1] - vocab.base("rotation"),
        q        = group[2] - vocab.base("q"),
        r        = group[3] - vocab.base("r"),
        pos      = group_start,
    )
    # fmt: on
    placements = [*state.placements, placement]
    if slot is not None:
        return replace(state, placements=placements, slot_use={**state.slot_use, slot: state.slot_use.get(slot, 0) + 1})
    if builtin.manipulator:
        return replace(state, placements=placements, arms=[*state.arms, builtin])
    return replace(state, placements=placements)


def _program_group(  # noqa: PLR0911
    group: list[int],
    group_start: int,
    state: ParseState,
    vocab: Vocab,
) -> ParseState | GrammarError:
    """
    Validate the in-group token at `len(group) - 1`;
    record the action when the group closes.
    """
    token = group[-1]
    if len(group) == 2 and not vocab.is_in(token, "tick"):
        return GrammarError(group_start, f"expected program triplet, got {[vocab.name(t) for t in group]}")
    if len(group) == 3 and not vocab.is_in(token, "action"):
        return GrammarError(group_start, f"expected program triplet, got {[vocab.name(t) for t in group]}")
    if len(group) < PROGRAM_GROUP:
        return state

    handle = group[0] - vocab.base("handle")
    tick = group[1] - vocab.base("tick")
    if handle < state.current_handle:
        return GrammarError(
            group_start, f"handle {handle} is out of canonical order after handle {state.current_handle}"
        )
    current_handle = max(state.current_handle, handle)
    step_type = vocab.step_type(group[2])
    if step_type is None:  # NOP: assign nothing at this tick
        return replace(state, current_handle=current_handle)
    if tick in (state.used_ticks[handle] if handle < len(state.used_ticks) else ()):
        return GrammarError(group_start, f"tick {tick} already used for handle {handle}")
    used_ticks = [*state.used_ticks] + [set() for _ in range(len(state.arms) - len(state.used_ticks))]
    used_ticks[handle] = used_ticks[handle] | {tick}
    action = Action(handle=handle, tick=tick, action=step_type, pos=group_start)
    return replace(state, actions=[*state.actions, action], used_ticks=used_ticks, current_handle=current_handle)
