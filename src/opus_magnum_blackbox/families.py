"""
Tier-1 single-atom family: fixed-entrypoint specs and reference solutions.

Spec:

- 6x6 board
- fixed 1-cell typed entrance and exit
- only `arm` allowed
- `cycling 0`
- default `target 2`

Geometry (measured against the simulator):

- arm at base `B` with rotation `r` puts its hand at `B + RING_OFFSETS[r]`;
  rotation 0 is `(-1, +1)` from the base and `ROTATE_CLW` is `r + 1`;
- the hand direction from `B` to a cell is `RING_OFFSETS.index(cell - B)`.

Cycle:

1) `GRAB`
2) `k` turns to exit
3) `RELEASE`
4) `6 - k` turns back

The entrance spawns on every step where its cell is free,
so held atom must never pass over the entrance.
Only `target` cycles are programmed, so the cycle must repeat exactly;
generator verifies each candidate with the simulator and
keeps the deterministic winner (lowest base, then clockwise, then shortest).

`tier-2 anywhere` family extends this to any distinct E != X pair on
board, with `{arm, arm_piston}` allow-list:
pistons reach "rings" 1-3 by leading their program with EXTEND actions,
and two-arm plans hand the atom over at shared ring cell, resulting in
release-then-grab timing.
"""

from dataclasses import dataclass

from opus_magnum_sym import (
    AtomType,
    ComponentType,
    Hex,
    Level,
    LevelSpec,
    ShapeCell,
    ShapeTemplate,
    StepActionType,
    mkentr,
    mkexit,
)

from opus_magnum_blackbox.codec import OpusTokenizer, apply_solution, decode_solution, solution_to_oms
from opus_magnum_blackbox.config import TokenizerConfig, VocabConfig
from opus_magnum_blackbox.grammar import Action, GrammarError, Placement, Solution
from opus_magnum_blackbox.masking import ActionMasker
from opus_magnum_blackbox.vocab import Vocab

ENTRANCE_SHAPE = "entrance_1"
EXIT_SHAPE = "exit_1"

# rotation -> (dr, ds) hand offset around the arm base; rotation 0 is (-1, +1), CLW is +1
RING_OFFSETS = ((-1, 1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0))

# one delivery cycle is GRAB + k turns + RELEASE + (6 - k) turns back: always 8 actions
CYCLE_ACTIONS = 8

_GRAB = StepActionType.GRAB
_RELEASE = StepActionType.RELEASE
_CLW = StepActionType.ROTATE_CLW
_CCLW = StepActionType.ROTATE_CCLW
_EXT = StepActionType.EXTEND
_RET = StepActionType.RETRACT


class FamilyError(ValueError):
    """Entrance/exit pair cannot be served by the single-arm family."""


@dataclass(frozen=True)
class ArmPlan:
    """
    One arm placement plus the program that delivers `target` molecules.
    """

    base: Hex
    rotation: int  # starting rotation: the hand rests on the entrance
    program: tuple[StepActionType, ...]  # one action per step; step 0 is the first grab


def hex_distance(a: Hex, b: Hex) -> int:
    return (abs(a.r - b.r) + abs(a.s - b.s) + abs(a.r + a.s - b.r - b.s)) // 2


# rows, cols of the rectangular board a spec is cut from
type BoardSize = tuple[int, int]


def single_atom_spec(  # noqa: PLR0913
    entrance: Hex,
    exit: Hex,
    *,
    target: int = 2,
    steps: int = 30,
    size: BoardSize = (6, 6),
    atom: AtomType = AtomType.WATER,
) -> LevelSpec:
    """
    Fixed-entrypoint spec for one arm delivering `target` typed single atoms entrance -> exit.

    Raises `FamilyError` for every pair outside the family: out of board, degenerate,
    farther than hex-distance 2, or with no verified delivery cycle.
    """
    rows, cols = size
    for name, cell in (("entrance", entrance), ("exit", exit)):
        if not (0 <= cell.r < rows and 0 <= cell.s < cols):
            raise FamilyError(f"{name} cell {tuple(cell)} is out of the {rows}x{cols} board")
    if entrance == exit:
        raise FamilyError("entrance and exit must not share a cell")
    distance = hex_distance(entrance, exit)
    if distance not in (1, 2):
        raise FamilyError(
            f"entrance {tuple(entrance)} and exit {tuple(exit)} are {distance} apart; "
            "a single arm needs hex-distance 1 or 2"
        )
    # parameter combos that would silently break training:
    # `max_program_actions` would truncate mid-program, steps over `max_steps` exceed the tick tokens
    program_actions = CYCLE_ACTIONS * target
    max_actions = TokenizerConfig().max_program_actions
    if program_actions > max_actions:
        raise FamilyError(
            f"target {target} programs {program_actions} actions, over the {max_actions} program-action cap"
        )
    max_steps = VocabConfig().max_steps
    if steps > max_steps:
        raise FamilyError(f"steps {steps} is over the {max_steps} tick-token cap")
    # actions land on ticks 1..program_actions, and tick tokens stop at steps - 1
    if program_actions >= steps:
        raise FamilyError(f"program of {program_actions} actions does not fit into the {steps}-step budget")
    spec = LevelSpec(
        size=size,
        shapes=(
            _cell_shape(ENTRANCE_SHAPE, ComponentType.ENTRANCE, atom),
            _cell_shape(EXIT_SHAPE, ComponentType.EXIT, atom),
        ),
        entrypoints=(mkentr(ENTRANCE_SHAPE, entrance), mkexit(EXIT_SHAPE, exit)),
        allowed_blueprints=frozenset({"arm"}),
        steps=steps,
        entrypoints_fixed=True,
        target=target,
        cycling=False,
    )
    if arm_plan(spec) is None:
        raise FamilyError(f"entrance {tuple(entrance)} and exit {tuple(exit)} have no verified delivery cycle")
    return spec


def arm_plan(spec: LevelSpec) -> ArmPlan | None:
    """
    Deterministically chosen simulator-verified arm placement, or None.

    Candidates are ordered by (base row, base col, clockwise before counter-clockwise,
    shortest cycle); the first one that actually solves the level wins.
    """
    entrance = _entrypoint_pos(spec, ComponentType.ENTRANCE)
    exit_ = _entrypoint_pos(spec, ComponentType.EXIT)
    for plan in _candidates(spec, entrance, exit_):
        if _solves(spec, plan):
            return plan
    return None


def reference_solution(spec: LevelSpec) -> Solution:
    """
    Canonical solution IR: one arm placement and the full delivery program.
    """
    plan = arm_plan(spec)
    if plan is None:
        raise FamilyError("spec has no single-arm reference solution")
    vocab = Vocab()
    arm = vocab.builtin(vocab.builtin_id("arm"))
    placement = Placement(
        builtin=arm,
        slot=None,
        template=arm.template,
        rotation=plan.rotation,
        q=plan.base.s,
        r=plan.base.r,
        pos=0,
    )
    actions = [Action(handle=0, tick=tick, action=action, pos=0) for tick, action in enumerate(plan.program, 1)]
    return Solution(placements=[placement], actions=actions, submitted=True, length=0)


def reference_tokens(spec: LevelSpec) -> list[int]:
    """
    Canonical token sequence of the reference solution.
    """
    sol = reference_solution(spec)
    return OpusTokenizer().encode_solution(sol.placements, sol.actions, spec)


def reference_oms(spec: LevelSpec) -> str:
    """
    `.oms` text of the reference solution, with component ids from a real placement.
    """
    sol = reference_solution(spec)
    level = Level(spec)
    report = apply_solution(sol, spec, level)
    if not report.ok:
        raise FamilyError(f"reference solution cannot be applied: {report.error}")
    return solution_to_oms(sol, report.handles, spec)


def verify(spec: LevelSpec, sol: Solution | None = None) -> int:
    """
    Full builder check mirroring `solutions._validate`: token round-trip,
    mask-prefix validity, apply + run solves. Returns the solved step count.
    """
    tokenizer = OpusTokenizer()
    vocab = tokenizer.vocab
    sol = sol or reference_solution(spec)
    tokens = tokenizer.encode_solution(sol.placements, sol.actions, spec)

    decoded = decode_solution(tokens, spec, vocab)
    if isinstance(decoded, GrammarError):
        raise FamilyError(f"reference decode failed at token {decoded.pos}: {decoded.message}")
    if _fingerprint(decoded) != _fingerprint(sol):
        raise FamilyError("reference sequence does not round-trip the solution")

    masker = ActionMasker(vocab)
    for position, token in enumerate(tokens):
        if not masker.mask(tokens[:position], spec)[token]:
            raise FamilyError(f"reference token {position} ({vocab.name(token)}) is not mask-valid at its prefix")

    level = Level(spec)
    report = apply_solution(decoded, spec, level)
    if not report.ok:
        raise FamilyError(f"reference apply failed: {report.error}")
    result = level.run()
    if not result.solved:
        status = result.status.value
        reason = result.error.reason if result.error is not None else None
        raise FamilyError(f"reference solution does not solve the level (status={status}, reason={reason})")
    # the level is dirty after a run; every builder output must leave it reset
    level.reset()
    return result.steps


def _candidates(spec: LevelSpec, entrance: Hex, exit_: Hex) -> list[ArmPlan]:
    """
    Every arm base that sees both cells on its ring, with both swing directions
    and the shortest matching cycle length, in deterministic order.
    """
    rows, cols = spec.size
    deliveries = spec.target or 1
    bases = [
        base
        for base in set(_ring_neighbors(entrance)) & set(_ring_neighbors(exit_))
        if base not in (entrance, exit_) and 0 <= base.r < rows and 0 <= base.s < cols
    ]
    ranked = []
    for base in bases:
        start = _direction(base, entrance)
        delta = _direction(base, exit_) - start
        # clockwise before counter-clockwise, then the shortest swing
        for order, (turn, k) in enumerate(((_CLW, delta % 6), (_CCLW, -delta % 6))):
            if not 1 <= k <= 5:
                continue
            plan = ArmPlan(base=base, rotation=start, program=_program(deliveries, turn, k))
            ranked.append(((base.r, base.s, order, k), plan))
    return [plan for _, plan in sorted(ranked, key=lambda item: item[0])]


def _program(deliveries: int, turn: StepActionType, k: int) -> tuple[StepActionType, ...]:
    """
    `deliveries` identical cycles: grab, `k` turns to the exit, release, `6 - k` turns back.

    The first step grabs while the entrance produces the atom on the same step,
    so the hand starts on the entrance cell.
    """
    cycle = (_GRAB, *[turn] * k, _RELEASE, *[turn] * (6 - k))
    return cycle * deliveries


def _solves(spec: LevelSpec, plan: ArmPlan) -> bool:
    """
    True when the plan's full program solves a freshly built level.
    """
    level = Level(spec)
    arm = level.place("arm", plan.base, rotation=plan.rotation)
    for tick, action in enumerate(plan.program, 1):
        level.assign_step_action(arm.comp_id, tick, action)
    return level.run().solved


def _cell_shape(name: str, kind: ComponentType, atom: AtomType) -> ShapeTemplate:
    return ShapeTemplate(name=name, comp_type=kind, cells=(ShapeCell(offset=Hex(0, 0), atom_type=atom),), links=())


def _entrypoint_pos(spec: LevelSpec, kind: ComponentType) -> Hex:
    for ep in spec.entrypoints:
        if ep.kind is kind and ep.pos is not None:
            return ep.pos
    raise FamilyError(f"spec has no fixed {kind.name.lower()} entrypoint")


def _ring_neighbors(cell: Hex) -> list[Hex]:
    return [Hex(cell.r + dr, cell.s + ds) for dr, ds in RING_OFFSETS]


def _direction(base: Hex, cell: Hex) -> int:
    for rotation, offset in enumerate(RING_OFFSETS):
        if (cell.r - base.r, cell.s - base.s) == offset:
            return rotation
    raise FamilyError(f"cell {tuple(cell)} is not adjacent to base {tuple(base)}")


def _fingerprint(sol: Solution) -> tuple[list[tuple], list[tuple]]:
    placements = [(p.slot, p.builtin.name if p.builtin else None, p.rotation, p.q, p.r) for p in sol.placements]
    actions = sorted(((a.handle, a.tick, a.action) for a in sol.actions), key=lambda item: item[:2])
    return placements, actions


# TODO: rework and cleanup that mess

# Same fixed 1-cell typed entrance/exit and 6x6 board as tier-1, but any distinct
# E != X positions and an `{arm, arm_piston}` allow-list. `arm_double`/`arm_triple`
# are declared by the simulator but unimplemented, and static-length tokens would
# be a wire-format break, so rings 2-3 are only ever expressed as `arm_piston`
# programs that lead with the matching number of EXTEND actions.
#
# Geometry: the hand orbits at radius = current hand length; EXTEND/RETRACT slide
# it one ring along the current rotation per tick. Rotation 0 is (-1, +1) from the
# base and CLW is +1, exactly as tier-1.
#
# Shapes:
# - single arm (ring 1-3): EXT^(L-1), GRAB, k turns out, RELEASE, (6-k) turns
#   home, RET^(L-1) — one 2L+8-action cycle per delivery, all turns in one
#   direction so the arm always returns to its rest state (radius 1, entrance
#   direction); ring 1 reduces to the tier-1 cycle byte for byte.
# - handoff (target 1): arm A grabs at (dir_E, ring_A_E) and releases at a shared
#   ring cell H, arm B grabs at H and releases at (dir_X, ring_B_X). The atom
#   persists on the board between release and grab, so B's grab is scheduled
#   exactly one tick after A's release; B's leg start is pulled earlier only to
#   make room for its pre-grab EXTEND actions.
#
# Candidates are enumerated in a deterministic rank order (fewer arms, shorter
# peak arm length, fewer actions, lexicographic positions), cap-filtered, and
# verified one by one on a fresh simulator `Level`; the first that solves wins.

ANYWHERE_BLUEPRINTS = frozenset({"arm", "arm_piston"})
MAX_RING = 3


@dataclass(frozen=True)
class ArmProgram:
    """
    One anywhere-family arm: where it stands and which action it takes each tick.
    """

    # fmt: off
    blueprint: str                                 # "arm" or "arm_piston"
    base: Hex
    rotation: int                                  # rest direction of the hand (radius 1)
    ticks: tuple[tuple[int, StepActionType], ...]  # (tick, action), unique ascending ticks
    # fmt: on


@dataclass(frozen=True)
class AnywherePlan:
    """
    One or two arms delivering `target` molecules entrance -> exit.
    """

    arms: tuple[ArmProgram, ...]
    target: int


def anywhere_spec(  # noqa: PLR0913
    entrance: Hex,
    exit: Hex,
    *,
    target: int | None = None,
    steps: int = 30,
    size: BoardSize = (6, 6),
    atom: AtomType = AtomType.WATER,
) -> LevelSpec:
    """
    Fixed-entrypoint spec for the anywhere family: `target` typed single atoms
    delivered entrance -> exit by one arm (rings 1-3) or a two-arm handoff.

    `target=None` picks the class rule from the winning plan: 2 for single-arm
    deliveries, 1 for handoffs. Raises `FamilyError` for every pair outside the
    family: out of board, degenerate, over a cap, or with no verified plan.
    """
    rows, cols = size
    for name, cell in (("entrance", entrance), ("exit", exit)):
        if not (0 <= cell.r < rows and 0 <= cell.s < cols):
            raise FamilyError(f"{name} cell {tuple(cell)} is out of the {rows}x{cols} board")
    if entrance == exit:
        raise FamilyError("entrance and exit must not share a cell")
    if steps > VocabConfig().max_steps:
        raise FamilyError(f"steps {steps} is over the {VocabConfig().max_steps} tick-token cap")
    plan = _search_plan(entrance, exit, target, steps, size, atom)
    if plan is None:
        raise FamilyError(f"entrance {tuple(entrance)} and exit {tuple(exit)} have no verified delivery plan")
    spec = _anywhere_level_spec(entrance, exit, plan.target, steps, size, atom)
    _assert_caps(plan, steps)
    return spec


def anywhere_plan(spec: LevelSpec) -> AnywherePlan | None:
    """
    Deterministically chosen simulator-verified anywhere plan for a spec, or None.

    Candidates are ranked (fewer arms, shorter peak arm length, fewer actions,
    lexicographic positions) and verified in order; `spec.target` selects the
    class (handoffs exist only at target 1, single arms at any target).
    """
    entrance = _entrypoint_pos(spec, ComponentType.ENTRANCE)
    exit_ = _entrypoint_pos(spec, ComponentType.EXIT)
    for _, plan in _plan_candidates(entrance, exit_, spec.target, spec.steps, spec.size):
        if _plan_solves(spec, plan):
            return plan
    return None


def anywhere_reference_solution(spec: LevelSpec) -> Solution:
    """
    Canonical solution IR: the winning plan's placements and full program.
    """
    plan = anywhere_plan(spec)
    if plan is None:
        raise FamilyError("spec has no anywhere reference solution")
    vocab = Vocab()
    placements = []
    actions = []
    for handle, arm in enumerate(plan.arms):
        builtin = vocab.builtin(vocab.builtin_id(arm.blueprint))
        placements.append(
            Placement(
                builtin=builtin,
                slot=None,
                template=builtin.template,
                rotation=arm.rotation,
                q=arm.base.s,
                r=arm.base.r,
                pos=0,
            )
        )
        actions.extend(Action(handle=handle, tick=tick, action=action, pos=0) for tick, action in arm.ticks)
    return Solution(placements=placements, actions=actions, submitted=True, length=0)


def anywhere_reference_tokens(spec: LevelSpec) -> list[int]:
    """
    Canonical token sequence of the anywhere reference solution.
    """
    sol = anywhere_reference_solution(spec)
    return OpusTokenizer().encode_solution(sol.placements, sol.actions, spec)


def anywhere_reference_oms(spec: LevelSpec) -> str:
    """
    `.oms` text of the anywhere reference solution, with component ids from a real placement.
    """
    sol = anywhere_reference_solution(spec)
    level = Level(spec)
    report = apply_solution(sol, spec, level)
    if not report.ok:
        raise FamilyError(f"reference solution cannot be applied: {report.error}")
    return solution_to_oms(sol, report.handles, spec)


def _search_plan(  # noqa: PLR0913, PLR0917
    entrance: Hex,
    exit: Hex,
    target: int | None,
    steps: int,
    size: BoardSize,
    atom: AtomType,
) -> AnywherePlan | None:
    """
    First ranked candidate that solves a freshly built level, or None.

    Verification needs a spec, built lazily per distinct target value.
    """
    specs: dict[int, LevelSpec] = {}
    for _, plan in _plan_candidates(entrance, exit, target, steps, size):
        spec = specs.get(plan.target)
        if spec is None:
            spec = specs[plan.target] = _anywhere_level_spec(entrance, exit, plan.target, steps, size, atom)
        if _plan_solves(spec, plan):
            return plan
    return None


def _anywhere_level_spec(  # noqa: PLR0913, PLR0917
    entrance: Hex,
    exit: Hex,
    target: int,
    steps: int,
    size: BoardSize,
    atom: AtomType,
) -> LevelSpec:
    return LevelSpec(
        size=size,
        shapes=(
            _cell_shape(ENTRANCE_SHAPE, ComponentType.ENTRANCE, atom),
            _cell_shape(EXIT_SHAPE, ComponentType.EXIT, atom),
        ),
        entrypoints=(mkentr(ENTRANCE_SHAPE, entrance), mkexit(EXIT_SHAPE, exit)),
        allowed_blueprints=ANYWHERE_BLUEPRINTS,
        steps=steps,
        entrypoints_fixed=True,
        target=target,
        cycling=False,
    )


def _plan_candidates(
    entrance: Hex,
    exit_: Hex,
    target: int | None,
    steps: int,
    size: BoardSize,
) -> list[tuple[tuple, AnywherePlan]]:
    """
    Every cap-checked unverified plan for the pair, in deterministic rank order.

    Handoffs deliver exactly one molecule, so they are enumerated only when the
    target allows it (`None` or 1).
    """
    candidates = _single_candidates(entrance, exit_, target, steps, size)
    if target is None or target == 1:
        candidates.extend(_handoff_candidates(entrance, exit_, steps, size))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _single_candidates(
    entrance: Hex,
    exit_: Hex,
    target: int | None,
    steps: int,
    size: BoardSize,
) -> list[tuple[tuple, AnywherePlan]]:
    """
    One arm whose ring of radius 1..MAX_RING sees both cells, both turn directions.
    """
    deliveries = target if target is not None else 2
    out = []
    for ring in range(1, MAX_RING + 1):
        for e_dir in range(6):
            base = _shifted(entrance, e_dir, -ring)
            if not _in_board(base, size) or base == exit_:
                continue
            x_rel = _ring_direction(base, exit_)
            if x_rel is None or x_rel[1] != ring:
                continue
            x_dir, _ = x_rel
            for clw in (True, False):
                k = (x_dir - e_dir) % 6 if clw else (e_dir - x_dir) % 6
                if not 1 <= k <= 5:
                    continue
                cycle = _single_cycle(ring, clw, k)
                arm = ArmProgram(
                    "arm" if ring == 1 else "arm_piston",
                    base,
                    e_dir,
                    _cycle_ticks(cycle, deliveries),
                )
                plan = AnywherePlan((arm,), deliveries)
                if not _within_caps(plan, steps):
                    continue
                key = (1, ring, len(cycle) * deliveries, base.r, base.s, 0 if clw else 1, k)
                out.append((key, plan))
    return out


def _handoff_candidates(
    entrance: Hex,
    exit_: Hex,
    steps: int,
    size: BoardSize,
) -> list[tuple[tuple, AnywherePlan]]:
    """
    Two arms sharing a handoff cell H: A grabs E and releases at H, B grabs H
    and releases at X, every leg radius in 1..MAX_RING, both legs' slides and
    swings free to pick either turn direction (the shorter swing wins the rank).
    """
    # bro.
    out = []
    for r_a_e in range(1, MAX_RING + 1):
        for a_dir in range(6):
            a_base = _shifted(entrance, a_dir, -r_a_e)
            if not _in_board(a_base, size) or a_base in (entrance, exit_):
                continue
            for r_a_h in range(1, MAX_RING + 1):
                for h_dir in range(6):
                    hand = _shifted(a_base, h_dir, r_a_h)
                    if not _in_board(hand, size) or hand in (entrance, exit_, a_base):
                        continue
                    for r_b_h in range(1, MAX_RING + 1):
                        for b_dir in range(6):
                            b_base = _shifted(hand, b_dir, -r_b_h)
                            if not _in_board(b_base, size) or b_base in (entrance, exit_, a_base, hand):
                                continue
                            x_rel = _ring_direction(b_base, exit_)
                            if x_rel is None:
                                continue
                            x_dir, r_b_x = x_rel
                            a_ticks = _leg_ticks(1, a_dir, r_a_e, h_dir, r_a_h)
                            # B grabs exactly one tick after A releases; its leg starts
                            # earlier only to fit the pre-grab EXTEND actions.
                            b_start = max(1, a_ticks[-1][0] + 2 - r_b_h)
                            b_ticks = _leg_ticks(b_start, b_dir, r_b_h, x_dir, r_b_x)
                            plan = AnywherePlan(
                                (
                                    ArmProgram(_blueprint((r_a_e, r_a_h)), a_base, a_dir, a_ticks),
                                    ArmProgram(_blueprint((r_b_h, r_b_x)), b_base, b_dir, b_ticks),
                                ),
                                1,
                            )
                            if not _within_caps(plan, steps):
                                continue
                            peak = max(r_a_e, r_a_h) + max(r_b_h, r_b_x)
                            triplets = sum(len(arm.ticks) for arm in plan.arms)
                            key = (
                                2, peak, triplets,
                                a_base.r, a_base.s, b_base.r, b_base.s,
                                r_a_e, r_a_h, r_b_x, hand.r, hand.s,
                            )  # fmt: skip
                            out.append((key, plan))
    return out


def _single_cycle(ring: int, clw: bool, k: int) -> tuple[StepActionType, ...]:
    """
    One delivery cycle: EXTEND out, grab, `k` turns to the exit, release, turns
    home, RETRACT. `k` is the turn count in the variant's own direction (CLW or
    CCLW); all turns run in that one direction, so the cycle always spans
    `2 * ring + 8` actions and ends back at the rest state (radius 1, entrance
    direction) — ring 1 is exactly the tier-1 cycle.
    """
    turn = _CLW if clw else _CCLW
    return (
        *(_EXT,) * (ring - 1),
        _GRAB,
        *(turn,) * k,
        _RELEASE,
        *(turn,) * (6 - k),
        *(_RET,) * (ring - 1),
    )


def _cycle_ticks(cycle: tuple[StepActionType, ...], deliveries: int) -> tuple[tuple[int, StepActionType], ...]:
    """
    `deliveries` back-to-back copies of `cycle`, ticks starting at 1.
    """
    return tuple(
        (delivery * len(cycle) + offset + 1, action)
        for delivery in range(deliveries)
        for offset, action in enumerate(cycle)
    )


def _leg_ticks(
    start: int,
    grab_dir: int,
    grab_ring: int,
    give_dir: int,
    give_ring: int,
) -> tuple[tuple[int, StepActionType], ...]:
    """
    (tick, action) pairs for one handoff leg: reach (grab_dir, grab_ring), grab,
    move the atom to (give_dir, give_ring) by radial slides then the shorter
    swing, release. Slides come first so the swing stays on a single ring.
    """
    ticks: list[tuple[int, StepActionType]] = []
    tick = start
    for _ in range(grab_ring - 1):
        ticks.append((tick, _EXT))
        tick += 1
    ticks.append((tick, _GRAB))
    tick += 1
    slides = (_RET,) * (grab_ring - give_ring) if grab_ring > give_ring else (_EXT,) * (give_ring - grab_ring)
    swing = (give_dir - grab_dir) % 6
    slides += (_CLW,) * swing if swing <= 3 else (_CCLW,) * (6 - swing)
    for action in slides:
        ticks.append((tick, action))
        tick += 1
    ticks.append((tick, _RELEASE))
    return tuple(ticks)


def _blueprint(rings: tuple[int, ...]) -> str:
    """
    A plain arm serves an all-radius-1 leg pair; anything longer needs a piston.
    """
    return "arm" if all(ring == 1 for ring in rings) else "arm_piston"


def _plan_solves(spec: LevelSpec, plan: AnywherePlan) -> bool:
    """
    True when the plan's full program solves a freshly built level.

    Placement errors (e.g. a blueprint the spec does not allow) count as unsolved.
    """
    level = Level(spec)
    try:
        for arm in plan.arms:
            record = level.place(arm.blueprint, arm.base, rotation=arm.rotation)
            for tick, action in arm.ticks:
                level.assign_step_action(record.comp_id, tick, action)
    except Exception:  # noqa: BLE001 - any placement failure just rejects the candidate
        return False
    return level.run().solved


def _within_caps(plan: AnywherePlan, steps: int) -> bool:
    try:
        _assert_caps(plan, steps)
    except FamilyError:
        return False
    return True


def _assert_caps(plan: AnywherePlan, steps: int) -> None:
    """
    Wire caps: program triplets <= 32, placements <= 16, ticks < steps.
    """
    config = TokenizerConfig()
    triplets = sum(len(arm.ticks) for arm in plan.arms)
    if triplets > config.max_program_actions:
        raise FamilyError(
            f"plan programs {triplets} actions, over the {config.max_program_actions} program-action cap"
        )
    if len(plan.arms) > config.max_placements:
        raise FamilyError(f"plan places {len(plan.arms)} arms, over the {config.max_placements} placement cap")
    last = max(tick for arm in plan.arms for tick, _ in arm.ticks)
    if last >= steps:
        raise FamilyError(f"plan uses tick {last}, outside the {steps}-step budget")


def _shifted(cell: Hex, rotation: int, radius: int) -> Hex:
    """
    `cell` moved `radius` rings along `rotation` (negative radius walks backwards).
    """
    dr, ds = RING_OFFSETS[rotation]
    return Hex(cell.r + dr * radius, cell.s + ds * radius)


def _in_board(cell: Hex, size: BoardSize) -> bool:
    rows, cols = size
    return 0 <= cell.r < rows and 0 <= cell.s < cols


def _ring_direction(base: Hex, cell: Hex) -> tuple[int, int] | None:
    """
    (rotation, radius) with radius 1..MAX_RING when `cell` lies straight out
    from `base`, else None.
    """
    for radius in range(1, MAX_RING + 1):
        for rotation in range(6):
            if _shifted(base, rotation, radius) == cell:
                return rotation, radius
    return None
