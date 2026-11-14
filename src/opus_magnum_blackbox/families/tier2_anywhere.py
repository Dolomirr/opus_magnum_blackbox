"""
Tier-2 anywhere family: fixed-entrypoint specs and reference solutions
for any distinct entrance/exit pair.

Spec:

- 6x6 board
- fixed 1-cell typed entrance and exit
- `arm` and `arm_piston` allowed; rings 2-3 are `arm_piston` programs leading
  with EXTEND actions (`arm_double`/`arm_triple` are declared by the simulator
  but unimplemented, and static-length tokens would be a wire-format break)
- `cycling 0`
- target 2 for single-arm plans, 1 for handoffs

Geometry (measured against the simulator):

- the hand orbits at radius = current hand length; EXTEND/RETRACT slide it one
  ring along the current rotation per tick;
- rotation 0 is `(-1, +1)` from the base and `ROTATE_CLW` is `r + 1`,
  exactly as tier-1.

Cycles:

- single arm (ring 1-3): `EXTEND x (L-1)`, `GRAB`, `k` turns to exit,
  `RELEASE`, `6 - k` turns home, `RETRACT x (L-1)` — one `2L + 8`-action cycle
  per delivery, all turns in one direction, so the arm always returns to its
  rest state (radius 1, entrance direction); ring 1 reduces to the tier-1
  cycle byte for byte.
- handoff: arm A grabs at E and releases at a shared ring cell H, arm B grabs
  at H and releases at X. The atom persists on the board between release and
  grab, so B's grab is scheduled exactly one tick after A's release; B's leg
  start is pulled earlier only to fit the pre-grab EXTEND actions.

Candidates are enumerated in a deterministic rank order (fewer arms, shorter
peak arm length, fewer actions, lexicographic positions), cap-filtered, and
verified one by one on a fresh simulator `Level`; the first that solves wins.
"""

from dataclasses import dataclass

from opus_magnum_sym import AtomType, ComponentType, Hex, Level, LevelSpec, StepActionType, mkentr, mkexit

from opus_magnum_blackbox.codec import OpusTokenizer, apply_solution, solution_to_oms
from opus_magnum_blackbox.config import TokenizerConfig, VocabConfig
from opus_magnum_blackbox.families._common import (
    _CCLW,
    _CLW,
    _EXT,
    _GRAB,
    _RELEASE,
    _RET,
    ENTRANCE_SHAPE,
    EXIT_SHAPE,
    RING_OFFSETS,
    BoardSize,
    FamilyError,
    _cell_shape,
    _entrypoint_pos,
)
from opus_magnum_blackbox.grammar import Action, Placement, Solution
from opus_magnum_blackbox.vocab import Vocab

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
