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


def single_atom_spec(  # noqa: PLR0913
    entrance: Hex,
    exit: Hex,
    *,
    target: int = 2,
    steps: int = 30,
    size: tuple[int, int] = (6, 6),
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
        raise FamilyError(f"target {target} programs {program_actions} actions, over the {max_actions} program-action cap")
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
