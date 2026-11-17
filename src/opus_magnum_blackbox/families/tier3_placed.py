"""
Tier-3 placed family: solution-placed (unfixed) entrypoints and single-arm
reference solutions over tiny enumerable config space.

Spec:

- 6x6 board
- exactly one 1-cell typed entrance and one 1-cell typed exit, both carried by
  the *solution* (`entrypoints_fixed=False`, entrypoints have no position)
- allow-list per config: `arm` only, or `arm` + `arm_l2` + `arm_l3`
- `cycling 0`, steps 30
- target 1 or 2

Config space:
exercises atom conditioning, target exercises program-length conditioning,
and allow-list exercises builtin conditioning - no previous tier varied the allow-list.

Canonical solver convention (deterministic across processes):

- each atom type owns one fixed anchor cell in rows 3-5 / cols 0-2, where
  direction 0 (offset (-1, +1)) reaches separation >= 3 on board;
- candidates enumerate (anchor, exit direction, separation) in the rank order
  (anchor, direction 0..5, separation 5..1); the exit is the anchor shifted
  separation cells along the direction, so a candidate's hex distance equals
  its separation (straight lines only);
- each candidate delegates arm planning to `anywhere_spec_plan` (ranked,
  simulator-verified) and accepts single-arm winners only - a handoff winner
  means the pair is outside the family;
- the winner is re-verified against the unfixed spec with the config's
  allow-list, entrypoints placed before the arm exactly as the solution
  layout does; the first candidate that passes wins.

Descending separation is what keeps the canonical layouts from degenerating:
distance-3/4/5 candidates have ring-2/3 winners only, so the `arm`-only
allow-list rejects them and falls through to the distance-2 straight line,
allow-list axis changes the emitted layout for every config, atom axis changes
it through anchors, and target changes the program length.
"""

from dataclasses import dataclass

from opus_magnum_sym import AtomType, ComponentType, Hex, Level, LevelSpec, mkentr, mkexit

from opus_magnum_blackbox.codec import OpusTokenizer, apply_solution, solution_to_oms
from opus_magnum_blackbox.config import VocabConfig
from opus_magnum_blackbox.families._common import (
    ENTRANCE_SHAPE,
    EXIT_SHAPE,
    BoardSize,
    FamilyError,
    _cell_shape,
    hex_distance,
)
from opus_magnum_blackbox.families.tier2_anywhere import ArmProgram, _in_board, _shifted, anywhere_spec_plan
from opus_magnum_blackbox.grammar import Action, Placement, Solution, spec_model
from opus_magnum_blackbox.vocab import Vocab

BOARD: BoardSize = (6, 6)
STEPS = 30
MAX_SEPARATION = 5
SHORT_BLUEPRINTS = frozenset({"arm"})
LONG_BLUEPRINTS = frozenset({"arm", "arm_l2", "arm_l3"})

# rows 3-5 / cols 0-2: direction 0 walks (-1, +1), so every anchor here keeps a
# separation-3 candidate on board — the floor of the allow-list fall-through.
ANCHOR_CELLS: dict[AtomType, Hex] = {
    AtomType.WATER: Hex(4, 1),
    AtomType.FIRE: Hex(3, 0),
    AtomType.EARTH: Hex(5, 2),
    AtomType.AIR: Hex(5, 0),
}


@dataclass(frozen=True)
class PlacedConfig:
    """
    One tier-3 config: atom, deliveries per exit, and the layout allow-list.
    """

    atom: AtomType
    target: int
    long_arms: bool

    @property
    def blueprints(self) -> frozenset[str]:
        return LONG_BLUEPRINTS if self.long_arms else SHORT_BLUEPRINTS


@dataclass(frozen=True)
class PlacedPlan:
    """
    The canonical entrance/exit cells and the single arm delivering between them.
    """

    entrance: Hex
    exit: Hex
    arm: ArmProgram
    target: int


def placed_configs() -> tuple[PlacedConfig, ...]:
    """
    All 16 family configs in deterministic (atom, target, allow-list) order.
    """
    return tuple(
        PlacedConfig(atom=atom, target=target, long_arms=long_arms)
        for atom in (AtomType.WATER, AtomType.FIRE, AtomType.EARTH, AtomType.AIR)
        for target in (1, 2)
        for long_arms in (False, True)
    )


def placed_spec(config: PlacedConfig, *, steps: int = STEPS, size: BoardSize = BOARD) -> LevelSpec:
    """
    Unfixed-entrypoint spec for the config: `target` typed single atoms delivered
    entrance -> exit by one static arm.

    Raises `FamilyError` for every config outside the family or with no
    verified delivery plan.
    """
    return placed_spec_plan(config, steps=steps, size=size)[0]


def placed_spec_plan(
    config: PlacedConfig, *, steps: int = STEPS, size: BoardSize = BOARD
) -> tuple[LevelSpec, PlacedPlan]:
    """
    `placed_spec` together with the winning plan that shaped it.

    Raises `FamilyError` under the same conditions as `placed_spec`.
    """
    _check_config(config, steps, size)
    plan = _solve_config(config, steps, size)
    if plan is None:
        raise FamilyError(
            f"config (atom={config.atom.name}, target={config.target}, "
            f"long_arms={config.long_arms}) has no verified delivery plan"
        )
    return _placed_level_spec(config, steps, size), plan


def placed_plan(spec: LevelSpec) -> PlacedPlan | None:
    """
    Deterministically chosen simulator-verified placed plan for a spec, or None.

    Candidates are ranked per the module convention and verified in order;
    a spec outside the family (fixed entrypoints, foreign shapes, allow-list,
    atom or target) raises `FamilyError`.
    """
    config = _config_from_spec(spec)
    _check_config(config, spec.steps, spec.size)
    return _solve_config(config, spec.steps, spec.size)


def placed_reference_solution(spec: LevelSpec) -> Solution:
    """
    Canonical solution IR: entrance, exit, then the arm — placements in layout
    order — followed by the arm's full delivery program.
    """
    plan = placed_plan(spec)
    if plan is None:
        raise FamilyError("spec has no placed-entrypoint reference solution")
    vocab = Vocab()
    model = spec_model(spec)
    builtin = vocab.builtin(vocab.builtin_id(plan.arm.blueprint))
    placements = [
        Placement(
            builtin=None,
            slot=slot,
            template=model.templates[slot],
            rotation=0,
            q=cell.s,
            r=cell.r,
            pos=0,
        )
        for slot, cell in enumerate((plan.entrance, plan.exit))
    ]
    placements.append(
        Placement(
            builtin=builtin,
            slot=None,
            template=builtin.template,
            rotation=plan.arm.rotation,
            q=plan.arm.base.s,
            r=plan.arm.base.r,
            pos=0,
        )
    )
    actions = [Action(handle=0, tick=tick, action=action, pos=0) for tick, action in plan.arm.ticks]
    return Solution(placements=placements, actions=actions, submitted=True, length=0)


def placed_reference_tokens(spec: LevelSpec) -> list[int]:
    """
    Canonical token sequence of the placed-family reference solution.
    """
    sol = placed_reference_solution(spec)
    return OpusTokenizer().encode_solution(sol.placements, sol.actions, spec)


def placed_reference_oms(spec: LevelSpec) -> str:
    """
    `.oms` text of the placed-family reference solution, with component ids from
    a real placement (entrypoints included — the solution places them).
    """
    sol = placed_reference_solution(spec)
    level = Level(spec)
    report = apply_solution(sol, spec, level)
    if not report.ok:
        raise FamilyError(f"reference solution cannot be applied: {report.error}")
    return solution_to_oms(sol, report.handles, spec)


def _check_config(config: PlacedConfig, steps: int, size: BoardSize) -> None:
    if config.atom not in ANCHOR_CELLS:
        raise FamilyError(f"atom {config.atom.name} is not one of the four elementals")
    if config.target not in (1, 2):
        raise FamilyError(f"target {config.target} is outside the family (1 or 2)")
    if steps > VocabConfig().max_steps:
        raise FamilyError(f"steps {steps} is over the {VocabConfig().max_steps} tick-token cap")
    if size != BOARD:
        raise FamilyError(f"the placed family is defined on the {BOARD} board only, got {size}")


def _solve_config(config: PlacedConfig, steps: int, size: BoardSize) -> PlacedPlan | None:
    """
    First ranked candidate whose anywhere plan is a single arm verified against
    the unfixed spec, or None.
    """
    spec = _placed_level_spec(config, steps, size)
    for entrance, exit_ in _candidate_pairs(config.atom, size):
        try:
            _, plan = anywhere_spec_plan(entrance, exit_, target=config.target, steps=steps, size=size, atom=config.atom)
        except FamilyError:
            continue  # no anywhere plan for this pair
        if len(plan.arms) != 1:
            continue  # handoff winner: outside the single-arm family
        if not _solves(spec, plan.arms[0], entrance, exit_):
            continue  # winner needs a blueprint outside the config allow-list
        return PlacedPlan(entrance=entrance, exit=exit_, arm=plan.arms[0], target=config.target)
    return None


def _candidate_pairs(atom: AtomType, size: BoardSize) -> list[tuple[Hex, Hex]]:
    """
    (entrance, exit) candidates for the atom's anchor in rank order:
    direction 0..5, separation 5..1, on board, straight lines only.

    Distance-5 lines are dropped outright: no ring 1-3 orbit sees two cells
    five apart, so the delegation could only ever return a handoff.
    """
    anchor = ANCHOR_CELLS[atom]
    out = []
    for direction in range(6):
        for separation in range(MAX_SEPARATION, 0, -1):
            exit_ = _shifted(anchor, direction, separation)
            if not _in_board(exit_, size) or hex_distance(anchor, exit_) == 5:
                continue
            out.append((anchor, exit_))
    return out


def _placed_level_spec(config: PlacedConfig, steps: int, size: BoardSize) -> LevelSpec:
    return LevelSpec(
        size=size,
        shapes=(
            _cell_shape(ENTRANCE_SHAPE, ComponentType.ENTRANCE, config.atom),
            _cell_shape(EXIT_SHAPE, ComponentType.EXIT, config.atom),
        ),
        entrypoints=(mkentr(ENTRANCE_SHAPE), mkexit(EXIT_SHAPE)),
        allowed_blueprints=config.blueprints,
        steps=steps,
        entrypoints_fixed=False,
        target=config.target,
        cycling=False,
    )


def _config_from_spec(spec: LevelSpec) -> PlacedConfig:
    """
    Re-derive the family config from a spec, for the reference functions.
    """
    if spec.entrypoints_fixed:
        raise FamilyError("the placed family serves unfixed-entrypoint specs only")
    shapes = {shape.name: shape for shape in spec.shapes}
    entrance = shapes.get(ENTRANCE_SHAPE)
    exit_ = shapes.get(EXIT_SHAPE)
    if entrance is None or exit_ is None or len(entrance.cells) != 1 or len(exit_.cells) != 1:
        raise FamilyError("the placed family serves single-cell entrance/exit shapes only")
    atom = entrance.cells[0].atom_type
    if atom is None or atom != exit_.cells[0].atom_type:
        raise FamilyError("entrance and exit must carry the same typed atom")
    if spec.allowed_blueprints == SHORT_BLUEPRINTS:
        long_arms = False
    elif spec.allowed_blueprints == LONG_BLUEPRINTS:
        long_arms = True
    else:
        raise FamilyError(f"allow-list {sorted(spec.allowed_blueprints)} is outside the family")
    return PlacedConfig(atom=atom, target=spec.target or 0, long_arms=long_arms)


def _solves(spec: LevelSpec, arm: ArmProgram, entrance: Hex, exit_: Hex) -> bool:
    """
    True when placing the entrypoints (solution order: entrance, exit, arm) and
    running the arm's full program solves a freshly built level.
    """
    level = Level(spec)
    try:
        level.place_entrypoint(_entrypoint_index(spec, ComponentType.ENTRANCE), entrance, rotation=0)
        level.place_entrypoint(_entrypoint_index(spec, ComponentType.EXIT), exit_, rotation=0)
        record = level.place(arm.blueprint, arm.base, rotation=arm.rotation)
        for tick, action in arm.ticks:
            level.assign_step_action(record.comp_id, tick, action)
    except Exception:  # noqa: BLE001 - any placement failure just rejects the candidate
        return False
    return level.run().solved


def _entrypoint_index(spec: LevelSpec, kind: ComponentType) -> int:
    for index, entrypoint in enumerate(spec.entrypoints):
        if entrypoint.kind is kind:
            return index
    raise FamilyError(f"spec has no {kind.name.lower()} entrypoint")
