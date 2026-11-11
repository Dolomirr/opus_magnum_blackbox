"""
Flat integer vocabulary for Opus Magnum RL translation layer.

Token categories (each is a contiguous id range):

| type       | examples / description              | comment                       |
|------------|-------------------------------------|-------------------------------|
| specials   | [PAD] [START_LAYOUT] [END_LAYOUT]   | meta tokens                   |
|            | [START_PROGRAM] [SUBMIT]            |                               |
| modes      | MODE_UNFIXED MODE_FIXED UNPLACED    | level config                  |
| ctypes     | CTYPE_ENTRANCE CTYPE_EXIT           | encoder shape segments        |
| atoms      | ATOM_NONE + one per real AtomType   | encoder typed cells           |
| builtins   | BP_<name> implemented built-in      |                               |
| entrypoint | shape slots: ESHAPE_0..K-1          | resolved against current spec |
| rotations  | ROT_0..5                            |                               |
| coords     | Q_0..max_cols-1 , R_0..max_rows-1   | maps to Hex.s/r               |
| deltas     | D_<off> encoder cell offsets        | in [-max_offset, max_offset]  |
| cells      | CELL_0..max_shape_cells-1           | encoder link endpoints        |
| handles    | H_0..H-1                            | n-th placed arm               |
| ticks      | TICK_0..max_steps-1                 |                               |
| actions    | ACT_GRAB ACT_RELEASE ACT_ROTATE_CLW |                               |
|            | ACT_ROTATE_CCLW ACT_EXTEND          |                               |
|            | ACT_RETRACT ACT_NOP                 |                               |

Numeric categories are impelemnted as `base(category) + value`.
"""

from dataclasses import dataclass

from opus_magnum_sym import AtomType, StepActionType
from opus_magnum_sym.elements.components import default_registry
from opus_magnum_sym.elements.components.blueprints import BPN, ManipulatorSpec, ShapeTemplate

from opus_magnum_blackbox.config import VocabConfig

SPECIAL_TOKENS = ("[PAD]", "[START_LAYOUT]", "[END_LAYOUT]", "[START_PROGRAM]", "[SUBMIT]")
# mode value mirrors the `entrypoints_fixed` flag: 0 = unfixed, 1 = fixed
MODE_TOKENS = ("MODE_UNFIXED", "MODE_FIXED", "UNPLACED")
CTYPE_TOKENS = ("CTYPE_ENTRANCE", "CTYPE_EXIT")
ROTATION_COUNT = 6

ACTION_TOKEN_NAMES = (
    "ACT_GRAB",
    "ACT_RELEASE",
    "ACT_ROTATE_CLW",
    "ACT_ROTATE_CCLW",
    "ACT_EXTEND",
    "ACT_RETRACT",
    "ACT_NOP",
)

# ACT_NOP is decode-time only ("assign nothing at this tick"); it has no StepActionType.
# fmt: off
ACTION_TO_STEP_TYPE = {
    "ACT_GRAB"        : StepActionType.GRAB,
    "ACT_RELEASE"     : StepActionType.RELEASE,
    "ACT_ROTATE_CLW"  : StepActionType.ROTATE_CLW,
    "ACT_ROTATE_CCLW" : StepActionType.ROTATE_CCLW,
    "ACT_EXTEND"      : StepActionType.EXTEND,
    "ACT_RETRACT"     : StepActionType.RETRACT,
}
# fmt: on

# Actions every arm accepts, plus the piston-only pair.
BASE_ACTION_NAMES = ("ACT_GRAB", "ACT_RELEASE", "ACT_ROTATE_CLW", "ACT_ROTATE_CCLW", "ACT_NOP")
PISTON_ACTION_NAMES = ("ACT_EXTEND", "ACT_RETRACT")

# Real atom types only; NONE marks untyped encoder cells, BASIC is not a game type.
ATOM_TOKEN_TYPES = tuple(t for t in AtomType if t not in (AtomType.NONE, AtomType.BASIC))
ATOM_TYPES = (None, *ATOM_TOKEN_TYPES)


@dataclass(frozen=True)
class Builtin:
    """
    Resolved capability of one built-in blueprint.
    Read-only.
    """

    # fmt: off
    name        : str
    manipulator : bool                 # consumes a program handle
    extendable  : bool                 # piston arm (accepts EXTEND/RETRACT)
    template    : ShapeTemplate | None # footprint geometry; None for arms/single-cell
    # fmt: on


def _resolve_builtins() -> tuple[Builtin, ...]:
    """
    Snapshots implemented built-ins from registry, in `BPN` order.
    """
    builtins = []
    for name in BPN:
        blueprint = default_registry.get(name)
        if not blueprint.implemented:
            continue
        if isinstance(blueprint, ManipulatorSpec):
            builtins.append(Builtin(name, True, blueprint.extendable, None))
        elif isinstance(blueprint, ShapeTemplate):
            builtins.append(Builtin(name, False, False, blueprint))
        else:  # ComponentSpec (e.g. glyph_duplication)
            builtins.append(Builtin(name, False, False, None))
    return tuple(builtins)


BUILTINS = _resolve_builtins()

# Debug prefixes for numeric categories; `name()` reproduces the token spelling.
# fmt: off
_TOKEN_PREFIXES = {
    "entry_slot" : "ESHAPE_",
    "rotation"   : "ROT_",
    "q"          : "Q_",
    "r"          : "R_",
    "delta"      : "D_",
    "cell"       : "CELL_",
    "handle"     : "H_",
    "tick"       : "TICK_",
    "builtin"    : "BP_",
}
# fmt: on


class Vocab:
    """
    Named token maps plus per-category id ranges.
    """

    def __init__(self, config: VocabConfig | None = None) -> None:
        self.config = config or VocabConfig()
        cfg = self.config

        builtin_names = tuple(b.name for b in BUILTINS)
        # fmt: off
        categories: dict[str, tuple[tuple[str, ...], int]] = {
            "special"    : (SPECIAL_TOKENS, len(SPECIAL_TOKENS)),
            "mode"       : (MODE_TOKENS, len(MODE_TOKENS)),
            "ctype"      : (CTYPE_TOKENS, len(CTYPE_TOKENS)),
            "atom"       : (tuple(f"ATOM_{t.name}" for t in ATOM_TOKEN_TYPES), len(ATOM_TYPES)),
            "builtin"    : (builtin_names, len(BUILTINS)),
            "entry_slot" : ((), cfg.max_entrypoint_slots),
            "rotation"   : ((), ROTATION_COUNT),
            "q"          : ((), cfg.max_cols),
            "r"          : ((), cfg.max_rows),
            "delta"      : ((), 2 * cfg.max_cell_offset + 1),
            "cell"       : ((), cfg.max_shape_cells),
            "handle"     : ((), cfg.max_handles),
            "tick"       : ((), cfg.max_steps),
            "action"     : (ACTION_TOKEN_NAMES, len(ACTION_TOKEN_NAMES)),
        }
        # fmt: on

        self._names: list[str | None] = []
        self._name_to_id: dict[str, int] = {}
        self.ranges: dict[str, range] = {}
        for category, (names, size) in categories.items():
            start = len(self._names)
            self._names.extend(names)
            self._names.extend([None] * (size - len(names)))
            self._name_to_id.update((name, start + offset) for offset, name in enumerate(names))
            self.ranges[category] = range(start, start + size)

        self.builtins = BUILTINS
        self.base_actions = tuple(self.id(name) for name in BASE_ACTION_NAMES)
        self.piston_actions = tuple(self.id(name) for name in PISTON_ACTION_NAMES)
        self._step_types = tuple(ACTION_TO_STEP_TYPE.get(name) for name in ACTION_TOKEN_NAMES)

    def __len__(self) -> int:
        return len(self._names)

    def id(self, name: str) -> int:
        return self._name_to_id[name]

    def name(self, token_id: int) -> str:
        for category, ids in self.ranges.items():
            if token_id in ids:
                name = self._names[token_id]
                if name is not None:
                    return f"BP_{name}" if category == "builtin" else name
                return f"{_TOKEN_PREFIXES[category]}{token_id - ids.start}"
        raise IndexError(f"token id {token_id} out of range")

    @property
    def pad_id(self) -> int:
        return self._name_to_id["[PAD]"]

    def special_id(self, name: str) -> int:
        return self._name_to_id[name]

    def is_in(self, token_id: int, category: str) -> bool:
        return token_id in self.ranges[category]

    def base(self, category: str) -> int:
        return self.ranges[category].start

    def value_id(self, category: str, value: int) -> int:
        """
        Numeric category token for `value` (deltas are shifted by max_cell_offset).
        """
        if category == "delta":
            value += self.config.max_cell_offset
        if not 0 <= value < len(self.ranges[category]):
            raise ValueError(f"{category} value out of vocabulary range")
        return self.ranges[category].start + value

    def atom_id(self, atom_type: AtomType | None) -> int:
        return self.value_id("atom", ATOM_TYPES.index(atom_type))

    def ctype_id(self, is_exit: bool) -> int:
        return self.value_id("ctype", is_exit)

    def builtin_id(self, name: str) -> int:
        return self._name_to_id[name]

    def builtin(self, token_id: int) -> Builtin:
        """
        Resolved capability of built-in token.
        """
        return self.builtins[token_id - self.base("builtin")]

    def step_type(self, action_id: int) -> StepActionType | None:
        """
        `StepActionType` for action token, or None for NOP.
        """
        return self._step_types[action_id - self.base("action")]
