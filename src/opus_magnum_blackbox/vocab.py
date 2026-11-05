from dataclasses import dataclass

from opus_magnum_sym import AtomType, StepActionType
from opus_magnum_sym.elements.components import default_registry
from opus_magnum_sym.elements.components.blueprints import BPN, ManipulatorSpec, ShapeTemplate

from opus_magnum_blackbox.config import VocabConfig

SPECIAL_TOKENS = ("[PAD]", "[START_LAYOUT]", "[END_LAYOUT]", "[START_PROGRAM]", "[SUBMIT]")
MODE_TOKENS = ("MODE_FIXED", "MODE_UNFIXED", "UNPLACED")
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

ACTION_TO_STEP_TYPE = {
    "ACT_GRAB": StepActionType.GRAB,
    "ACT_RELEASE": StepActionType.RELEASE,
    "ACT_ROTATE_CLW": StepActionType.ROTATE_CLW,
    "ACT_ROTATE_CCLW": StepActionType.ROTATE_CCLW,
    "ACT_EXTEND": StepActionType.EXTEND,
    "ACT_RETRACT": StepActionType.RETRACT,
}

BASE_ACTION_NAMES = ("ACT_GRAB", "ACT_RELEASE", "ACT_ROTATE_CLW", "ACT_ROTATE_CCLW", "ACT_NOP")
PISTON_ACTION_NAMES = ("ACT_EXTEND", "ACT_RETRACT")

ATOM_TOKEN_TYPES = tuple(t for t in AtomType if t not in (AtomType.NONE, AtomType.BASIC))
ATOM_TYPES = (None, *ATOM_TOKEN_TYPES)


@dataclass(frozen=True)
class Builtin:
    name: str
    manipulator: bool  # consumes a program handle
    extendable: bool  # piston arm (accepts EXTEND/RETRACT)
    template: ShapeTemplate | None  # footprint geometry; None for arms/single-cell


def _resolve_builtins() -> tuple[Builtin, ...]:
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

_TOKEN_PREFIXES = {
    "entry_slot": "ESHAPE_",
    "rotation": "ROT_",
    "q": "Q_",
    "r": "R_",
    "delta": "D_",
    "cell": "CELL_",
    "handle": "H_",
    "tick": "TICK_",
    "builtin": "BP_",
}


class Vocab:
    def __init__(self, config: VocabConfig | None = None) -> None:
        self.config = config or VocabConfig()
        cfg = self.config

        builtin_names = tuple(b.name for b in BUILTINS)
        categories: dict[str, tuple[tuple[str, ...], int]] = {
            "special": (SPECIAL_TOKENS, len(SPECIAL_TOKENS)),
            "mode": (MODE_TOKENS, len(MODE_TOKENS)),
            "ctype": (CTYPE_TOKENS, len(CTYPE_TOKENS)),
            "atom": (tuple(f"ATOM_{t.name}" for t in ATOM_TOKEN_TYPES), len(ATOM_TYPES)),
            "builtin": (builtin_names, len(BUILTINS)),
            "entry_slot": ((), cfg.max_entrypoint_slots),
            "rotation": ((), ROTATION_COUNT),
            "q": ((), cfg.max_cols),
            "r": ((), cfg.max_rows),
            "delta": ((), 2 * cfg.max_cell_offset + 1),
            "cell": ((), cfg.max_shape_cells),
            "handle": ((), cfg.max_handles),
            "tick": ((), cfg.max_steps),
            "action": (ACTION_TOKEN_NAMES, len(ACTION_TOKEN_NAMES)),
        }

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
        return self.builtins[token_id - self.base("builtin")]

    def step_type(self, action_id: int) -> StepActionType | None:
        return self._step_types[action_id - self.base("action")]
