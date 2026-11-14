"""
Shared geometry and spec helpers for the family modules.
"""

from opus_magnum_sym import AtomType, ComponentType, Hex, LevelSpec, ShapeCell, ShapeTemplate, StepActionType

ENTRANCE_SHAPE = "entrance_1"
EXIT_SHAPE = "exit_1"

# rotation -> (dr, ds) hand offset around the arm base; rotation 0 is (-1, +1), CLW is +1
RING_OFFSETS = ((-1, 1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0))

_GRAB = StepActionType.GRAB
_RELEASE = StepActionType.RELEASE
_CLW = StepActionType.ROTATE_CLW
_CCLW = StepActionType.ROTATE_CCLW
_EXT = StepActionType.EXTEND
_RET = StepActionType.RETRACT


class FamilyError(ValueError):
    """Entrance/exit pair cannot be served by the requested family."""


def hex_distance(a: Hex, b: Hex) -> int:
    return (abs(a.r - b.r) + abs(a.s - b.s) + abs(a.r + a.s - b.r - b.s)) // 2


# rows, cols of the rectangular board a spec is cut from
type BoardSize = tuple[int, int]


def _cell_shape(name: str, kind: ComponentType, atom: AtomType) -> ShapeTemplate:
    return ShapeTemplate(name=name, comp_type=kind, cells=(ShapeCell(offset=Hex(0, 0), atom_type=atom),), links=())


def _entrypoint_pos(spec: LevelSpec, kind: ComponentType) -> Hex:
    for ep in spec.entrypoints:
        if ep.kind is kind and ep.pos is not None:
            return ep.pos
    raise FamilyError(f"spec has no fixed {kind.name.lower()} entrypoint")
