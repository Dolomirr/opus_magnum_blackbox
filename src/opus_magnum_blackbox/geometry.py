"""
Placement geometry for the masker: footprint, bounds and overlap.

Mirrors the simulator's placement rules without mutating `Level`.
"""

from functools import cache

from opus_magnum_sym import Hex, LevelSpec
from opus_magnum_sym.elements.components.blueprints import ShapeTemplate

from opus_magnum_blackbox.grammar import SpecModel
from opus_magnum_blackbox.vocab import ROTATION_COUNT

# TODO: i dont like it, maybe rewrite this


@cache
def shape_cells(template: ShapeTemplate, rotation: int) -> frozenset[tuple[int, int]]:
    """
    Rotated cell offsets (r, s) relative to placement origin.
    """
    origin = Hex(0, 0)
    # fmt: off
    return frozenset(
        (cell.offset.rotate_around(origin, rotation).r,
         cell.offset.rotate_around(origin, rotation).s,)
        for cell in template.cells
    )
    # fmt: on


def rotations(template: ShapeTemplate | None) -> range | frozenset[int]:
    if template is not None and template.allowed_rotations is not None:
        return template.allowed_rotations
    return range(ROTATION_COUNT)


def footprint(template: ShapeTemplate | None, rotation: int, r: int, s: int) -> frozenset[tuple[int, int]]:
    """
    Board cells placement occupies; manipulators and single-cell specs occupy their origin.
    """
    if template is None:
        return frozenset({(r, s)})
    return frozenset((r + dr, s + ds) for dr, ds in shape_cells(template, rotation))


def occupied_cells(spec: LevelSpec, model: SpecModel, placements) -> frozenset[tuple[int, int]]:
    """
    Cells already taken by fixed entrypoints and earlier layout placements.
    """
    cells: set[tuple[int, int]] = set()
    if spec.entrypoints_fixed:
        shapes = {shape.name: shape for shape in spec.shapes}
        for ep in spec.entrypoints:
            cells |= footprint(shapes[ep.shape], ep.rotation, ep.pos.r, ep.pos.s)
    for placement in placements:
        cells |= footprint(placement.template, placement.rotation, placement.r, placement.q)
    return frozenset(cells)


def placement_valid(  # noqa: PLR0913, PLR0917
    template: ShapeTemplate | None,
    rotation: int,
    r: int,
    s: int,
    rows: int,
    cols: int,
    occupied: frozenset,
) -> bool:
    for cell_r, cell_s in footprint(template, rotation, r, s):
        if not (0 <= cell_r < rows and 0 <= cell_s < cols) or (cell_r, cell_s) in occupied:
            return False
    return True
