"""
Family modules: level builders and simulator-verified reference solutions.

- `tier1`: single-atom family (6x6 board, `arm` only, hex-distance 1-2).
- `tier2_anywhere`: any distinct entrance/exit pair on the 6x6 board (`arm`, `arm_l2`/`arm_l3`
  static parts plus `arm_piston` as the fallback for radius-changing handoff legs;
  single arms over rings 1-3 plus two-arm handoffs).
- `tier3_placed`: solution-placed (unfixed) entrypoints over a 16-config space
  (4 elementals x target 1-2 x two allow-lists), single static arm per solution.
"""

from opus_magnum_blackbox.families._common import (
    ENTRANCE_SHAPE,
    EXIT_SHAPE,
    RING_OFFSETS,
    BoardSize,
    FamilyError,
    hex_distance,
)
from opus_magnum_blackbox.families.tier1 import (
    ArmPlan,
    arm_plan,
    reference_oms,
    reference_solution,
    reference_tokens,
    single_atom_spec,
    verify,
)
from opus_magnum_blackbox.families.tier2_anywhere import (
    ANYWHERE_BLUEPRINTS,
    MAX_RING,
    AnywherePlan,
    ArmProgram,
    anywhere_plan,
    anywhere_reference_oms,
    anywhere_reference_solution,
    anywhere_reference_tokens,
    anywhere_spec,
    anywhere_spec_plan,
)
from opus_magnum_blackbox.families.tier3_placed import (
    ANCHOR_CELLS,
    LONG_BLUEPRINTS,
    SHORT_BLUEPRINTS,
    PlacedConfig,
    PlacedPlan,
    placed_configs,
    placed_plan,
    placed_reference_oms,
    placed_reference_solution,
    placed_reference_tokens,
    placed_spec,
    placed_spec_plan,
)

__all__ = [
    "ANCHOR_CELLS",
    "ANYWHERE_BLUEPRINTS",
    "ENTRANCE_SHAPE",
    "EXIT_SHAPE",
    "LONG_BLUEPRINTS",
    "MAX_RING",
    "RING_OFFSETS",
    "SHORT_BLUEPRINTS",
    "AnywherePlan",
    "ArmPlan",
    "ArmProgram",
    "BoardSize",
    "FamilyError",
    "PlacedConfig",
    "PlacedPlan",
    "anywhere_plan",
    "anywhere_reference_oms",
    "anywhere_reference_solution",
    "anywhere_reference_tokens",
    "anywhere_spec",
    "anywhere_spec_plan",
    "arm_plan",
    "hex_distance",
    "placed_configs",
    "placed_plan",
    "placed_reference_oms",
    "placed_reference_solution",
    "placed_reference_tokens",
    "placed_spec",
    "placed_spec_plan",
    "reference_oms",
    "reference_solution",
    "reference_tokens",
    "single_atom_spec",
    "verify",
]
