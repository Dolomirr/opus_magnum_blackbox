"""
Family modules: fixed-entrypoint level builders and simulator-verified reference solutions.

- `tier1`: single-atom family (6x6 board, `arm` only, hex-distance 1-2).
- `tier2_anywhere`: any distinct entrance/exit pair on the 6x6 board (`{arm, arm_piston}`,
  single arms over rings 1-3 plus two-arm handoffs).
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
)

__all__ = [
    "ANYWHERE_BLUEPRINTS",
    "ENTRANCE_SHAPE",
    "EXIT_SHAPE",
    "MAX_RING",
    "RING_OFFSETS",
    "AnywherePlan",
    "ArmPlan",
    "ArmProgram",
    "BoardSize",
    "FamilyError",
    "anywhere_plan",
    "anywhere_reference_oms",
    "anywhere_reference_solution",
    "anywhere_reference_tokens",
    "anywhere_spec",
    "arm_plan",
    "hex_distance",
    "reference_oms",
    "reference_solution",
    "reference_tokens",
    "single_atom_spec",
    "verify",
]
