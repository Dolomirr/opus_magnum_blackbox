"""
Level corpus: `.oml` files under `levels/` plus their RL config entries.
"""

from pathlib import Path

from opus_magnum_blackbox.env import LevelEntry

# TODO: i may need to do some generator...

LEVELS_DIR = Path(__file__).resolve().parents[2] / "levels"

TIER0_LEVELS = [
    LevelEntry(
        source=LEVELS_DIR / "tier0_typed_delivery.oml",
        cost_max=50,
        name="tier0_typed_delivery",
        solutions=(LEVELS_DIR / "tier0_typed_delivery.oms",),
    ),
]
