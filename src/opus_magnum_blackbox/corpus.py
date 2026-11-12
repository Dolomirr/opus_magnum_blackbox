"""
Level corpus: `.oml` files under `levels/` plus their RL config entries.

Scans `levels/` recursively: if neseted directory is presented in `levels/`,
treat it as train/test catalogue for single stage/tier of training.
Each pair of `.oms` and `.oml` is single entry of level+solution.
Resolve name at single level, or a directory for group
of every level under it (recursively, sorted).

Config comes from the nearest `manifest.toml` in or above the level's directory:

- `cost_max` (int, required)
- `weight` (default 1.0).

Top-level levels without any manifest keep default (`RewardConfig().cost_max`).

Example layout of `levels/`:

```
.
├── tier0.oml
├── tier0.oms
└── tier1
    ├── eval
    │   ├── vNN.oml
    │   └── vNN.oms
    │── train
    │   ├── vNN.oml
    │   └── vNN.oms
    └── manifest.toml
```

"""

import tomllib
from pathlib import Path

from opus_magnum_blackbox.config import RewardConfig
from opus_magnum_blackbox.env import LevelEntry

LEVELS_DIR = Path(__file__).resolve().parents[2] / "levels"

MANIFEST_NAME = "manifest.toml"

# Explicit entries for levels before the scanner:
# name, cost_max and solutions are explicitly provided in files themselfs.
TIER0_LEVELS = [
    LevelEntry(
        source=LEVELS_DIR / "tier0_typed_delivery.oml",
        cost_max=50,
        name="tier0_typed_delivery",
        solutions=(LEVELS_DIR / "tier0_typed_delivery.oms",),
    ),
]


def scan_levels(levels_dir: Path = LEVELS_DIR) -> dict[str, LevelEntry]:
    """
    Registry of corpus: explicit `TIER0_LEVELS` and one scanned entry
    per `.oml` under `levels_dir`.
    Explicit entries takes priority on conflict.
    """
    registry = {entry.name: entry for entry in TIER0_LEVELS}
    manifests: dict[Path, dict | None] = {}
    for oml in sorted(levels_dir.rglob("*.oml")):
        name = oml.relative_to(levels_dir).with_suffix("").as_posix()
        solutions = (oms,) if (oms := oml.with_suffix(".oms")).exists() else ()
        cost_max, weight = _entry_config(oml, levels_dir, manifests)
        registry.setdefault(
            name, LevelEntry(source=oml, cost_max=cost_max, name=name, weight=weight, solutions=solutions)
        )
    return registry


def resolve_entries(names: list[str], levels_dir: Path = LEVELS_DIR) -> list[LevelEntry]:
    """
    Corpus entries for `names`, in given order;
    directory name resolves to every level under it,
    recursively, sorted.
    Unknown names raise an error.
    """
    registry = scan_levels(levels_dir)
    entries: list[LevelEntry] = []
    missing: list[str] = []
    for name in names:
        if name in registry:
            entries.append(registry[name])
        elif (levels_dir / name).is_dir():
            group = [entry for entry_name, entry in registry.items() if entry_name.startswith(f"{name}/")]
            if not group:
                raise ValueError(f"level group {name!r} contains no level .oml files")
            entries.extend(sorted(group, key=lambda entry: entry.name))
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"unknown levels: {missing}; corpus has: {sorted(registry)}")
    return entries


def _entry_config(
    oml: Path,
    levels_dir: Path,
    manifests: dict[Path, dict | None],
) -> tuple[int, float]:
    """
    For each scanned level reads (`cost_max`, `weight`)
    from manifest in or above its directory.
    """
    for directory in (oml.parent, *oml.parent.parents):
        manifest = directory / MANIFEST_NAME
        if manifest not in manifests:
            manifests[manifest] = _read_manifest(manifest)
        data = manifests[manifest]
        if data is not None:
            return _manifest_config(data, manifest)
        if directory == levels_dir:
            break
    if oml.parent == levels_dir:
        return RewardConfig().cost_max, 1.0
    name = oml.relative_to(levels_dir).as_posix()
    raise ValueError(
        f"level {name!r}: no {MANIFEST_NAME} in {oml.parent} or any directory above it;"
        + " subdirectory levels must declare cost_max"
    )


def _read_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("rb") as file:
        return tomllib.load(file)


def _manifest_config(data: dict, manifest: Path) -> tuple[int, float]:
    cost_max = data.get("cost_max")
    if not isinstance(cost_max, int):
        raise TypeError(f"{manifest}: integer `cost_max` is required, got {cost_max!r}")
    weight = data.get("weight", 1.0)
    if not isinstance(weight, (int, float)):
        raise TypeError(f"{manifest}: `weight` must be a number, got {weight!r}")
    return cost_max, float(weight)
