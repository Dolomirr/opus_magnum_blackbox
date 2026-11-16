"""
Level corpus: `.oml` files under `levels/` plus their RL config entries.

Scans `levels/` recursively: if neseted directory is presented in `levels/`,
treat it as train/test catalogue for single stage/tier of training.
Each pair of `.oms` and `.oml` is single entry of level+solution.
Resolve name at single level, or a directory for group
of every level under it (recursively, sorted).

Config comes from the nearest `manifest.toml` in or above the level's directory:

- `cost_max` (int, required)
- `weight` (default 1.0)
- `level_weights` (optional table): `"train/v123" = 3.0` overrides the directory
  weight for that named variant, with the key relative to the manifest directory
  (suffix dropped); a key matching no scanned level raises.

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
    weighted: dict[Path, set[str]] = {}
    for oml in sorted(levels_dir.rglob("*.oml")):
        name = oml.relative_to(levels_dir).with_suffix("").as_posix()
        solutions = (oms,) if (oms := oml.with_suffix(".oms")).exists() else ()
        cost_max, weight, override = _entry_config(oml, levels_dir, manifests, weighted)
        if override is not None:
            weight = override
        registry.setdefault(
            name, LevelEntry(source=oml, cost_max=cost_max, name=name, weight=weight, solutions=solutions)
        )
    _check_level_weights(manifests, weighted)
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
    weighted: dict[Path, set[str]],
) -> tuple[int, float, float | None]:
    """
    For each scanned level reads (`cost_max`, `weight`) from the manifest in or
    above its directory, plus an optional per-level `level_weights` override.

    Consumed manifest-relative names are recorded per manifest.
    """
    for directory in (oml.parent, *oml.parent.parents):
        manifest = directory / MANIFEST_NAME
        if manifest not in manifests:
            manifests[manifest] = _read_manifest(manifest)
        data = manifests[manifest]
        if data is not None:
            name = oml.relative_to(directory).with_suffix("").as_posix()
            weighted.setdefault(manifest, set()).add(name)
            return (*_manifest_config(data, manifest), _level_weight(data, name, manifest))
        if directory == levels_dir:
            break
    if oml.parent == levels_dir:
        return RewardConfig().cost_max, 1.0, None
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


def _level_weight(data: dict, name: str, manifest: Path) -> float | None:
    """
    Per-level sampling weight for level from the manifest.
    """
    weights = data.get("level_weights")
    if weights is None:
        return None
    if not isinstance(weights, dict):
        raise TypeError(f"{manifest}: `level_weights` must be a table, got {weights!r}")
    weight = weights.get(name)
    if weight is None:
        return None
    if not isinstance(weight, (int, float)):
        raise TypeError(f"{manifest}: `level_weights[{name!r}]` must be a number, got {weight!r}")
    if weight <= 0:
        raise ValueError(f"{manifest}: `level_weights[{name!r}]` must be positive, got {weight!r}")
    return float(weight)


def _check_level_weights(manifests: dict[Path, dict | None], weighted: dict[Path, set[str]]) -> None:
    """
    Every level_weights key must name level scanned under its related manifest.
    """
    for manifest, data in manifests.items():
        if not isinstance(data, dict) or not isinstance(data.get("level_weights"), dict):
            continue
        unknown = sorted(set(data["level_weights"]) - weighted.get(manifest, set()))
        if unknown:
            raise ValueError(f"{manifest}: `level_weights` name(s) not in the corpus: {unknown}")
