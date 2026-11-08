---
name: oms-authoring
description: Author and style Opus Magnum puzzle scripts — `.oms` solution files (DSL commands and tokens, component-ID allocation, arm hand positioning, required house style, demo-file conventions) and `.oml` level files (size/steps, local shape definitions with typed cells, entrance/exit placements, allow-list, fixed vs solution-placed entrypoints, declared atom types) — plus the validate/run/save workflow.
---

# Authoring `.oms` Solution Scripts and `.oml` Level Files

## When to use this skill

Use it whenever you create or edit a `.oms` solution file or a `.oml` level file
(usually under `examples/`), or are asked to script an Opus Magnum puzzle/level, lay out
components and step actions, define a level's starting conditions, or fix a validation/runtime error.

## How a `.oms` file works

A `.oms` file is a plain-text DSL, loaded line by line by
`SimInterpreter.load_file` (`src/opus_magnum_sym/interpreter/sim_interpreter.py`). Rules:

- Blank lines are ignored.
- A `#` starts a comment; it may be a full line or inline after a command.
- Each command is tokenized with `shlex`, so `key=value` stays atomic and quoting works if needed.
- Commands run top-to-bottom; component IDs follow this order (see "Component IDs").
- Parsing is lenient: a bad line prints `Error parsing line N: ...`, `Unknown command ...`,
  or `Failed to connect ...` and is skipped — it does not abort the load. Watch for these messages.

## Command reference

```text
comp <r> <s> <type> [key=value ...]      # place a component (alias: c)
place <blueprint> <r> <s> [rotation=R] [key=value ...]  # place a blueprint
action <comp_id> <step> <action_type>    # schedule a step action (alias: a)
remove <comp_id>                         # remove a component (alias: rm)
connection <id_a> <id_b>                 # link two components
disconnection <id_a> <id_b>              # unlink two components
```

- `comp`: `r` = row, `s` = column (hex `Hex(r, s)`). `<type>` is a `ComponentType` name, case-insensitive.
- `action`: `<comp_id>` is the component's id, `<step>` the 0-based simulation step,
  `<action_type>` a `StepActionType` name (case-insensitive).

## Component types

| token | class | required kwargs | notes |
|-------|-------|-----------------|-------|
| `arm` | `Manipulator` | — | optional `rotation`, `length`; only component that accepts `action` |
| `entrance` | `Enter` | `atom_type` | spawns an atom each step its cell is empty |
| `exit` | `Exit` | `atom_type` | consumes a matching molecule; required by `validate`/`run` |
| `connector` | `Connector` | `atom_type` | bonds atoms sitting on its glyph + connected glyph partners |
| `disconnector` | `Disconnector` | `atom_type` | breaks bonds between atoms on its glyph + connected partners |

`transporter` is an enum value but has no concrete class in `_COMPONENT_MAP`; using it in
`comp` raises `Unknown component`.

## Blueprints (`place`)

`place` expands a registered blueprint into one or more primitive components (and,
for shapes, their `connection` links) in a single line. `<blueprint>` is a name from
`list_blueprints()`. The two flavors:

- Shape templates (`connector_straight_2`, `connector_tri`, `connector_tri_odd`,
  `salt_transmuter`) expand into a multi-cell group with pre-wired links. Entrance/exit
  shapes are **not** registered globally; define them per-level (see the `.oml` section).
- Manipulator specs (`arm`) place a single arm with a stored capability descriptor.
  The piston arm `arm_piston` is also placeable. Multi-arm variants (`arm_double`,
  `arm_triple`, `arm_hex`) are registered but `implemented=False`; `place` on them
  raises a load-time error.

`rotation=R` rotates the whole shape around the origin cell; some shapes restrict the
allowed rotations (e.g. the Y shapes only accept {0,2,4} or {1,3,5}). Remaining
`key=value` pairs merge onto every placed cell — `atom_type=` reaches all cells of a
connector/entrance/exit shape. A per-cell typed `:type` (from a `.oml`-defined shape)
wins over a blanket `atom_type=` override. Note: `place` rejects `atom_type=basic` ("cannot use
atom_type 'basic'"); for basic atoms build the shape from `comp` lines plus `connection`.

## Component kwargs

- `atom_type=<name>` — required for `entrance`, `exit`, `connector`, `disconnector`.
  Valid names: `fire`, `water`, `earth`, `air`, `salt`, `metal`. `basic` and `none` are
  placeholders rejected by `place` and by level entrypoints (but accepted by `comp`).
- `rotation=<0..5>` — optional initial rotation (default `0`).
- `length=<int>` — arm hand length (default `1`); arm only.

Omitting a required `atom_type` makes the `comp` line fail at parse time (component not placed).

## Step actions

Usable on an arm:

| token | effect |
|-------|--------|
| `none` | remove any action assigned to that step (rarely used in files) |
| `grab` | pick up the whole molecule at the hand (no-op if empty) |
| `release` | drop the held molecule onto its current cells (no-op if blocked) |
| `rotate_clw` | full rotation: arm swings around its base, carrying the held molecule |
| `rotate_cclw` | full rotation, opposite direction |
| `rotate_molecule_clw` | in-place: arm stays, held molecule rotates around the hand |
| `rotate_molecule_cclw` | in-place, opposite direction |

`extend`/`retract` raise `ActionAssignmentError` on the basic arm but work on the piston arm
(`arm_piston`). `move_pos`/`move_neg` are declared but unimplemented everywhere.

`action` targets only non-entrypoint components (in practice, arms). Assigning an
`action` to an entrance/exit/connector/disconnector id raises a load-time error (the line is skipped).

## Coordinates, board size, adjacency

- The CLI creates a fixed `Simulator((10, 10))`, so valid coordinates are `0..9` for
  both `r` and `s`. Out-of-bounds `comp` lines fail placement.
- Hex cells are adjacent along the six `Direction.H*` offsets. `connection`/`disconnection`
  only work between **adjacent** cells of the **same type** (and only for
  entrance/exit/connector/disconnector).

## Component IDs

- IDs are allocated by `index_manager.allocate()` starting at `0`, in the order `comp` lines appear.
- `remove` frees an id back into the pool, so a later `comp` may reuse it. When you write
  `# id n`, use the id the component actually receives at that point in the file.
- Document every `comp` line with an inline `# id n` comment (house rule below).

## Arm hand position

The hand sits at the arm's cell plus the rotation offset (scaled by `length`):

| rotation | direction | offset (dr, ds) |
|----------|-----------|-----------------|
| 0 | H1 | (−1, +1) |
| 1 | H3 | (0, +1) |
| 2 | H5 | (+1, 0) |
| 3 | H7 | (+1, −1) |
| 4 | H9 | (0, −1) |
| 5 | H11 | (−1, 0) |

Hand cell = `(r + dr*length, s + ds*length)`. Default `rotation=0`, `length=1` → hand at `(r−1, s+1)`.

## House style (required)

Follow these rules for every new `.oms` file:

1. **Leading blank line** — the file starts with one empty line before anything else.
2. **Context comment** — immediately after, a short, dry comment stating the scenario,
   e.g. `# verification of connector/disconnector components`. One or two lines.
3. **Logical sections** — group related lines (components, connections, actions) and
   separate sections with one blank line.
4. **Section comments only when needed** — annotate a section only when it is not
   self-explanatory or is important to the scenario. Do not comment every section or every step.
5. **`# id n` on every component** — each `comp` line ends with an inline `# id n`
   showing the allocated id.
6. **`place` lines in their own section** — put `place` lines in a section after all
   `comp` lines, with a context comment noting the blueprint and its atom type.
7. **Ascending step order** — within a component, list `action` lines by increasing `step`.
8. **Deprecation marker** — a file that uses the low-level `comp`/`connection` primitives
   instead of `place <blueprint>` marks its header with a `# DEPRECATED - uses low-level
   `comp` primitives.` line, separated from the context comment by one blank `#` line.
   Files built on `place` (or `.oml` levels) carry no marker.

These rules are cosmetic; the parser ignores blank lines and comments. They exist for
human and agent readability.

## Demonstration files (feature / component demos)

An example written to demonstrate a feature or component is meant to be *watched* in the
pygame viewer, not scored. Unless the user explicitly asks for a minimal or tightly packed
program, every demo `.oms` file follows the house style **plus** these defaults:

1. **Leave gaps in every program.** Never schedule actions on consecutive steps. A gap is
   simply an unassigned step index — the component executes `NONE` there and the viewer gets a
   frame where the previous pose is readable. Leave at least two idle steps between meaningful
   actions: `1 grab`, `4 rotate_cclw`, `7 rotate_cclw`, `10 release` instead of `1, 2, 3, 4`.
   Do not pad gaps with `none` lines; skip the step number.
2. **Keep at least one loose atom or molecule on the board.** A demo that funnels every atom
   into an exit ends on an empty board and shows nothing. Always leave something unclaimed:
   - an extra entrance outside every arm's reach (its atom spawns and stays);
   - an arm that grabs a molecule and releases it on open board instead of into an exit;
   - two adjacent `entrance` lines joined with `connection` — linked entrances bond their
     spawned atoms, so a bonded 2-atom molecule sits on the board.
3. **Show molecules, not just atoms.** Bonds are rendered, so atoms only look like a molecule
   when they are actually bonded: linked entrances, linked `connector` glyphs, or an arm that
   grabs a bonded group. Unbonded neighbours read as separate atoms.
4. **Keep the file runnable.** `validate`/`run` require one entrance and one exit, so keep an
   exit even if the demo never delivers to it. Loose atoms must not cover an exit's group cells,
   or the exit consumes them.
5. **Comment the leftovers.** One dry comment on the block that produces the spare
   atom/molecule, e.g. `# spare molecule stays for visualization`.
6. **Size the run past the last action.** `run <file> N` / `!run N` must reach the last
   scheduled step plus a few idle steps, otherwise the final pose never renders
   (last action at 10 → `--steps 16`).

Example of the pattern (idle gaps plus a bonded molecule left on the board):

```text

# demo: linked entrances spawn a bonded 2-atom molecule; the arm grabs it and leaves it on open board

# exit is required by validate/run; nothing is delivered here
comp 1 1 exit atom_type=basic # id 0

# linked entrances bond their spawned atoms into one molecule
comp 5 4 entrance atom_type=basic # id 1
comp 4 5 entrance atom_type=basic # id 2
connection 1 2

comp 5 5 arm rotation=4 # id 3 (hand at (5,4))

# two idle steps between phases
action 3 1 grab
action 3 4 rotate_cclw
action 3 7 release
```

The canonical template below packs its actions on consecutive steps; that pacing is fine for a
syntax reference but must not be copied into a demo file.

## Canonical template

See `templates/basic.oms`, reproduced here:

```text

# minimal entrance -> arm -> exit delivery
#
# DEPRECATED - uses low-level `comp` primitives.

# Entrance spawns a basic atom at (2,3).
comp 2 3 entrance atom_type=basic # id 0

# Exit consumes the atom once it is released here.
comp 3 2 exit atom_type=basic     # id 1

# Arm at (3,3): rotation 0 -> hand at (2,4); length defaults to 1.
comp 3 3 arm                      # id 2

# Move the atom from the entrance to the exit.
action 2 1 rotate_cclw
action 2 2 grab
action 2 3 rotate_cclw
action 2 4 release
```

## Authoring `.oml` Level Files

A `.oml` file defines a level's **starting conditions** (board size, step limit, entrypoint
shapes and placements, allow-list); the solution is a separate `.oms` file. The two are paired:
`examples/26_level_fixed.oml` (level) + `examples/26_level_fixed.oms` (solution).

### How a `.oml` file works

Loaded line by line by `LevelLoader` (`src/opus_magnum_sym/interpreter/level_loader.py`),
which builds a validated `LevelSpec`. Unlike `SimInterpreter` (lenient), the loader is
**strict**: any error raises `LevelLoadError` with the offending line number and aborts the load.

- Blank lines and full-line `#` comments are ignored.
- `key=value` tokens stay atomic (same `shlex` splitting as `.oms`).
- `shape` definitions must precede the `entrance`/`exit` lines that reference them.

### Command reference

```text
size <rows> <cols>
steps <n>
allow <name> [name ...]
atom_types <type> [type ...]
entrypoints_fixed 0|1
shape <name> <kind> cells=(r,s)[:type],(r,s)[:type],... links=i-j[,i-j...] [rotations=0,1,...]
entrance <shape> [<r> <s>] [rotation=k] [atom_type=<type>]
exit <shape> [<r> <s>] [rotation=k] [atom_type=<type>]
```

- `size` and `steps` are required and must be positive.
- `shape <name> <kind>` defines a local `ShapeTemplate`. `<kind>` is a `ComponentType` name
  restricted to entrypoint kinds — `entrance`, `exit`, `connector`, `disconnector`,
  `salt_transmuter` (not `arm`). `cells=(r,s),...` lists offsets relative to the origin; exactly
  one cell must be `(0,0)` and offsets must be distinct. `links=i-j` pairs adjacent cell indexes
  (use `links=` for a single-cell shape). `rotations=` restricts allowed rotations (omit for any).
- **Typed cells**: each cell may carry a colon-suffixed atom type — `(0,0):salt,(-1,1):water`.
  A typed cell overrides the entrypoint's `atom_type`; an untyped cell falls back to it. The
  loader rejects unknown typed-cell keys and non-`AtomType`/`basic`/`none` values.
- `entrance`/`exit` place an entrypoint using a **locally defined** `shape` (never a global
  blueprint). `rotation=k` (default 0) and `atom_type=<type>` are `key=value` pairs; their order
  after the position is free. `atom_type` must be a real atom type — `basic`/`none` are rejected —
  and is **omittable only when every cell of the shape is typed**.
- **Fixed vs unfixed**: `entrypoints_fixed 1` (default) requires the `<r> <s>` position on every
  `entrance`/`exit` (the level places them). With `entrypoints_fixed 0` the position is omitted and
  the solution must `place` each defined entrypoint shape exactly once (matching per shape name).
- `atom_types <type> [type ...]` declares the level's used atom types (optional; empty =
  unconstrained). When non-empty, every entrypoint type — fallback and typed cells — must be a
  member.
- `allow <name> [name ...]` lists the blueprints the solution may `place`; each name resolves
  against the level's local shapes plus the global default registry and must be `implemented`
  (`arm` and `arm_piston` are; `arm_double`/`arm_triple`/`arm_hex` are not). Entrypoint shapes
  need not be listed — in unfixed mode the level auto-adds them to the allow-list.

### Invariants

- The spec must have at least one entrance and one exit; entrypoint cells (shape offsets +
  rotation) must be in bounds and pairwise non-overlapping (fixed mode) — the loader raises
  otherwise. In unfixed mode every entrypoint must omit its position.
- A level runs with its own env (fresh id allocator and result counter) and a cloned blueprint
  registry. In fixed mode entrypoints are placed first, so the solution's component ids start
  after them (in `examples/26_level_fixed.oms` the arm is `# id 4`). In unfixed mode the solution
  places the entrypoints itself, so their ids come from the solution's own `place` order.
- The exit verifies the delivered molecule's **bond pattern exactly**: for every pair of covered
  cells, a bond must exist iff the two exit components are directly linked (no missing, no extra
  bonds). A molecule whose bonds mismatch the shape's `links` is rejected, not consumed.

### House style

Follow `examples/26_level_fixed.oml` (minimal copy: `templates/basic.oml`):

1. **Context comment block first** — a dry comment stating the scenario, that this is the
   level, and pointing at its solution file.
2. **`size` then `steps`** at the top.
3. **`shape` definitions grouped** in their own section, before the entrypoints that use them.
4. **`entrance`/`exit` placements grouped**, one comment each when the geometry matters.
5. **`allow` last** — the solution's available blueprints.

```text
# Reference level demo: delivery through straight-2 entrance and exit pair.
#
# The level file defines the starting conditions.
# The solution defines placement and actions.
# The solution lives in `examples/26_level_fixed.oms`

size 6 6
steps 30

# Entrypoint shapes are defined locally.
shape entrance_straight_2 entrance cells=(0,0),(-1,1) links=0-1
shape exit_straight_2 exit cells=(0,0),(-1,1) links=0-1

# Entrance pair spawns bonded 2-atom molecule on step 0.
entrance entrance_straight_2 2 3 rotation=1 atom_type=salt

# Exit pair consumes molecule only if it covers both cells exactly.
exit exit_straight_2 3 2 rotation=2 atom_type=salt

# Blueprints available to the solution.
allow arm
```

### Typed cells & solution-placed entrypoints

Follow `examples/27_level_typed.oml` when a level uses typed cells or leaves entrypoint
placement to the solution:

1. Put `entrypoints_fixed 0` (and `atom_types ...` when declared) directly after `steps`,
   before the `shape` definitions.
2. Give every cell of a typed shape a `:type` suffix so no `atom_type` fallback is needed;
   otherwise keep `atom_type=` on the entrypoint for the untyped cells.
3. In unfixed mode, `entrance`/`exit` take no position; the solution `place`s them (the level
   auto-adds entrypoint shapes to the allow-list).

```text
# Typed, solution-placed entrypoints: a water+salt molecule is delivered to a
# typed exit that verifies the exact bond pattern.
#
# The level defines the typed shapes and leaves entrypoint placement to the
# solution (`entrypoints_fixed 0`).
# The solution lives in `examples/27_level_typed.oms`

size 6 6
steps 30
entrypoints_fixed 0
atom_types salt water

# The arm's 4-step swing mirrors the molecule, so the entrance's origin cell is
# water and its beam salt; the exit carries the opposite assignment.
shape entrance_typed entrance cells=(0,0):water,(-1,1):salt links=0-1
shape exit_typed exit cells=(0,0):salt,(-1,1):water links=0-1

# Both cells are typed, so no `atom_type` fallback is needed.
entrance entrance_typed
exit exit_typed

# Blueprints available to the solution (entrypoint shapes are added automatically).
allow arm
```

## Authoring & verification workflow

```bash
# Validate syntax + entrance/exit presence (does not run the simulation)
just validate examples/my_level.oms

# Run headlessly and print the delivered-molecule count
just run examples/my_level.oms 30

# Run a level (starting conditions) + its solution together
uv run opus-magnum run --level examples/my_level.oml examples/my_level.oms
```

## Caveats / known limitations that affect scripts

- Entrances re-spawn the instant their cell is emptied, so a spare atom can remain after an arm grabs.
- Steps run non-entrypoint components first, then entrypoints; simultaneous actions can race.
- `!save` / `!edit` emit a minimal canonical form and drop `connection`/`disconnection`
  links, `# id n` comments, section comments, and the leading blank line. Treat
  hand-written files as the source of truth for style and connections.
- `connection` is the only link keyword; there is no `disconnect` alias (use `disconnection`).
- `AtomType` now includes `basic`, `fire`, `water`, `earth`, `air`, `salt`, and `metal`.
  An exit consumes a molecule only when it covers the exit cluster exactly, each covered atom
  matches its cell's type (per-cell typed type, else the entrypoint fallback), **and** the
  molecule's bonds equal the exit shape's links exactly — a missing or extra bond is rejected.
  Entrances likewise bond their spawned atoms only along direct links, so a branched entrance
  shape spawns a molecule bonded exactly along its `links`.
