"""
`.oms` solution loading: known solutions to validated token sequences.

Supported DSL subset (mirrors what token grammar can represent):
- `place <name> <r> <s> [rotation=R]`
- `action <comp_id> <tick> <TYPE>`

File is replayed on a scratch `Level` to resolve component ids,
translated to solution IR, encoded, and validated on load.
"""

# Now way I am doing this manuall again.

import random
import shlex
from pathlib import Path

from opus_magnum_sym import Hex, Level, LevelSpec, StepActionType

from opus_magnum_blackbox.codec import OpusTokenizer, apply_solution, decode_solution
from opus_magnum_blackbox.env import LevelEntry
from opus_magnum_blackbox.grammar import (
    Action,
    GrammarError,
    Placement,
    Solution,
    spec_model,
)
from opus_magnum_blackbox.masking import ActionMasker
from opus_magnum_blackbox.vocab import Vocab


class SolutionLoadError(Exception):
    """A `.oms` solution cannot be represented as valid token sequence."""


def load_solutions(entry: LevelEntry) -> list[list[int]]:
    """
    One validated token sequence per `.oms` file listed on the entry.
    """
    tokenizer = OpusTokenizer()
    return [_load_solution_file(path, entry, tokenizer) for path in entry.solutions]


def augment(tokens: list[int], spec: LevelSpec, seed: int, max_variants: int) -> list[list[int]]:
    """
    Deterministic validated variants of one solution token sequence.

    Layout quadruplets are kept mixed, so that placements share
    entrypoint slot in relative order and relabelling program
    handles to follow the new manipulator order;
    triplets are mixed within in single.
    The original sequence is not included;
    only `max_variants` distinct sequences are returned.
    """
    tokenizer = OpusTokenizer()
    parsed = decode_solution(list(tokens), spec, tokenizer.vocab)
    if isinstance(parsed, GrammarError):
        raise SolutionLoadError(f"cannot augment: decode failed at token {parsed.pos}: {parsed.message}")

    rng = random.Random(seed)
    groups = _layout_groups(parsed.placements)
    old_handles = _handles(parsed.placements)
    variants: list[list[int]] = []
    seen = {tuple(tokens)}
    for _ in range(8 * max_variants):
        if len(variants) >= max_variants:
            break
        order = [*groups]
        rng.shuffle(order)
        indexed = [(index, placement) for group in order for index, placement in group]
        handle_map, handle = {}, 0
        for index, _ in indexed:
            if index in old_handles:
                handle_map[old_handles[index]] = handle
                handle += 1
        triplets: dict[int, list[Action]] = {}
        for action in parsed.actions:
            triplets.setdefault(handle_map[action.handle], []).append(action)
        actions: list[Action] = []
        for new_handle in sorted(triplets):
            group = triplets[new_handle][:]
            rng.shuffle(group)
            actions.extend(group)
        ir = Solution(placements=[p for _, p in indexed], actions=actions, submitted=True, length=0)
        variant = tokenizer.encode_solution(ir.placements, actions, spec)
        if tuple(variant) in seen:
            continue
        _validate(variant, ir, spec, tokenizer, "augmented variant")
        seen.add(tuple(variant))
        variants.append(variant)
    return variants


def _load_solution_file(path: Path, entry: LevelEntry, tokenizer: OpusTokenizer) -> list[int]:
    level = Level.from_file(entry.source) if isinstance(entry.source, Path) else Level(entry.source)
    spec = level.spec
    placements, actions = _parse_solution_file(path, level, spec, tokenizer.vocab)
    actions.sort(key=lambda action: (action.handle, action.tick))
    ir = Solution(placements=placements, actions=actions, submitted=True, length=0)
    tokens = tokenizer.encode_solution(placements, actions, spec)
    _validate(tokens, ir, spec, tokenizer, str(path))
    return tokens


def _parse_solution_file(
    path: Path,
    level: Level,
    spec: LevelSpec,
    vocab: Vocab,
) -> tuple[list[Placement], list[Action]]:
    model = spec_model(spec)
    placements: list[Placement] = []
    actions: list[Action] = []

    # `.oms` component id -> program handle
    comp_handles: dict[int, int] = {}
    for line_num, words in _words(path):
        command, args = words[0], words[1:]
        if command == "place":
            if len(args) < 3:
                raise SolutionLoadError(f"{path}:{line_num}: 'place' expects <name> <r> <s> [rotation=R]")
            name, r, s = args[0], int(args[1]), int(args[2])
            rotation = 0
            for arg in args[3:]:
                key, value = arg.split("=", 1)
                if key != "rotation":
                    raise SolutionLoadError(f"{path}:{line_num}: unsupported 'place' argument {arg!r}; only rotation=R")
                rotation = int(value)
            if name in model.slots:
                slot, builtin = model.slots.index(name), None
            elif name in model.builtins:
                slot, builtin = None, vocab.builtin(vocab.builtin_id(name))
            else:
                raise SolutionLoadError(
                    f"{path}:{line_num}: blueprint {name!r} is neither an entrypoint shape nor an allowed built-in"
                )
            try:
                record = level.place(name, Hex(r, s), rotation=rotation)
            except Exception as exc:
                raise SolutionLoadError(f"{path}:{line_num}: placement failed: {exc}") from exc
            if builtin is not None and builtin.manipulator:
                comp_handles[record.comp_id] = len(comp_handles)
            placements.append(
                Placement(
                    builtin=builtin,
                    slot=slot,
                    template=model.templates[slot] if slot is not None else builtin.template,
                    rotation=rotation,
                    q=s,
                    r=r,
                    pos=0,
                )
            )
        elif command == "action":
            actions.append(_action(args, comp_handles, path, line_num))
        else:
            raise SolutionLoadError(
                f"{path}:{line_num}: unsupported command {command!r}; only 'place' and 'action' are supported"
            )
    return placements, actions


def _action(args: list[str], comp_handles: dict[int, int], path: Path, line_num: int) -> Action:
    """
    One `action` line, with `.oms` component id mapped to a program handle.
    """
    if len(args) != 3:
        raise SolutionLoadError(f"{path}:{line_num}: 'action' expects <comp_id> <tick> <TYPE>")
    comp_id, tick = int(args[0]), int(args[1])
    try:
        step_type = StepActionType[args[2].upper()]
    except KeyError:
        raise SolutionLoadError(f"{path}:{line_num}: unknown action type {args[2]!r}") from None
    if comp_id not in comp_handles:
        raise SolutionLoadError(
            f"{path}:{line_num}: action references component {comp_id}, which is not a placed manipulator"
        )
    return Action(handle=comp_handles[comp_id], tick=tick, action=step_type, pos=0)


def _words(path: Path) -> list[tuple[int, list[str]]]:
    """
    `(line number, shlex words)` for every non-empty line, comments stripped.
    """
    words = []
    for line_num, line in enumerate(path.read_text().splitlines(), 1):
        parts = shlex.split(line, comments=True)
        if parts:
            words.append((line_num, parts))
    return words


def _validate(tokens: list[int], ir: Solution, spec: LevelSpec, tokenizer: OpusTokenizer, source: str) -> None:
    """
    Full load-time check: decode round-trip, apply+run solves, mask-prefix walk.
    """
    vocab = tokenizer.vocab
    decoded = decode_solution(tokens, spec, vocab)
    if isinstance(decoded, GrammarError):
        raise SolutionLoadError(f"{source}: decode failed at token {decoded.pos}: {decoded.message}")
    if _fingerprint(decoded) != _fingerprint(ir):
        raise SolutionLoadError(f"{source}: decoded sequence does not round-trip the solution")

    level = Level(spec)
    report = apply_solution(decoded, spec, level)
    if not report.ok:
        raise SolutionLoadError(f"{source}: apply failed: {report.error}")
    result = level.run()
    level.reset()
    if not result.solved:
        raise SolutionLoadError(f"{source}: solution does not solve the level (status={result.status.value})")

    masker = ActionMasker(vocab)
    for position, token in enumerate(tokens):
        if not masker.mask(tokens[:position], spec)[token]:
            raise SolutionLoadError(f"{source}: token {position} ({vocab.name(token)}) is not mask-valid at its prefix")


def _fingerprint(sol: Solution) -> tuple[list[tuple], list[tuple]]:
    """
    Order-sensitive placement keys plus order-insensitive action keys.
    """
    placements = [(p.slot, p.builtin.name if p.builtin else None, p.rotation, p.q, p.r) for p in sol.placements]
    actions = sorted(((a.handle, a.tick, a.action) for a in sol.actions), key=lambda item: item[:2])
    return placements, actions


def _layout_groups(placements: list[Placement]) -> list[list[tuple[int, Placement]]]:
    """
    Permutable placement groups: placements sharing an entrypoint slot stay
    together (in order), each built-in placement commutes on its own.
    """
    groups: dict[tuple, list[tuple[int, Placement]]] = {}
    for index, placement in enumerate(placements):
        key = ("slot", placement.slot) if placement.slot is not None else ("builtin", index)
        groups.setdefault(key, []).append((index, placement))
    return list(groups.values())


def _handles(placements: list[Placement]) -> dict[int, int]:
    """
    Placement index -> program handle: ordinal among manipulator placements,
    mirroring how `apply_solution` builds handles.
    """
    handles: dict[int, int] = {}
    for index, placement in enumerate(placements):
        if placement.builtin is not None and placement.builtin.manipulator:
            handles[index] = len(handles)
    return handles
