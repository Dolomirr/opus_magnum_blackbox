"""
Spec encoder and solution codec.

Coordinate mapping (owned here; nothing else reasons about `Hex`):
token `Q` -> `Hex.s` (column, bounded by `spec.size[1]`),
token `R` -> `Hex.r` (row, bounded by `spec.size[0]`).

Entrypoint-shape slot tokens (`ESHAPE_i`) resolve against the current spec:
slots enumerate the unique shape names of `spec.entrypoints` in order of first
occurrence. Applying an entrypoint placement routes to the n-th spec entrypoint
of that shape name (n = times the slot was already placed) via
`Level.place_entrypoint`; built-in blueprint placements route to `Level.place`.
"""

from dataclasses import dataclass

import numpy as np
from opus_magnum_sym import Hex, Level, LevelSpec
from opus_magnum_sym.elements.components import ComponentType

from opus_magnum_blackbox.config import TokenizerConfig
from opus_magnum_blackbox.grammar import (
    GrammarError,
    Solution,
    entrypoint_index,
    parse,
    solution,
    spec_model,
)
from opus_magnum_blackbox.vocab import Vocab


@dataclass
class DecodeReport:
    """
    Outcome of decoding token sequence onto a `Level`.
    """

    ok: bool
    error: str | None = None
    error_token_pos: int | None = None
    placements: int = 0
    program_actions: int = 0


def decode_solution(tokens: list[int] | np.ndarray, spec: LevelSpec, vocab: Vocab) -> Solution | GrammarError:
    """
    Pure parse: token sequence to `Solution`.
    """
    state = parse(tokens, spec, vocab)
    return state if isinstance(state, GrammarError) else solution(state)


def apply_solution(sol: Solution, spec: LevelSpec, level: Level) -> DecodeReport:
    """
    Place/assign from IR  onto resetted `Level`.
    Never raises exceptions.
    """
    model = spec_model(spec)
    slot_use: dict[int, int] = {}
    handles: list[int] = []
    for i, placement in enumerate(sol.placements):
        pos = Hex(r=placement.r, s=placement.q)
        try:
            if placement.slot is not None:
                use = slot_use.get(placement.slot, 0)
                slot_use[placement.slot] = use + 1
                index = entrypoint_index(model, placement.slot, use)
                record = level.place_entrypoint(index, pos, rotation=placement.rotation)
            else:
                record = level.place(placement.builtin.name, pos, rotation=placement.rotation)
                if placement.builtin.manipulator:
                    handles.append(record.comp_id)
        except Exception as exc:
            return DecodeReport(ok=False, error=f"placement failed: {exc}", error_token_pos=placement.pos, placements=i)

    for action in sol.actions:
        try:
            level.assign_step_action(handles[action.handle], action.tick, action.action)
        except Exception as exc:
            return DecodeReport(
                ok=False,
                error=f"action assignment failed: {exc}",
                error_token_pos=action.pos,
                placements=len(sol.placements),
            )

    return DecodeReport(ok=True, placements=len(sol.placements), program_actions=len(sol.actions))


class OpusTokenizer:
    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.vocab = Vocab(self.config.vocab)

    def encode_level_spec(self, spec: LevelSpec) -> tuple[np.ndarray, int]:
        """
        Encode spec as fixed-length padded id array + its true length.

        Layout:
        - header (mode, rows, cols, steps, target),
        - allowed built-ins,
        - per-shape segments (slot, comp type, cell triplets, link pairs),
        - per-entrypoint segments (slot, rotation, position or UNPLACED)
        """
        vocab = self.vocab
        rows, cols = spec.size
        model = spec_model(spec)
        shapes_by_name = {shape.name: shape for shape in spec.shapes}

        # fmt: off
        tokens = [
            vocab.value_id("mode", spec.entrypoints_fixed),
            vocab.value_id("r",    rows),
            vocab.value_id("q",    cols),
            vocab.value_id("tick", spec.steps),
            vocab.value_id("tick", spec.target) if spec.target else vocab.special_id("UNPLACED"),
        ]
        # fmt: on
        tokens.extend(vocab.builtin_id(name) for name in sorted(model.builtins))
        for slot, shape_name in enumerate(model.slots):
            shape = shapes_by_name[shape_name]
            tokens.append(vocab.value_id("entry_slot", slot))
            tokens.append(vocab.ctype_id(shape.comp_type is ComponentType.EXIT))
            for cell in shape.cells:
                tokens += [
                    vocab.value_id("delta", cell.offset.r),
                    vocab.value_id("delta", cell.offset.s),
                    vocab.atom_id(cell.atom_type),
                ]
            for a, b in shape.links:
                tokens += [vocab.value_id("cell", a), vocab.value_id("cell", b)]
        for ep in spec.entrypoints:
            tokens.append(vocab.value_id("entry_slot", model.slots.index(ep.shape)))
            tokens.append(vocab.value_id("rotation", ep.rotation))
            if ep.pos is None:
                tokens.append(vocab.special_id("UNPLACED"))
            else:
                tokens += [vocab.value_id("r", ep.pos.r), vocab.value_id("q", ep.pos.s)]

        length = len(tokens)
        if length > self.config.max_encoder_len:
            raise ValueError(
                f"encoded spec has {length} tokens, exceeding max_encoder_len={self.config.max_encoder_len}"
            )
        padded = np.full(self.config.max_encoder_len, vocab.pad_id, dtype=np.int64)
        padded[:length] = tokens
        return padded, length

    def decode_solution_to_level(self, tokens: list[int] | np.ndarray, spec: LevelSpec, level: Level) -> DecodeReport:
        """
        Parse and apply solution token sequence to resetted `Level`.

        Never raises: grammar violations and simulator placement errors are
        reported as static failure (`ok=False`) with wrong token position.
        Incomplete trailing groups are ignored.
        """
        sol = decode_solution(tokens, spec, self.vocab)
        if isinstance(sol, GrammarError):
            return DecodeReport(ok=False, error=sol.message, error_token_pos=sol.pos)
        return apply_solution(sol, spec, level)
