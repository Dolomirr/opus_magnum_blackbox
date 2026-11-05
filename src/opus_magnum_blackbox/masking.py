"""
`ActionMasker`: checks action masks over the vocabulary.

State is recomputed from decoder token history on every call
via grammar parser, so masker is stateless;
a history that deviates from grammar yields a PAD-only mask.
"""

import numpy as np
from opus_magnum_sym import LevelSpec

from opus_magnum_blackbox import geometry
from opus_magnum_blackbox.grammar import GrammarError, ParseState, Phase, SpecModel, parse, spec_model
from opus_magnum_blackbox.vocab import Vocab


class ActionMasker:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def mask(self, history: list[int], spec: LevelSpec) -> np.ndarray:
        """
        Bool-mask over the vocabulary for the next token.
        """
        valid: np.ndarray = np.zeros(len(self.vocab), dtype=bool)
        state = parse(history, spec, self.vocab)
        if isinstance(state, GrammarError) or state.phase is Phase.DONE:
            valid[self.vocab.pad_id] = True
            return valid

        vocab = self.vocab
        if state.phase is Phase.START:
            valid[vocab.special_id("[START_LAYOUT]")] = True
        elif state.phase is Phase.LAYOUT:
            self._layout_mask(valid, state, spec, spec_model(spec))
        elif state.phase is Phase.LAYOUT_DONE:
            valid[vocab.special_id("[START_PROGRAM]")] = True
        else:
            self._program_mask(valid, state, spec)
        return valid

    def _layout_mask(self, valid: np.ndarray, state: ParseState, spec: LevelSpec, model: SpecModel) -> None:
        vocab = self.vocab
        rows, cols = spec.size
        group_pos = len(state.pending)

        if group_pos == 0:
            for name in model.builtins:
                valid[vocab.builtin_id(name)] = True
            for slot, required in enumerate(model.required):
                if state.slot_use.get(slot, 0) < required:
                    valid[vocab.base("entry_slot") + slot] = True
            if all(state.slot_use.get(slot, 0) >= required for slot, required in enumerate(model.required)):
                valid[vocab.special_id("[END_LAYOUT]")] = True
            return

        name_tok = state.pending[0]
        if vocab.is_in(name_tok, "entry_slot"):
            template = model.templates[name_tok - vocab.base("entry_slot")]
        else:
            template = vocab.builtin(name_tok).template
        if group_pos == 1:
            for rotation in geometry.rotations(template):
                valid[vocab.base("rotation") + rotation] = True
        elif group_pos == 2:
            rotation = state.pending[1] - vocab.base("rotation")
            occupied = geometry.occupied_cells(spec, model, state.placements)
            for q in range(cols):
                if any(geometry.placement_valid(template, rotation, r, q, rows, cols, occupied) for r in range(rows)):
                    valid[vocab.base("q") + q] = True
        else:
            rotation = state.pending[1] - vocab.base("rotation")
            q = state.pending[2] - vocab.base("q")
            occupied = geometry.occupied_cells(spec, model, state.placements)
            for r in range(rows):
                if geometry.placement_valid(template, rotation, r, q, rows, cols, occupied):
                    valid[vocab.base("r") + r] = True

    def _program_mask(self, valid: np.ndarray, state: ParseState, spec: LevelSpec) -> None:
        vocab = self.vocab
        group_pos = len(state.pending)

        if group_pos == 0:
            for handle in range(state.current_handle, len(state.arms)):
                valid[vocab.base("handle") + handle] = True
            valid[vocab.special_id("[SUBMIT]")] = True
        elif group_pos == 1:
            handle = state.pending[0] - vocab.base("handle")
            used = state.used_ticks[handle] if handle < len(state.used_ticks) else set()
            for tick in range(spec.steps):
                if tick not in used:
                    valid[vocab.base("tick") + tick] = True
        else:
            handle = state.pending[0] - vocab.base("handle")
            arm = state.arms[handle] if handle < len(state.arms) else None
            for action in vocab.base_actions:
                valid[action] = True
            if arm is not None and arm.extendable:
                for action in vocab.piston_actions:
                    valid[action] = True
