"""
Configuration for RL translation layer.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VocabConfig:
    """
    Bounds the vocabulary
    """

    # fmt: off
    max_rows             : int = 10
    max_cols             : int = 10
    max_steps            : int = 64
    max_handles          : int = 8
    max_entrypoint_slots : int = 8
    max_shape_cells      : int = 8
    max_cell_offset      : int = 8
    # fmt: on


@dataclass(frozen=True)
class TokenizerConfig:
    """
    Sequence-length caps, derived per curriculum tier.
    """

    # fmt: off
    vocab               : VocabConfig = field(default_factory=VocabConfig)
    max_placements      : int         = 16
    max_program_actions : int         = 32
    max_encoder_len     : int         = 256
    # fmt: on

    @property
    def max_decoder_len(self) -> int:
        # [START_LAYOUT] + quadruplets + [END_LAYOUT] + [START_PROGRAM] + triplets + [SUBMIT]
        return 1 + 4 * self.max_placements + 1 + 1 + 3 * self.max_program_actions + 1


