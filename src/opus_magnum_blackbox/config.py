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


# fmt: off
@dataclass(frozen=True)
class RewardConfig:
    """
    All reward terms, so each curriculum stage can use its own values.

    Stage 0 sets every metric weight to zero (learn to deliver at all);
    later stages raise alpha/beta/gamma to optimize cost/area/cycles.
    """

    # terminal base rewards
    solve_reward           : float = 1.0  # solved episode
    failure_penalty        : float = -1.0 # static failure or FAILED
    error_base             : float = -0.5 # ERROR with unknown crash step
    error_progress         : float = 0.4  # ERROR over crash_step / spec.steps
    unsolved_reward        : float = 0.0  # ran to completion but did not reach target

    # pareto metric bonuses, set per stage
    alpha                  : float = 0.0  # cost
    beta                   : float = 0.0  # area
    gamma                  : float = 0.0  # cycles
    cost_max               : int   = 100  # per-level normalization (overridden by LevelEntry)

    # shaping / progress
    shaping_weight         : float = 0.0  # delivered on unsolved COMPLETED
    placement_reward       : float = 0.0  # per correct placement, credited on failure
    invalid_action_penalty : float = 0.0  # per-step, for a token the mask disallowed
# fmt: on


@dataclass(frozen=True)
class Curriculum:
    """
    Ordered reward configs, one per training stage.
    """

    stages: tuple[RewardConfig, ...] = (RewardConfig(),)
