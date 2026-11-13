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


@dataclass(frozen=True)
class ModelConfig:
    """
    Transformer value sizes.
    """

    # fmt: off
    d_model         : int   = 128
    nhead           : int   = 4
    encoder_layers  : int   = 2
    decoder_layers  : int   = 2
    dim_feedforward : int   = 512
    dropout         : float = 0.0
    # fmt: on


@dataclass(frozen=True)
class TrainConfig:
    """
    Masked REINFORCE hyperparameters;

    Overfit-run defaults.

    `device` "" auto-resolves to cuda when available,
    else cpu (kept as a string so this config stays importable without torch).
    """

    # fmt: off
    iterations             : int   = 2000
    episodes_per_iteration : int   = 32
    lr                     : float = 3e-4
    entropy_coef           : float = 0.01
    explore_epsilon        : float = 0.0       # per-step mask-uniform exploration on sampled rollouts
    sample_temperature     : float = 1.0       # logits / T before masked softmax on sampled rollouts
    sil_coef               : float = 0.0       # self-imitation BC weight over buffered solved episodes
    sil_batch_size         : int   = 8         # rows per SIL minibatch
    sil_buffer_cap         : int   = 256       # buffered solved episodes, deduped per level (latest kept)
    sil_weighting          : str   = "uniform" # SIL minibatch draw: "uniform" or "scarcity" (replay ∝ 1/solve count)
    value_coef             : float = 0.5
    gamma                  : float = 1.0
    grad_clip              : float = 1.0
    eval_interval          : int   = 10
    eval_episodes          : int   = 8
    checkpoint_interval    : int   = 50
    seed                   : int   = 0
    device                 : str   = ""        # auto-resolves to cuda then avaliable
    run_dir                : str   = "runs"
    # fmt: on


@dataclass(frozen=True)
class PretrainConfig:
    """
    Behavior-cloning (bootstrapping solution) stage hyperparameters.
    """

    # fmt: off
    epochs                : int   = 300
    batch_size            : int   = 32
    lr                    : float = 1e-3
    augment               : bool  = True
    max_variants          : int   = 16
    eval_interval         : int   = 10
    early_stop_solve_rate : float = 1.0
    early_stop_loss       : float = 1e-3
    # fmt: on


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
    progress_weight        : float = 0.0  # atom progress toward the exit, on COMPLETED
    placement_reward       : float = 0.0  # per correct placement, credited on failure
    invalid_action_penalty : float = 0.0  # per-step, for a token the mask disallowed
# fmt: on


@dataclass(frozen=True)
class Curriculum:
    """
    Ordered reward configs, one per training stage.
    """

    stages: tuple[RewardConfig, ...] = (RewardConfig(),)
