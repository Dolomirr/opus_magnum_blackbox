"""
Terminal reward: pure function of the decode report, `LevelResult` and config.

Each outcome contributes a configurable term (see `config.RewardConfig`):

- static failure (decode error or `status=FAILED`): `failure_penalty`, plus
  `placement_reward` per placement that decoded before the failure
- `status=ERROR`: `error_base` + `error_progress * (crash_step / spec.steps)`,
  `error_base` when the crash step is unknown
- `COMPLETED` but not solved: `unsolved_reward`, plus an optional
  `shaping_weight` term proportional to the delivered fraction
- solved: `solve_reward` plus clamped cost/area/cycles bonuses weighted by
  alpha/beta/gamma
"""

from opus_magnum_sym import LevelSpec
from opus_magnum_sym.levels.result import LevelResult, LevelStatus

from opus_magnum_blackbox.codec import DecodeReport
from opus_magnum_blackbox.config import RewardConfig


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def compute_reward(
    report: DecodeReport,
    result: LevelResult | None,
    spec: LevelSpec,
    config: RewardConfig,
) -> float:
    if not report.ok or result is None or result.status is LevelStatus.FAILED:
        return config.failure_penalty + config.placement_reward * report.placements

    if result.status is LevelStatus.ERROR:
        crash_step = result.error.step if result.error else None
        if crash_step is None:
            return config.error_base
        return config.error_base + config.error_progress * (crash_step / spec.steps)

    if not result.solved:
        reward = config.unsolved_reward
        if config.shaping_weight <= 0.0 or not spec.target:
            return reward
        num_exits = len(result.finished_by_exit)
        required = spec.target * num_exits
        delivered = sum(min(count, spec.target) for count in result.finished_by_exit.values())
        return reward + config.shaping_weight * (delivered / required) if required > 0 else reward

    rows, cols = spec.size
    completion_steps = result.metrics["completion_steps"]
    reward = config.solve_reward
    reward += config.alpha * _clamp01(1.0 - result.cost / config.cost_max)
    reward += config.beta * _clamp01(1.0 - result.area / (rows * cols))
    if completion_steps is not None:
        reward += config.gamma * _clamp01(1.0 - completion_steps / spec.steps)
    return reward
