"""Reward mapping over crafted decode reports and LevelResult values."""

from opus_magnum_sym.levels.result import FailureReason, LevelError, LevelResult, LevelStatus

from opus_magnum_blackbox.codec import DecodeReport
from opus_magnum_blackbox.config import RewardConfig
from opus_magnum_blackbox.reward import compute_reward


def _result(**overrides) -> LevelResult:
    kwargs = {
        "status": LevelStatus.COMPLETED,
        "error": None,
        "steps": 30,
        "finished": {},
        "finished_by_exit": {0: 1},
        "total_finished": 1,
        "target": 1,
        "finished_steps_by_exit": {0: 6},
        "cost": 10,
        "area": 7,
        "solved": True,
    }
    kwargs.update(overrides)
    return LevelResult(**kwargs)


def test_static_failure_decode_error(spec):
    report = DecodeReport(ok=False, error="boom", error_token_pos=3)
    assert compute_reward(report, None, spec, RewardConfig()) == -1.0


def test_static_failure_failed_status(spec):
    report = DecodeReport(ok=True)
    result = _result(
        status=LevelStatus.FAILED,
        error=LevelError("x", "missing", None, FailureReason.ENTRYPOINT_MISSING),
        solved=False,
    )
    assert compute_reward(report, result, spec, RewardConfig()) == -1.0


def test_error_reward_scales_with_crash_step(spec):
    report = DecodeReport(ok=True)
    result = _result(
        status=LevelStatus.ERROR,
        error=LevelError("Collision", "boom", 15, FailureReason.COLLISION_SWEEP_ATOM),
        solved=False,
    )
    assert compute_reward(report, result, spec, RewardConfig()) == -0.5 + 0.4 * (15 / 30)


def test_error_reward_unknown_step(spec):
    result = _result(
        status=LevelStatus.ERROR,
        error=LevelError("Collision", "boom", None, FailureReason.COLLISION_SWEEP_MOVING),
        solved=False,
    )
    assert compute_reward(DecodeReport(ok=True), result, spec, RewardConfig()) == -0.5


def test_unsolved_completed_is_zero_without_shaping(spec):
    result = _result(solved=False, finished_by_exit={0: 0}, total_finished=0, finished_steps_by_exit={0: None})
    assert compute_reward(DecodeReport(ok=True), result, spec, RewardConfig()) == 0.0


def test_unsolved_completed_shaping_proportional_to_delivered(spec):
    result = _result(
        solved=False,
        finished_by_exit={0: 1, 1: 0},
        total_finished=1,
        finished_steps_by_exit={0: 6, 1: None},
    )
    config = RewardConfig(shaping_weight=0.5)
    assert compute_reward(DecodeReport(ok=True), result, spec, config) == 0.5 * 0.5


def test_solved_base_reward(spec):
    assert compute_reward(DecodeReport(ok=True), _result(), spec, RewardConfig()) == 1.0


def test_solved_bonuses_clamped(spec):
    config = RewardConfig(alpha=0.3, beta=0.2, gamma=0.1, cost_max=50)
    reward = compute_reward(DecodeReport(ok=True), _result(), spec, config)
    expected = 1.0 + 0.3 * (1 - 10 / 50) + 0.2 * (1 - 7 / 36) + 0.1 * (1 - 7 / 30)
    assert reward == expected

    # Cost above cost_max clamps the bonus to 0, never negative.
    pricey = _result(cost=500)
    assert compute_reward(DecodeReport(ok=True), pricey, spec, config) < expected
    assert compute_reward(DecodeReport(ok=True), pricey, spec, config) >= 1.0


def test_base_rewards_are_configurable(spec):
    config = RewardConfig(solve_reward=2.0, failure_penalty=-3.0, error_base=-1.0, unsolved_reward=-0.1)
    assert compute_reward(DecodeReport(ok=True), _result(), spec, config) == 2.0
    assert compute_reward(DecodeReport(ok=False), None, spec, config) == -3.0
    assert compute_reward(DecodeReport(ok=True), _result(solved=False, finished_by_exit={0: 0}), spec, config) == -0.1


def test_placement_progress_credited_on_failure(spec):
    config = RewardConfig(placement_reward=0.05)
    # Two placements applied before the third failed.
    report = DecodeReport(ok=False, error="placement failed", error_token_pos=8, placements=2)
    assert compute_reward(report, None, spec, config) == -1.0 + 0.05 * 2
