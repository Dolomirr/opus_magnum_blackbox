"""OpusEnv: gymnasium compliance, determinism, truncation, full solve."""

import numpy as np
import torch
from conftest import known_solution_tokens, make_spec
from gymnasium.utils.env_checker import check_env
from opus_magnum_sym import Hex, LevelSpec, StepActionType

from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.config import RewardConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.families import arm_plan, hex_distance, reference_tokens, single_atom_spec
from opus_magnum_blackbox.grammar import Action, Placement, Solution


def make_env(**kwargs) -> OpusEnv:
    return OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")], **kwargs)


def test_check_env_compliance():
    check_env(make_env(), skip_render_check=True)


def test_full_solve_episode(vocab):
    env = make_env()
    env.reset(seed=0)
    tokens = known_solution_tokens(vocab)
    for token in tokens[:-1]:
        _, reward, terminated, truncated, _ = env.step(token)
        assert reward == 0.0
        assert not terminated
        assert not truncated
    _, reward, terminated, truncated, info = env.step(tokens[-1])
    assert terminated
    assert not truncated
    assert reward == 1.0
    assert info["solved"]
    assert info["status"] == "completed"
    assert info["episode"]["solved"]


def test_determinism_same_tokens_same_reward(vocab):
    rewards = []
    for _ in range(2):
        env = make_env()
        env.reset(seed=0)
        reward = 0.0
        for token in known_solution_tokens(vocab):
            _, reward, terminated, _, _ = env.step(token)
        assert terminated
        rewards.append(reward)
    assert rewards[0] == rewards[1] == 1.0


def test_truncation_at_length_cap(vocab):
    env = make_env()
    env.reset(seed=0)
    cap = env.tokenizer.config.max_decoder_len
    pad = vocab.pad_id
    truncated = False
    reward = 0.0
    info = {}
    for _ in range(cap):
        _, reward, terminated, truncated, info = env.step(pad)
        if terminated or truncated:
            break
    assert truncated
    assert not terminated
    assert reward == -1.0  # garbage sequence decodes as a static failure
    assert info["status"] == "static_failure"
    assert not info["episode"]["solved"]


def test_static_failure_episode(vocab):
    env = make_env()
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step(vocab.special_id("[SUBMIT]"))
    assert terminated
    assert reward == -1.0
    assert info["status"] == "static_failure"


# --- grammar dead-end termination (stage 2.5 of the hardening plan) ---


def test_dead_end_terminates_as_failure():
    """
    Greedy-first arm spam on the unfixed-entrypoint spec fills the board mid-layout
    before [END_LAYOUT] becomes legal: the env terminates the episode itself, before
    the length cap, with the failure reward and a PAD-only mask (147-token repro).
    """
    env = make_env()
    obs, _ = env.reset(seed=0)
    steps = 0
    while True:
        assert int(obs["action_mask"].sum()) >= 1  # no collector ever sees an empty mask row
        action = int(np.argmax(obs["action_mask"]))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    assert info.get("dead_end") is True
    assert terminated  # dead end, not length truncation
    assert not truncated
    assert steps < env.tokenizer.config.max_decoder_len
    assert reward == -1.0  # failure penalty via the standard finish path
    assert info["solved"] is False  # a dead end never submitted: not a solve
    assert info["episode"]["solved"] is False
    assert obs["action_mask"].sum() == 1  # PAD-only, like DONE/error paths
    assert obs["action_mask"][env.tokenizer.vocab.pad_id] == 1


def test_masks_never_empty_across_episodes():
    """
    Mask invariant on fixed and unfixed specs: random-ish policies step full episodes
    and every returned observation keeps >= 1 valid token.
    """
    levels = [
        LevelEntry(source=make_spec(), cost_max=50, name="tier0_test"),
        LevelEntry(source=single_atom_spec(Hex(2, 3), Hex(2, 2)), cost_max=100, name="tier1_fixed"),
    ]
    torch.manual_seed(3)
    for entry in levels:
        env = OpusEnv([entry])
        for episode in range(3):
            obs, _ = env.reset(seed=episode)
            while True:
                assert int(obs["action_mask"].sum()) >= 1
                valid = np.flatnonzero(obs["action_mask"])
                action = int(valid[torch.randint(0, len(valid), (1,)).item()])
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            assert int(obs["action_mask"].sum()) >= 1  # terminal observation is PAD-only, never empty


def test_level_reuse_across_episodes(vocab):
    env = make_env()
    for _ in range(2):
        env.reset(seed=0)
        for token in known_solution_tokens(vocab):
            _, reward, _terminated, _, info = env.step(token)
        assert reward == 1.0
        assert info["solved"]


# --- progress shaping threading (stage 1 of the discovery plan) ---


# fixed-entrypoint d2 level: entrance (2,3), exit (2,1), hex distance 2
def make_family_env(**kwargs) -> tuple[OpusEnv, LevelSpec]:
    spec = single_atom_spec(Hex(2, 3), Hex(2, 1))
    env = OpusEnv([LevelEntry(source=spec, cost_max=100, name="tier1_d2_test")], **kwargs)
    return env, spec


def arm_solution_tokens(spec, vocab, base, rotation, actions) -> list[int]:
    tokenizer = OpusTokenizer()
    arm = vocab.builtin(vocab.builtin_id("arm"))
    placement = Placement(builtin=arm, slot=None, template=arm.template, rotation=rotation, q=base.s, r=base.r, pos=0)
    acts = [Action(handle=0, tick=tick, action=action, pos=0) for tick, action in enumerate(actions, 1)]
    sol = Solution(placements=[placement], actions=acts, submitted=True, length=0)
    return tokenizer.encode_solution(sol.placements, sol.actions, spec)


def run_tokens(env: OpusEnv, tokens: list[int]) -> tuple[float, dict]:
    env.reset(seed=0)
    reward, info = 0.0, {}
    terminated = False
    for token in tokens:
        _, reward, terminated, _, info = env.step(token)
    assert terminated
    return reward, info


def test_progress_shaping_partial_credit_for_released_near_miss(vocab):
    # One turn short on the swing: the atom rests 1 cell from the exit -> half of the span.
    env, spec = make_family_env(reward_config=RewardConfig(progress_weight=0.3))
    plan = arm_plan(spec)
    swing = plan.program[1]
    turns = plan.program[1 : plan.program.index(StepActionType.RELEASE)].count(swing)
    tokens = arm_solution_tokens(
        spec, vocab, plan.base, plan.rotation, (StepActionType.GRAB, *[swing] * (turns - 1), StepActionType.RELEASE)
    )
    reward, info = run_tokens(env, tokens)
    assert info["status"] == "completed"
    assert not info["solved"]
    initial = hex_distance(spec.entrypoints[0].pos, spec.entrypoints[1].pos)
    assert initial == 2
    assert reward == 0.3 * (initial - 1) / initial


def test_progress_shaping_counts_held_atoms(vocab):
    # Grab and swing short but never release: the atom stays held, still worth partial credit.
    env, spec = make_family_env(reward_config=RewardConfig(progress_weight=0.3))
    plan = arm_plan(spec)
    swing = plan.program[1]
    turns = plan.program[1 : plan.program.index(StepActionType.RELEASE)].count(swing)
    tokens = arm_solution_tokens(spec, vocab, plan.base, plan.rotation, (StepActionType.GRAB, *[swing] * (turns - 1)))
    reward, info = run_tokens(env, tokens)
    assert info["status"] == "completed"
    assert not info["solved"]
    initial = hex_distance(spec.entrypoints[0].pos, spec.entrypoints[1].pos)
    assert reward == 0.3 * (initial - 1) / initial


def test_progress_shaping_zero_weight_and_solved_unchanged(vocab):
    # Default config keeps the near-miss at 0.0, and a solved run gains no spurious bonus:
    # the entrance keeps an atom at the entrance cell, so progress fraction is 0.
    env, spec = make_family_env()
    tokens = reference_tokens(spec)
    reward, info = run_tokens(env, tokens)
    assert info["solved"]
    assert reward == 1.0

    env, _ = make_family_env(reward_config=RewardConfig(progress_weight=0.3))
    reward, info = run_tokens(env, tokens)
    assert info["solved"]
    assert reward == 1.0


def test_progress_shaping_no_bonus_on_crash(vocab):
    # Two near-miss cycles crash on the second release; crashes never earn progress.
    env, spec = make_family_env(reward_config=RewardConfig(progress_weight=0.3))
    plan = arm_plan(spec)
    swing = plan.program[1]
    turns = plan.program[1 : plan.program.index(StepActionType.RELEASE)].count(swing)
    cycle = (StepActionType.GRAB, *[swing] * (turns - 1), StepActionType.RELEASE, *[swing] * (6 - (turns - 1)))
    tokens = arm_solution_tokens(spec, vocab, plan.base, plan.rotation, cycle * 2)
    reward, info = run_tokens(env, tokens)
    assert info["status"] == "error"
    assert reward == RewardConfig().error_base + RewardConfig().error_progress * (info["crash_step"] / spec.steps)
