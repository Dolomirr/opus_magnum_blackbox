"""OpusEnv: gymnasium compliance, determinism, truncation, full solve."""

from conftest import known_solution_tokens, make_spec
from gymnasium.utils.env_checker import check_env

from opus_magnum_blackbox.env import LevelEntry, OpusEnv


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


def test_level_reuse_across_episodes(vocab):
    env = make_env()
    for _ in range(2):
        env.reset(seed=0)
        for token in known_solution_tokens(vocab):
            _, reward, _terminated, _, info = env.step(token)
        assert reward == 1.0
        assert info["solved"]
