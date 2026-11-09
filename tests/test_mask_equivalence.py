"""
Stage 2 safety net: the env's incremental mask equals the stateless masker output.

Randomized histories cover valid prefixes, deliberately invalid tokens
(broken-history latch) and DONE-phase continuations; spec variants cover
fixed entrypoints (non-empty initial occupied set) and multiple blueprints.
"""

import pickle

import numpy as np
import pytest
from conftest import known_solution_tokens, make_spec
from opus_magnum_sym import Hex, mkentr, mkexit

from opus_magnum_blackbox.env import LevelEntry, OpusEnv


def make_env(**spec_overrides) -> OpusEnv:
    return OpusEnv([LevelEntry(source=make_spec(**spec_overrides), cost_max=50, name="tier0_test")])


FIXED_ENTRYPOINTS = {
    "entrypoints": (mkentr("entrance_typed", Hex(2, 3), rotation=1), mkexit("exit_typed", Hex(3, 2), rotation=2)),
    "entrypoints_fixed": True,
}


def assert_mask_matches_stateless(env: OpusEnv, obs: dict) -> None:
    incremental = obs["action_mask"].astype(bool)
    stateless = env.masker.mask(env._history, env._spec)
    assert np.array_equal(incremental, stateless)


def drive(env: OpusEnv, rng: np.random.Generator, episodes: int, max_steps: int, invalid_prob: float) -> None:
    """Random walk mixing valid continuations with (mostly invalid) random tokens."""
    vocab_size = len(env.tokenizer.vocab)
    for _ in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        assert_mask_matches_stateless(env, obs)
        for _ in range(max_steps):
            if rng.random() < invalid_prob:
                action = int(rng.integers(vocab_size))
            else:
                valid = np.flatnonzero(obs["action_mask"])
                action = int(valid[rng.integers(len(valid))])
            obs, _, terminated, truncated, _ = env.step(action)
            assert_mask_matches_stateless(env, obs)
            if terminated or truncated:
                break


@pytest.mark.parametrize(
    "spec_overrides",
    [
        {},
        FIXED_ENTRYPOINTS,
        {"allowed_blueprints": frozenset({"arm", "arm_piston"})},
    ],
)
def test_incremental_mask_matches_stateless(spec_overrides):
    env = make_env(**spec_overrides)
    drive(env, np.random.default_rng(0), episodes=3, max_steps=200, invalid_prob=0.3)


def test_incremental_mask_matches_stateless_always_invalid():
    env = make_env()
    drive(env, np.random.default_rng(1), episodes=2, max_steps=60, invalid_prob=1.0)


def test_incremental_mask_matches_stateless_past_done(vocab):
    env = make_env()
    env.reset(seed=0)
    tokens = known_solution_tokens(vocab)
    for token in tokens[:-1]:
        obs, _, terminated, _, _ = env.step(token)
        assert not terminated
        assert_mask_matches_stateless(env, obs)
    obs, _, terminated, _, _ = env.step(tokens[-1])
    assert terminated
    assert_mask_matches_stateless(env, obs)
    for _ in range(5):  # stepping past DONE: PAD-only latch, still stateless-identical
        obs, _, _, _, _ = env.step(vocab.pad_id)
        assert_mask_matches_stateless(env, obs)


def test_pickle_drops_incremental_caches(vocab):
    env = make_env()
    env.reset(seed=0)
    for token in known_solution_tokens(vocab)[:7]:
        env.step(token)
    assert env._mask_state is not None

    replica = pickle.loads(pickle.dumps(env))
    assert replica._mask_state is None
    assert replica._occupied is None
    assert replica._mask is None
    drive(replica, np.random.default_rng(2), episodes=1, max_steps=100, invalid_prob=0.3)
