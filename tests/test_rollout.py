"""Rollout collection: mask validity, discounted returns, batch alignment, determinism."""

import pytest
import torch
from conftest import make_spec

from opus_magnum_blackbox.config import ModelConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import build_batch, collect_batch, collect_episode, discounted_returns

TINY = ModelConfig(d_model=32, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64)


def make_env() -> OpusEnv:
    return OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")])


@pytest.fixture
def env() -> OpusEnv:
    return make_env()


@pytest.fixture
def policy(tokenizer) -> OpusPolicy:
    torch.manual_seed(0)
    policy = OpusPolicy(tokenizer.config, TINY)
    return policy.eval()


def test_discounted_returns_hand_computed():
    rewards = [1.0, -1.0, 3.0]
    assert discounted_returns(rewards, gamma=1.0) == [3.0, 2.0, 3.0]
    assert discounted_returns(rewards, gamma=0.5) == [1.25, 0.5, 3.0]


def test_collected_actions_are_mask_valid(env, policy):
    for greedy in (False, True):
        episode = collect_episode(env, policy, "cpu", greedy=greedy)
        assert len(episode.tokens) > 0
        for token, mask in zip(episode.tokens, episode.masks, strict=True):
            assert mask[token]


def test_batch_alignment_with_policy_forward(env, policy):
    """Teacher-forced forward on the batch reproduces the per-step rollout data."""
    episodes = [collect_episode(env, policy, "cpu") for _ in range(2)]
    batch = build_batch(episodes, pad_id=0)

    lengths = torch.tensor([len(e.tokens) for e in episodes])
    assert (batch.decoder_length == lengths).all()
    positions = torch.arange(batch.actions.shape[1])
    assert (batch.valid == (positions.unsqueeze(0) < lengths.unsqueeze(1))).all()
    for i, episode in enumerate(episodes):
        length = lengths[i]
        assert (batch.actions[i, :length] == torch.tensor(episode.tokens)).all()

    with torch.no_grad():
        logits, values = policy.forward(
            batch.encoder_input,
            batch.encoder_length,
            batch.actions,
            batch.decoder_length,
        )
    for i, episode in enumerate(episodes):
        length = lengths[i]
        # logits are BOS-shifted: output position t holds the distribution for target token t
        dist = policy.masked_distribution(logits[i, :length], batch.action_masks[i, :length])
        assert torch.allclose(dist.log_prob(batch.actions[i, :length]), torch.tensor(episode.logprobs), atol=1e-5)
        assert torch.allclose(values[i, :length], torch.tensor(episode.values), atol=1e-5)


def test_collect_batch_shape_and_stats(env, policy):
    batch = collect_batch([env], policy, 3, "cpu")
    assert batch.actions.shape[0] == 3
    assert batch.mean_length == pytest.approx(float(batch.decoder_length.float().mean()))
    assert 0.0 <= batch.solve_rate <= 1.0
    assert batch.mean_return == pytest.approx(batch.returns[:, 0].mean().item())

    for i in range(3):
        length = int(batch.decoder_length[i])
        assert batch.action_masks[i, :length].gather(-1, batch.actions[i, :length].unsqueeze(-1)).all()


def test_seeded_determinism(env, policy):
    env.reset(seed=0)
    torch.manual_seed(0)
    first = collect_episode(env, policy, "cpu")
    torch.manual_seed(0)
    second = collect_episode(env, policy, "cpu")
    assert first.tokens == second.tokens
    assert first.logprobs == second.logprobs
    assert first.rewards == second.rewards
