"""Rollout collection: mask validity, discounted returns, batch alignment, determinism."""

import math

import pytest
import torch
from conftest import known_solution_tokens, make_spec

from opus_magnum_blackbox.config import ModelConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import (
    Episode,
    build_batch,
    build_supervised,
    collect_batch,
    collect_batched,
    collect_episode,
    discounted_returns,
    mixture_distribution,
    replicate_envs,
)

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


def test_batched_episodes_replay_clean(env, policy):
    """
    Lockstep slots never share state: episodes replay valid under the stateless
    masker and end on [SUBMIT] or at the truncation cap.
    """
    batch = collect_batched([env], policy, 4, "cpu")
    spec = env.levels[0].source
    masker = env.masker
    submit_id = env.tokenizer.vocab.special_id("[SUBMIT]")
    max_dec = env.tokenizer.config.max_decoder_len
    for i in range(4):
        length = int(batch.decoder_length[i])
        tokens = batch.actions[i, :length].tolist()
        assert tokens[-1] == submit_id or length == max_dec
        for t, token in enumerate(tokens):
            assert masker.mask(tokens[:t], spec)[token]


def test_replicate_envs_seed_independence_and_reproducibility():
    """
    Pre-fix bug: pickle copies shared one RNG stream, so every replica drew the same
    variant every reset. Seeded replicas must draw independently, reproduce their
    streams for equal seeds, and diverge under a different seed.
    """
    entries = [LevelEntry(source=make_spec(), cost_max=50, name=f"bank/v{i:02d}") for i in range(4)]

    def draw(seed: int) -> list[list[str]]:
        replicas = replicate_envs([OpusEnv(entries)], 6, seed=seed)
        return [[env.reset()[1]["level"] for _ in range(4)] for env in replicas]

    first = draw(7)
    assert draw(7) == first  # same seed reproduces identical draw sequences
    assert len({seq[0] for seq in first}) > 1  # replicas do not draw in lockstep
    assert len({tuple(seq) for seq in first}) > 1  # streams diverge over resets too
    assert draw(8) != first  # different seed diverges


class _CountingEnv(OpusEnv):
    def __init__(self, levels):
        super().__init__(levels)
        self.step_calls = 0

    def step(self, action):
        self.step_calls += 1
        return super().step(action)


def test_batched_collection_discards_no_steps(policy):
    """
    Every env.step belongs to a counted episode. The only accepted residue is
    slots ordered before the finishing slot in the final token-step (one step
    each, bounded by n-1); the Stage 1 refill policy discarded whole episodes.
    """
    n = 4
    envs = []
    for _ in range(n):
        counting = _CountingEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")])
        counting.reset(seed=0)
        envs.append(counting)
    torch.manual_seed(0)
    batch = collect_batched(envs, policy, n, "cpu")

    total_tokens = int(batch.valid.sum())
    total_steps = sum(item.step_calls for item in envs)
    assert batch.actions.shape[0] == n
    assert total_steps - total_tokens < n  # zero discarded episodes, bounded residue only


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


@pytest.mark.parametrize("collector", [collect_batch, collect_batched])
def test_collect_batch_shape_and_stats(env, policy, collector):
    batch = collector([env], policy, 3, "cpu")
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
    first = collect_batched([env], policy, 3, "cpu")
    env.reset(seed=0)
    torch.manual_seed(0)
    second = collect_batched([env], policy, 3, "cpu")
    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.old_logprobs, second.old_logprobs)
    assert torch.equal(first.old_values, second.old_values)
    assert torch.equal(first.returns, second.returns)
    assert first.mean_return == second.mean_return
    assert first.solve_rate == second.solve_rate
    assert first.mean_length == second.mean_length


# --- mask-uniform exploratory sampling (stage 2 of the discovery plan) ---


def test_zero_epsilon_matches_default_rollouts(env, policy):
    """explore_epsilon=0 reproduces today's seeded trajectories exactly, both collectors."""
    env.reset(seed=0)
    torch.manual_seed(0)
    default_batch = collect_batched([env], policy, 3, "cpu")
    env.reset(seed=0)
    torch.manual_seed(0)
    zero_batch = collect_batched([env], policy, 3, "cpu", explore_epsilon=0.0)
    assert torch.equal(default_batch.actions, zero_batch.actions)
    assert torch.equal(default_batch.old_logprobs, zero_batch.old_logprobs)

    env.reset(seed=0)
    torch.manual_seed(0)
    default_episode = collect_episode(env, policy, "cpu")
    env.reset(seed=0)
    torch.manual_seed(0)
    zero_episode = collect_episode(env, policy, "cpu", explore_epsilon=0.0)
    assert default_episode.tokens == zero_episode.tokens
    assert default_episode.logprobs == zero_episode.logprobs


def test_full_epsilon_single_env_uniform_valid(env, policy):
    """explore_epsilon=1: every token is mask-valid and scored log(u) = -log(num_valid)."""
    env.reset(seed=0)
    torch.manual_seed(1)
    episode = collect_episode(env, policy, "cpu", explore_epsilon=1.0)
    assert len(episode.tokens) > 0
    for token, mask, logprob in zip(episode.tokens, episode.masks, episode.logprobs, strict=True):
        assert mask[token]
        assert logprob == pytest.approx(-math.log(int(mask.sum())), abs=1e-5)


def test_full_epsilon_batched_uniform_valid(env, policy):
    """Lockstep collector at epsilon=1: mask-valid tokens scored under the uniform-valid logprob."""
    env.reset(seed=0)
    torch.manual_seed(1)
    batch = collect_batched([env], policy, 3, "cpu", explore_epsilon=1.0)
    for i in range(3):
        length = int(batch.decoder_length[i])
        for t in range(length):
            mask = batch.action_masks[i, t]
            action = int(batch.actions[i, t])
            assert mask[action]
            assert batch.old_logprobs[i, t] == pytest.approx(-math.log(int(mask.sum())))


def test_mixture_distribution_hand_computed():
    """Masked softmax [1, 3, 0, 1]/5 mixed with eps=0.3 uniform over 3 valid tokens."""
    logits = torch.tensor([[0.0, math.log(3.0), float("-inf"), 0.0]])
    mask = torch.tensor([[True, True, False, True]])
    probs = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    mixture = mixture_distribution(probs, mask, 0.3)
    expected = torch.tensor([[0.7 * 0.2 + 0.1, 0.7 * 0.6 + 0.1, 0.0, 0.7 * 0.2 + 0.1]])
    assert torch.allclose(mixture.probs, expected)
    assert mixture.log_prob(torch.tensor(1)) == pytest.approx(math.log(0.52))
    assert mixture.probs[0, 2].item() == 0.0  # masked token carries no mixture mass


def test_partial_epsilon_records_component_logprobs(env, policy):
    """
    With 0 < eps < 1 each recorded logprob belongs to the distribution its token was
    drawn from: the policy (exploit coin) or the (1-eps) p + eps u mixture (explore coin).
    Both branches must occur over a seeded run.
    """
    epsilon = 0.5
    env.reset(seed=0)
    torch.manual_seed(3)
    episode = collect_episode(env, policy, "cpu", explore_epsilon=epsilon)
    length = len(episode.tokens)

    with torch.no_grad():
        padded = torch.zeros(1, env.tokenizer.config.max_decoder_len, dtype=torch.long)
        padded[0, :length] = torch.tensor(episode.tokens)
        logits, _ = policy(
            torch.from_numpy(episode.encoder_input).unsqueeze(0),
            torch.tensor([episode.encoder_length]),
            padded,
            torch.tensor([length]),
        )

    explored = exploited = 0
    for t, (token, mask, logprob) in enumerate(zip(episode.tokens, episode.masks, episode.logprobs, strict=True)):
        mask_tensor = torch.from_numpy(mask)
        dist = policy.masked_distribution(logits[0, t], mask_tensor)
        policy_logprob = float(dist.log_prob(torch.tensor(token)))
        mixture_logprob = float(mixture_distribution(dist.probs, mask_tensor, epsilon).log_prob(torch.tensor(token)))
        if math.isclose(logprob, policy_logprob, rel_tol=1e-5):
            exploited += 1
        elif math.isclose(logprob, mixture_logprob, rel_tol=1e-5):
            explored += 1
        else:
            pytest.fail(f"step {t}: logprob {logprob} matches neither policy nor mixture")
    assert explored > 0
    assert exploited > 0


def test_greedy_ignores_epsilon(env, policy):
    """Greedy rollouts (train-bank eval, CLI eval) must be identical under any epsilon."""
    env.reset(seed=0)
    torch.manual_seed(4)
    plain_batch = collect_batched([env], policy, 2, "cpu", greedy=True)
    env.reset(seed=0)
    torch.manual_seed(4)
    explored_batch = collect_batched([env], policy, 2, "cpu", greedy=True, explore_epsilon=1.0)
    assert torch.equal(plain_batch.actions, explored_batch.actions)
    assert torch.equal(plain_batch.old_logprobs, explored_batch.old_logprobs)

    env.reset(seed=0)
    plain_episode = collect_episode(env, policy, "cpu", greedy=True)
    env.reset(seed=0)
    explored_episode = collect_episode(env, policy, "cpu", greedy=True, explore_epsilon=1.0)
    assert plain_episode.tokens == explored_episode.tokens
    assert plain_episode.logprobs == explored_episode.logprobs


# --- per-episode solve flags and level names (stage 3 of the discovery plan) ---


def test_batch_solved_flags_and_levels_align(env, policy, tokenizer, masker):
    """episode_solved mirrors Episode.solved, level_names mirror Episode.level, per row."""
    spec = env.levels[0].source
    vocab = env.tokenizer.vocab
    padded, encoder_length = tokenizer.encode_level_spec(spec)
    good = known_solution_tokens(vocab)

    def episode(tokens, solved, level):
        return Episode(
            encoder_input=padded,
            encoder_length=encoder_length,
            tokens=tokens,
            masks=[masker.mask(tokens[:t], spec) for t in range(len(tokens))],
            logprobs=[0.0] * len(tokens),
            values=[0.0] * len(tokens),
            rewards=[0.0] * len(tokens),
            returns=[0.0] * len(tokens),
            solved=solved,
            terminated=True,
            level=level,
        )

    episodes = [episode(good, True, "lvl/a"), episode([vocab.special_id("[SUBMIT]")], False, "lvl/b")]
    batch = build_batch(episodes, vocab.pad_id)
    assert batch.episode_solved.tolist() == [True, False]
    assert batch.level_names == ["lvl/a", "lvl/b"]
    assert batch.mean_return == pytest.approx(episodes[0].returns[0])

    supervised = build_supervised([good], spec, tokenizer, masker)
    assert supervised.episode_solved.tolist() == [False]
    assert supervised.level_names == [""]

    assert collect_episode(env, policy, "cpu").level == "tier0_test"
