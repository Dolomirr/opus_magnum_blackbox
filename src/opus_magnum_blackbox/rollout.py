"""
Rollout collection: masked-sampling episodes and teacher-forced training batches.

Batchues already contain old values per step,
so shoul be ready for PPO.
"""

from dataclasses import dataclass

import numpy as np
import torch

from opus_magnum_blackbox.env import OpusEnv
from opus_magnum_blackbox.model import OpusPolicy


@dataclass
class Episode:
    """
    One collected episode: token stream + per-step rollout data.
    """

    # fmt: off
    encoder_input  : np.ndarray       # padded spec ids of level this episode played
    encoder_length : int
    tokens         : list[int]        # emitted token ids (decoder history)
    masks          : list[np.ndarray] # per step, [vocab] mask that decided current choice
    logprobs       : list[float]      # old logprob of each token under policy
    values         : list[float]      # old value estimate of each state
    rewards        : list[float]
    returns        : list[float]      # G_t, from rewards and gamma
    solved         : bool
    terminated     : bool             # False for truncated episodes
    # fmt: on


@dataclass(frozen=True)
class RolloutBatch:
    """
    Teacher-forced tensors over a set of episodes, plus episode statistics.
    """

    # fmt: off
    encoder_input  : torch.Tensor # [B, E] int64
    encoder_length : torch.Tensor # [B] int64
    actions        : torch.Tensor # [B, T] int64 targets; the policy BOS-shifts internally
    decoder_length : torch.Tensor # [B] int64, per-episode token counts
    action_masks   : torch.Tensor # [B, T, V] bool, per-step masks
    old_logprobs   : torch.Tensor # [B, T] float
    old_values     : torch.Tensor # [B, T] float
    returns        : torch.Tensor # [B, T] float
    valid          : torch.Tensor # [B, T] bool, positions below decoder_length

    mean_return    : float
    solve_rate     : float
    mean_length    : float
    # fmt: on


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """
    G_t = r_t + gamma * G_{t+1}, accumulated backwards from the terminal reward.
    """
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def _forward_observation(policy: OpusPolicy, obs: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    decoder_length = int(obs["decoder_length"])
    encoder_input = torch.from_numpy(obs["encoder_input"]).unsqueeze(0).to(device)
    encoder_length = torch.tensor([obs["encoder_length"]], dtype=torch.long, device=device)
    decoder_history = torch.from_numpy(obs["decoder_history"]).unsqueeze(0).to(device)
    decoder_length_tensor = torch.tensor([decoder_length], dtype=torch.long, device=device)
    logits, values = policy(encoder_input, encoder_length, decoder_history, decoder_length_tensor)
    return logits[0, decoder_length], values[0, decoder_length]


def collect_episode(env: OpusEnv, policy: OpusPolicy, device, greedy: bool = False, gamma: float = 1.0) -> Episode:
    """
    Roll-out one episode for masked policy (sampled, or argmax when greedy).
    """
    obs, _ = env.reset()
    encoder_input = obs["encoder_input"]
    encoder_length = int(obs["encoder_length"])
    tokens, masks, logprobs, values, rewards = [], [], [], [], []
    solved = terminated = False

    with torch.no_grad():
        while True:
            mask = obs["action_mask"].astype(bool)
            logits, value = _forward_observation(policy, obs, device)
            dist = policy.masked_distribution(logits, torch.from_numpy(mask).to(device))
            action = int(dist.probs.argmax()) if greedy else int(dist.sample())
            logprob = float(dist.log_prob(torch.tensor(action, device=device)))

            obs, reward, terminated, truncated, info = env.step(action)
            tokens.append(action)
            masks.append(mask)
            logprobs.append(logprob)
            values.append(float(value))
            rewards.append(float(reward))
            if terminated or truncated:
                solved = bool(info.get("solved", False))
                break

    return Episode(
        encoder_input=encoder_input,
        encoder_length=encoder_length,
        tokens=tokens,
        masks=masks,
        logprobs=logprobs,
        values=values,
        rewards=rewards,
        returns=discounted_returns(rewards, gamma),
        solved=solved,
        terminated=terminated,
    )


def collect_batch(envs: list[OpusEnv], policy: OpusPolicy, n: int, device, gamma: float = 1.0) -> RolloutBatch:
    """
    N episodes spread over given env instances (round-robin), stacked into one batch.
    """
    episodes = [collect_episode(envs[i % len(envs)], policy, device, gamma=gamma) for i in range(n)]
    pad_id = envs[0].tokenizer.vocab.pad_id
    return build_batch(episodes, pad_id)


def build_batch(episodes: list[Episode], pad_id: int) -> RolloutBatch:
    """
    Stack episodes into teacher-provided (forced) batch, padded to longest episode.
    """
    batch_size = len(episodes)
    max_len = max(len(e.tokens) for e in episodes)
    vocab_size = episodes[0].masks[0].shape[0]

    encoder_input = torch.stack([torch.from_numpy(e.encoder_input) for e in episodes])
    encoder_length = torch.tensor([e.encoder_length for e in episodes], dtype=torch.long)
    decoder_length = torch.tensor([len(e.tokens) for e in episodes], dtype=torch.long)

    actions = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    action_masks = torch.zeros((batch_size, max_len, vocab_size), dtype=torch.bool)
    old_logprobs = torch.zeros((batch_size, max_len))
    old_values = torch.zeros((batch_size, max_len))
    returns = torch.zeros((batch_size, max_len))
    valid = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, episode in enumerate(episodes):
        length = len(episode.tokens)
        actions[i, :length] = torch.tensor(episode.tokens, dtype=torch.long)
        action_masks[i, :length] = torch.from_numpy(np.stack(episode.masks))
        old_logprobs[i, :length] = torch.tensor(episode.logprobs)
        old_values[i, :length] = torch.tensor(episode.values)
        returns[i, :length] = torch.tensor(episode.returns)
        valid[i, :length] = True

    return RolloutBatch(
        encoder_input=encoder_input,
        encoder_length=encoder_length,
        actions=actions,
        decoder_length=decoder_length,
        action_masks=action_masks,
        old_logprobs=old_logprobs,
        old_values=old_values,
        returns=returns,
        valid=valid,
        mean_return=sum(e.returns[0] for e in episodes) / batch_size,
        solve_rate=sum(e.solved for e in episodes) / batch_size,
        mean_length=sum(len(e.tokens) for e in episodes) / batch_size,
    )
