"""
Rollout collection: masked-sampling episodes and teacher-forced training batches.

Batches already contain old values per step,
so should be ready for PPO.
"""

import pickle
from dataclasses import dataclass

import numpy as np
import torch
from opus_magnum_sym import LevelSpec

from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.env import OpusEnv
from opus_magnum_blackbox.masking import ActionMasker
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


@dataclass
class _Slot:
    """
    One lockstep episode slot: fixed env binding + episode record.
    """

    env: OpusEnv
    obs: dict
    encoder_input: np.ndarray
    encoder_length: int
    tokens: list[int]
    masks: list[np.ndarray]
    logprobs: list[float]
    values: list[float]
    rewards: list[float]

    def reset(self) -> None:
        self.obs, _ = self.env.reset()
        self.encoder_input = self.obs["encoder_input"]
        self.encoder_length = int(self.obs["encoder_length"])
        self.tokens, self.masks, self.logprobs, self.values, self.rewards = [], [], [], [], []

    def episode(self, terminated: bool, info: dict, gamma: float) -> Episode:
        """
        Freeze the record exactly as `collect_episode` would.
        """
        return Episode(
            encoder_input=self.encoder_input,
            encoder_length=self.encoder_length,
            tokens=self.tokens,
            masks=self.masks,
            logprobs=self.logprobs,
            values=self.values,
            rewards=self.rewards,
            returns=discounted_returns(self.rewards, gamma),
            solved=bool(info.get("solved", False)),
            terminated=terminated,
        )


def _shared_encoder_memory(
    policy: OpusPolicy,
    slots: list[_Slot],
    device,
    cache: dict[bytes, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Precomputed encoder outputs reused across token steps.
    Every slot use same spec (matching `encoder_input` bytes);
    None when rows mix levels, in which case caller runs per-row encoder instead.
    Cache lives for one collector call, so entries vanish when their rows leave the batch.
    """
    key = slots[0].encoder_input.tobytes()
    if any(slot.encoder_input.tobytes() != key for slot in slots[1:]):
        return None
    entry = cache.get(key)
    if entry is None:
        encoder_input = torch.from_numpy(slots[0].encoder_input).unsqueeze(0).to(device)
        encoder_length = torch.tensor([slots[0].encoder_length], dtype=torch.long, device=device)
        entry = policy.encode(encoder_input, encoder_length)
        cache[key] = entry
    memory, pad_mask = entry
    return memory.expand(len(slots), -1, -1), pad_mask.expand(len(slots), -1)


def _batched_step(  # noqa: PLR0913, PLR0917
    policy: OpusPolicy,
    staging: torch.Tensor,
    rows: int,
    max_dec: int,
    device,
    memory: torch.Tensor,
    memory_pad_mask: torch.Tensor,
    greedy: bool,
) -> tuple[list[int], list[float], list[float]]:
    """
    One policy decision for `rows` stacked observations
    (staging row layout: decoder_history | decoder_length | action_mask)

    Single host->device transfer and one batched forward over.
    Masked logprobs and state values come back together in one GPU->CPU sync.
    """
    buffer = staging[:rows].to(device, non_blocking=True)
    decoder_history = buffer[:, :max_dec]
    decoder_length = buffer[:, max_dec]
    action_mask = buffer[:, max_dec + 1 :] != 0

    logits, values = policy(None, None, decoder_history, decoder_length, memory, memory_pad_mask)
    index = decoder_length.view(-1, 1, 1)
    step_logits = logits.gather(1, index.expand(-1, 1, logits.shape[2])).squeeze(1)
    step_values = values.gather(1, decoder_length.view(-1, 1)).squeeze(1)

    masked = step_logits.masked_fill(~action_mask, float("-inf"))
    logsumexp = masked.logsumexp(dim=-1)
    probs = masked.softmax(dim=-1)
    action = probs.argmax(dim=-1) if greedy else torch.multinomial(probs, 1).squeeze(1)
    logprob = masked.gather(1, action.unsqueeze(1)).squeeze(1) - logsumexp

    packet = torch.stack((action.float(), logprob, step_values), dim=1).cpu()
    return packet[:, 0].long().tolist(), packet[:, 1].tolist(), packet[:, 2].tolist()


def replicate_envs(envs: list[OpusEnv], n: int, seed: int | None = None) -> list[OpusEnv]:
    """
    `n` independent copies of the env pool.

    Pickle drop shared Level cache, so steps never mutate each other's stepping state.
    With `seed`, replica's RNG is reseeded from deterministic child of that seed:
    unpickled copies would otherwise share seed and use identical level sequences.
    """
    replicas = [pickle.loads(pickle.dumps(envs[i % len(envs)])) for i in range(n)]
    if seed is not None:
        children = np.random.SeedSequence(seed).spawn(n)
        for replica, child in zip(replicas, children, strict=True):
            replica.np_random = np.random.default_rng(child)
    return replicas


def collect_batched(  # noqa: PLR0913, PLR0917
    envs: list[OpusEnv], policy: OpusPolicy, n: int, device, greedy: bool = False, gamma: float = 1.0
) -> RolloutBatch:
    """
    N episodes over `n` lockstep slots,
    slot i bound to `envs[i % len(envs)]`.

    Each token-step issues one batched policy forward across the active slots.

    Finished slot refills only while "demand" is begger then other slots capacity,
    otherwise retires from active set;
    every stepped token belongs to a counted episode, minus the accepted residue of slots
    ordered before the finishing slot in the final token-step.
    Slots never share stepping state:
    pool smaller than `n` is replicated, so pass pre-replicated pools where profiling must
    see the replicas.
    """
    device = torch.device(device)
    if len(envs) < n:
        envs = replicate_envs(envs, n)
    vocab_size = len(envs[0].tokenizer.vocab)
    max_dec = envs[0].tokenizer.config.max_decoder_len
    slots = [
        _Slot(
            env=envs[i % len(envs)],
            obs=None,
            encoder_input=None,
            encoder_length=0,
            tokens=[],
            masks=[],
            logprobs=[],
            values=[],
            rewards=[],
        )
        for i in range(n)
    ]
    for slot in slots:
        slot.reset()

    finished: list[Episode] = []
    active = list(slots)  # active count tracks remaining demand: no discarded replacement episodes

    # Staging row layout: decoder_history | decoder_length | action_mask.
    staging = torch.empty((n, max_dec + 1 + vocab_size), dtype=torch.int64, pin_memory=device.type == "cuda")
    staging_rows = staging.numpy()
    memory_cache: dict[bytes, tuple[torch.Tensor, torch.Tensor]] = {}

    with torch.no_grad():
        while len(finished) < n:
            shared = _shared_encoder_memory(policy, active, device, memory_cache)
            if shared is None:
                encoder_input = torch.from_numpy(np.stack([slot.encoder_input for slot in active])).to(device)
                encoder_length = torch.tensor([slot.encoder_length for slot in active], dtype=torch.long, device=device)
                memory, memory_pad_mask = policy.encode(encoder_input, encoder_length)
            else:
                memory, memory_pad_mask = shared

            for row, slot in enumerate(active):
                obs = slot.obs
                staging_rows[row, :max_dec] = obs["decoder_history"]
                staging_rows[row, max_dec] = obs["decoder_length"]
                staging_rows[row, max_dec + 1 :] = obs["action_mask"]
            actions, logprobs, values = _batched_step(
                policy, staging, len(active), max_dec, device, memory, memory_pad_mask, greedy
            )

            # Snapshot: finishing slots may retire from `active` below.
            for slot, action, logprob, value in list(zip(active, actions, logprobs, values, strict=True)):
                mask = slot.obs["action_mask"].astype(bool)
                slot.obs, reward, terminated, truncated, info = slot.env.step(action)
                slot.tokens.append(action)
                slot.masks.append(mask)
                slot.logprobs.append(logprob)
                slot.values.append(value)
                slot.rewards.append(float(reward))
                if terminated or truncated:
                    finished.append(slot.episode(terminated, info, gamma))
                    if len(finished) == n:
                        break
                    # Refill only while demand exceeds the other slots' capacity; else retire.
                    if n - len(finished) > len(active) - 1:
                        slot.reset()
                    else:
                        active.remove(slot)

    pad_id = envs[0].tokenizer.vocab.pad_id
    return build_batch(finished, pad_id)


def build_batch(episodes: list[Episode], pad_id: int) -> RolloutBatch:
    """
    Stack episodes into teacher-provided (forced) batch, padded to longest episode.
    """
    # fmt: off
    batch_size = len(episodes)
    max_len    = max(len(e.tokens) for e in episodes)
    vocab_size = episodes[0].masks[0].shape[0]

    encoder_input  = torch.stack([torch.from_numpy(e.encoder_input) for e in episodes])
    encoder_length = torch.tensor([e.encoder_length for e in episodes], dtype=torch.long)
    decoder_length = torch.tensor([len(e.tokens) for e in episodes], dtype=torch.long)

    actions        = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    action_masks   = torch.zeros((batch_size, max_len, vocab_size), dtype=torch.bool)
    old_logprobs   = torch.zeros((batch_size, max_len))
    old_values     = torch.zeros((batch_size, max_len))
    returns        = torch.zeros((batch_size, max_len))
    valid          = torch.zeros((batch_size, max_len), dtype=torch.bool)
    # fmt: on

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


def build_supervised(
    sequences: list[list[int]],
    spec: LevelSpec,
    tokenizer: OpusTokenizer,
    masker: ActionMasker,
) -> RolloutBatch:
    """
    Stack static solution sequences into the `RolloutBatch` format.

    All rows share the given spec;
    per-position masks are recomputed over each prefix,
    reward-derived fields are zeroed.
    """
    vocab = tokenizer.vocab
    batch_size = len(sequences)
    max_len = max(len(tokens) for tokens in sequences)

    padded, encoder_length = tokenizer.encode_level_spec(spec)
    encoder_input = torch.from_numpy(padded).unsqueeze(0).repeat(batch_size, 1)
    encoder_length = torch.full((batch_size,), encoder_length, dtype=torch.long)
    decoder_length = torch.tensor([len(tokens) for tokens in sequences], dtype=torch.long)

    actions = torch.full((batch_size, max_len), vocab.pad_id, dtype=torch.long)
    action_masks = torch.zeros((batch_size, max_len, len(vocab)), dtype=torch.bool)
    valid = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, tokens in enumerate(sequences):
        actions[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        for t in range(len(tokens)):
            action_masks[i, t] = torch.from_numpy(masker.mask(tokens[:t], spec))
        valid[i, : len(tokens)] = True

    zeros = torch.zeros(batch_size, max_len)
    return RolloutBatch(
        encoder_input=encoder_input,
        encoder_length=encoder_length,
        actions=actions,
        decoder_length=decoder_length,
        action_masks=action_masks,
        old_logprobs=zeros,
        old_values=zeros,
        returns=zeros,
        valid=valid,
        mean_return=0.0,
        solve_rate=0.0,
        mean_length=sum(len(tokens) for tokens in sequences) / batch_size,
    )
