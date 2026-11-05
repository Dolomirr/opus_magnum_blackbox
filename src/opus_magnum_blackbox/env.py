"""
`OpusEnv`: gymnasium environment over Opus Magnum simulator.

Episode is a single autoregressive token sequence:
layout quadruplets between [START_LAYOUT] and [END_LAYOUT],
program triplets between [START_PROGRAM] and [SUBMIT].

Rewards are terminal-only.

`Level` runtimes built per spec and reused across episodes,
so the env is safe to spawn under in async.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from opus_magnum_sym import Level, LevelSpec

from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.config import RewardConfig, TokenizerConfig
from opus_magnum_blackbox.masking import ActionMasker
from opus_magnum_blackbox.reward import compute_reward


@dataclass(frozen=True)
class LevelEntry:
    source: Path | LevelSpec
    cost_max: int
    name: str
    weight: float = 1.0


class OpusEnv(gym.Env):
    metadata = {  # noqa: RUF012
        "render_modes": [],
    }

    def __init__(
        self,
        levels: list[LevelEntry],
        tokenizer_config: TokenizerConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        if not levels:
            raise ValueError("OpusEnv requires at least one level")
        self.levels = list(levels)
        self.tokenizer = OpusTokenizer(tokenizer_config)
        self.masker = ActionMasker(self.tokenizer.vocab)
        self.reward_config = reward_config or RewardConfig()

        vocab_size = len(self.tokenizer.vocab)
        max_enc = self.tokenizer.config.max_encoder_len
        max_dec = self.tokenizer.config.max_decoder_len
        # fmt: off
        self.observation_space = spaces.Dict({
            "encoder_input"   : spaces.Box(low=0, high=vocab_size - 1, shape=(max_enc,), dtype=np.int64),
            "encoder_length"  : spaces.Discrete(max_enc + 1),
            "decoder_history" : spaces.Box(low=0, high=vocab_size - 1, shape=(max_dec,), dtype=np.int64),
            "decoder_length"  : spaces.Discrete(max_dec + 1),
            "action_mask"     : spaces.MultiBinary(vocab_size),
        })
        self.action_space = spaces.Discrete(vocab_size)

        self._level_cache    : dict[str, Level]  = {}
        self._current        : LevelEntry | None = None
        self._spec           : LevelSpec | None  = None
        self._encoder_input  : np.ndarray | None = None
        self._encoder_length : int               = 0
        self._history        : list[int]         = []
        # fmt: on

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_level_cache"] = {}
        return state

    def _level_for(self, entry: LevelEntry) -> Level:
        level = self._level_cache.get(entry.name)
        if level is None:
            level = Level.from_file(entry.source) if isinstance(entry.source, Path) else Level(entry.source)
            self._level_cache[entry.name] = level
        else:
            level.reset()
        return level

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)
        weights = np.array([entry.weight for entry in self.levels])
        index = self.np_random.choice(len(self.levels), p=weights / weights.sum())
        self._current = self.levels[index]
        level = self._level_for(self._current)
        self._spec = level.spec
        self._encoder_input, self._encoder_length = self.tokenizer.encode_level_spec(self._spec)
        self._history = []
        return self._observation(), {"level": self._current.name}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        token = int(action)
        valid = bool(self.masker.mask(self._history, self._spec)[token])
        self._history.append(token)
        terminated = token == self.tokenizer.vocab.special_id("[SUBMIT]")
        truncated = not terminated and len(self._history) >= self.tokenizer.config.max_decoder_len

        reward = 0.0 if valid else self.reward_config.invalid_action_penalty
        info: dict = {"level": self._current.name}
        if terminated or truncated:
            reward, info = self._finish(info)
            info["level"] = self._current.name
        return self._observation(), reward, terminated, truncated, info

    def _finish(self, info: dict) -> tuple[float, dict]:
        level = self._level_cache[self._current.name]
        report = self.tokenizer.decode_solution_to_level(self._history, self._spec, level)
        result = level.run() if report.ok else None
        # The level is dirty after decoding; guarantee a clean slate for the next episode.
        level.reset()
        reward = compute_reward(
            report,
            result,
            self._spec,
            replace(self.reward_config, cost_max=self._current.cost_max),
        )

        # fmt: off
        if result is not None:
            info.update({
                "status"         : result.status.value,
                "reason"         : result.reason.name if result.reason else None,
                "solved"         : result.solved,
                "metrics"        : result.metrics,
                "crash_step"     : result.error.step  if result.error  else None,
                "total_finished" : result.total_finished,
            })
        else:
            info.update({
                "status": "static_failure",
                "reason": report.error,
                "solved": False,
            })

        info["episode"] = {
            "solved": info["solved"],
            "reward": reward,
            "length": len(self._history),
        }
        # fmt: on

        return reward, info

    def _observation(self) -> dict:
        vocab = self.tokenizer.vocab
        max_dec = self.tokenizer.config.max_decoder_len
        history = np.full(max_dec, vocab.pad_id, dtype=np.int64)
        length = min(len(self._history), max_dec)
        history[:length] = self._history[:length]

        # fmt: off
        return {
            "encoder_input"   : self._encoder_input.copy(),
            "encoder_length"  : self._encoder_length,
            "decoder_history" : history,
            "decoder_length"  : len(self._history),
            "action_mask"     : self.masker.mask(self._history, self._spec).astype(np.int8),
        }
        # fmt: on
