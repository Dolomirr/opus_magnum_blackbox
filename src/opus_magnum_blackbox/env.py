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
from opus_magnum_sym import ComponentType, Hex, Level, LevelSpec

from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.config import RewardConfig, TokenizerConfig, encoder_feature_dim
from opus_magnum_blackbox.families import MAX_RING, hex_distance
from opus_magnum_blackbox.geometry import footprint, occupied_cells
from opus_magnum_blackbox.grammar import GrammarError, ParseState, advance, spec_model
from opus_magnum_blackbox.masking import ActionMasker
from opus_magnum_blackbox.reward import compute_reward


@dataclass(frozen=True)
class LevelEntry:
    """
    One corpus level: its source plus curriculum/normalization config.
    """

    source: Path | LevelSpec
    cost_max: int
    name: str
    weight: float = 1.0
    solutions: tuple[Path, ...] = ()


def progress_distances(level: Level) -> tuple[float | None, float | None]:
    """
    Entrance->exit hex distance and the smallest remaining-atom distance
    to nearest exit cell, applying progress from `compute_reward`.

    Atoms still held by manipulators also counts.
    Returns `(None, None)` when entrypoint cells are unavailable.
    """
    entrances, exits = _entrypoint_cells(level.sim.entrypoints)
    if not entrances or not exits:
        return None, None
    initial = min(hex_distance(cell, exit_cell) for cell in entrances for exit_cell in exits)
    atoms = list(level.sim.board.atoms)
    for comp in level.sim.components.values():
        holding = getattr(comp, "holding", None)
        if holding:
            atoms.extend(holding)
    final = min((min(hex_distance(cell, exit_cell) for exit_cell in exits) for cell in atoms), default=0)
    return float(initial), float(final)


def _entrypoint_cells(entrypoints) -> tuple[set[Hex], set[Hex]]:
    """
    Entrance and exit footprint cells of a level's fixed entrypoints.
    """
    entrances: set[Hex] = set()
    exits: set[Hex] = set()
    for comp in entrypoints.values():
        cells = set(comp.footprint())
        if comp.type is ComponentType.EXIT:
            exits |= cells
        else:
            entrances |= cells
    return entrances, exits


def spec_encoder_features(spec: LevelSpec, features: tuple[str, ...], entrypoints=None) -> np.ndarray:
    """
    Derived relational encoder features as one float32 vector (canonical order,
    see `ENCODER_FEATURE_NAMES`), a pure function of the spec:

    - `distance`: entrance->exit hex distance normalized by the board diameter.
    - `ring_reach` per ring L: whether some in-board base cell lies exactly L
      rings from both an entrance and an exit cell — i.e. both lie on one
      ring-L static arm's orbit (family geometry: a static arm's hand sweeps
      the hex ring of its radius around its base).

    Missing entrypoints degrade to zeros. `entrypoints` injects an already
    built level's fixed entrypoints (the env's reset path) instead of building
    a fresh `Level(spec)`.
    """
    if entrypoints is None:
        entrypoints = Level(spec).sim.entrypoints
    rows, cols = spec.size
    entrances, exits = _entrypoint_cells(entrypoints)
    values: list[float] = []
    if "distance" in features:
        diameter = rows + cols - 2
        distance = min((hex_distance(cell, exit_cell) for cell in entrances for exit_cell in exits), default=0)
        values.append(distance / diameter if diameter else 0.0)
    if "ring_reach" in features:
        bases = [Hex(r, s) for r in range(rows) for s in range(cols)]
        for ring in range(1, MAX_RING + 1):
            reach = any(
                hex_distance(base, cell) == ring and hex_distance(base, exit_cell) == ring
                for base in bases
                for cell in entrances
                for exit_cell in exits
            )
            values.append(float(reach))
    vector = np.array(values, dtype=np.float32)
    return vector


class OpusEnv(gym.Env):
    metadata = {  # noqa: RUF012
        "render_modes": [],
    }

    def __init__(
        self,
        levels: list[LevelEntry],
        tokenizer_config: TokenizerConfig | None = None,
        reward_config: RewardConfig | None = None,
        encoder_features: tuple[str, ...] = (),
    ) -> None:
        if not levels:
            raise ValueError("OpusEnv requires at least one level")
        self.levels = list(levels)
        self.tokenizer = OpusTokenizer(tokenizer_config)
        self.masker = ActionMasker(self.tokenizer.vocab)
        self.reward_config = reward_config or RewardConfig()
        self.encoder_features = tuple(encoder_features)

        vocab_size = len(self.tokenizer.vocab)
        max_enc = self.tokenizer.config.max_encoder_len
        max_dec = self.tokenizer.config.max_decoder_len
        # fmt: off
        observation = spaces.Dict({
            "encoder_input"   : spaces.Box(low=0, high=vocab_size - 1, shape=(max_enc,), dtype=np.int64),
            "encoder_length"  : spaces.Discrete(max_enc + 1),
            "decoder_history" : spaces.Box(low=0, high=vocab_size - 1, shape=(max_dec,), dtype=np.int64),
            "decoder_length"  : spaces.Discrete(max_dec + 1),
            "action_mask"     : spaces.MultiBinary(vocab_size),
        })
        if self.encoder_features:
            observation["encoder_features"] = spaces.Box(low=0.0, high=1.0, shape=(encoder_feature_dim(self.encoder_features),), dtype=np.float32)
        self.observation_space = observation
        self.action_space = spaces.Discrete(vocab_size)

        self._level_cache    : dict[str, Level]                 = {}
        self._current        : LevelEntry | None                = None
        self._spec           : LevelSpec | None                 = None
        self._encoder_input  : np.ndarray | None                = None
        self._encoder_length : int                              = 0
        self._features       : np.ndarray | None                = None  # derived per-spec side channel, fixed per episode
        self._history        : list[int]                        = []
        self._mask_state     : ParseState | GrammarError | None = None  # running parse, rebuilt on reset
        self._occupied       : set[tuple[int, int]] | None      = None  # fixed entrypoints + placements so far
        self._mask           : np.ndarray | None                = None  # mask for the current history
        # fmt: on

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_level_cache"] = {}
        # replica envs never inherit stepping caches; alwasy reset
        state["_mask_state"] = None
        state["_occupied"] = None
        state["_mask"] = None
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
        self._features = (
            spec_encoder_features(self._spec, self.encoder_features, level.sim.entrypoints)
            if self.encoder_features
            else None
        )
        self._history = []
        self._mask_state = ParseState()
        self._occupied = set(occupied_cells(self._spec, spec_model(self._spec), ()))
        return self._observation(), {"level": self._current.name}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        token = int(action)
        valid = bool(self._mask[token])
        self._history.append(token)
        terminated = token == self.tokenizer.vocab.special_id("[SUBMIT]")
        truncated = not terminated and len(self._history) >= self.tokenizer.config.max_decoder_len

        reward = 0.0 if valid else self.reward_config.invalid_action_penalty
        info: dict = {"level": self._current.name}
        obs = self._observation()
        if not terminated and int(self._mask.sum()) == 0:
            # empty mask is true: no legal token exists, so finish over the normal path
            reward, info = self._finish(info)
            info["level"] = self._current.name
            info["dead_end"] = True
            # dead end never submitted, so it is not a solve
            info["solved"] = False
            info["episode"]["solved"] = False
            terminated = True
            truncated = False
            obs["action_mask"][:] = 0
            obs["action_mask"][self.tokenizer.vocab.pad_id] = 1
        elif terminated or truncated:
            reward, info = self._finish(info)
            info["level"] = self._current.name
        return obs, reward, terminated, truncated, info

    def _finish(self, info: dict) -> tuple[float, dict]:
        level = self._level_cache[self._current.name]
        report = self.tokenizer.decode_solution_to_level(self._history, self._spec, level)
        result = level.run() if report.ok else None
        # atom distances must be read before reset
        initial_dist, final_dist = progress_distances(level)
        # The level is dirty after decoding; guarantee a clean slate for the next episode.
        level.reset()
        reward = compute_reward(
            report,
            result,
            self._spec,
            replace(self.reward_config, cost_max=self._current.cost_max),
            initial_dist=initial_dist,
            final_dist=final_dist,
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
        self._sync_mask_state()
        self._mask = self.masker.mask_state(self._mask_state, self._spec, self._occupied)

        # fmt: off
        observation = {
            "encoder_input"   : self._encoder_input.copy(),
            "encoder_length"  : self._encoder_length,
            "decoder_history" : history,
            "decoder_length"  : len(self._history),
            "action_mask"     : self._mask.astype(np.int8),
        }
        if self._features is not None:
            observation["encoder_features"] = self._features.copy()
        return observation
        # fmt: on

    def _sync_mask_state(self) -> None:
        """
        Feed tokens not yet consumed to the running parse state, one `advance` each.

        `GrammarError` ensure every later mask stays `PAD`-only, until `reset`.
        When a layout group closes, set of occupied cell extends
        by new placement's footprint instead of being rebuilt from all placements.
        """
        if isinstance(self._mask_state, GrammarError):
            return
        state = self._mask_state
        while state.length < len(self._history):
            placements = len(state.placements)
            state = advance(state, self._history[state.length], self._spec, self.masker.vocab)
            if isinstance(state, GrammarError):
                self._mask_state = state
                return
            if len(state.placements) > placements:
                placement = state.placements[-1]
                self._occupied |= footprint(placement.template, placement.rotation, placement.r, placement.q)
            self._mask_state = state
