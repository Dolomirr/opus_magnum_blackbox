"""
`Trainer`: masked REINFORCE with learned value baseline, in one visible loop.

`update` is PPO: replacing its body (ratio clip, GAE, epochs per batch)
upgrades algorithm; batch format and everything outside is unaffected.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from opus_magnum_blackbox.config import ModelConfig, RewardConfig, TrainConfig
from opus_magnum_blackbox.env import OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import RolloutBatch, collect_batch, collect_episode


class Trainer:
    """
    Owns policy, optimizer, envs, configs, run directory and rng.
    """

    def __init__(
        self,
        policy: OpusPolicy,
        envs: list[OpusEnv],
        train_config: TrainConfig | None = None,
        reward_config: RewardConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.policy = policy
        self.envs = envs
        self.train_config = train_config or TrainConfig()
        self.reward_config = reward_config or RewardConfig()
        self.model_config = model_config or policy.model_config
        self.pad_id = envs[0].tokenizer.vocab.pad_id
        self.device = torch.device(self.train_config.device or self._auto_device())  # can be forced in `TrainConfig`
        self.policy.to(self.device)  # apply Module to com. device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.train_config.lr)
        self.iteration = 0
        self.run_dir = Path(self.train_config.run_dir) / self._run_name()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _auto_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _run_name(self) -> str:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        levels = sorted({entry.name for env in self.envs for entry in env.levels})
        return f"{stamp}-{'-'.join(levels)}"

    def run(self) -> None:
        """
        Collect, update, log, evaluate and checkpoint until the iteration target is reached.
        """
        if self.iteration == 0:
            self._seed()
        while self.iteration < self.train_config.iterations:
            self.iteration += 1
            started = time.time()

            self.policy.eval()
            batch = collect_batch(
                self.envs,
                self.policy,
                self.train_config.episodes_per_iteration,
                self.device,
                gamma=self.train_config.gamma,
            )
            policy_loss, value_loss, entropy = self.update(batch)
            self._log_metrics(batch, policy_loss, value_loss, entropy, time.time() - started)

            if self.iteration % self.train_config.eval_interval == 0:
                self._log_eval(self.evaluate())

            if self.iteration % self.train_config.checkpoint_interval == 0:
                self.save_checkpoint(numbered=True)

        self.save_checkpoint()

    def update(self, batch: RolloutBatch) -> tuple[float, float, float]:
        """
        One gradient step.

        Returns:
            Tuple, each averaged over valid positions:
                - policy_loss
                - value_loss
                - entropy

        """
        self.policy.train()

        # fmt:off
        encoder_input  = batch.encoder_input.to(self.device)
        encoder_length = batch.encoder_length.to(self.device)
        actions        = batch.actions.to(self.device)
        decoder_length = batch.decoder_length.to(self.device)
        action_masks   = batch.action_masks.to(self.device)
        returns        = batch.returns.to(self.device)
        valid          = batch.valid.to(self.device)
        # fmt:on

        # Padded steps have all-False masks;
        # make PAD valid there to preserve proper distribution.
        # Those are excluded from loss by `valid`.
        action_masks = action_masks.clone()
        action_masks[~valid, self.pad_id] = True

        logits, values = self.policy(encoder_input, encoder_length, actions, decoder_length)
        # Drop the extra BOS output position: position t is distribution for target t.
        logits, values = logits[:, :-1], values[:, :-1]

        dist = self.policy.masked_distribution(logits, action_masks)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        advantages = (returns - values).detach()

        policy_loss = -(logprobs * advantages)[valid].mean()
        value_loss = ((values - returns) ** 2)[valid].mean()
        entropy = entropy[valid].mean()

        loss = (
            policy_loss
            + self.train_config.value_coef * value_loss
            - self.train_config.entropy_coef * entropy
        )  # fmt: skip
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.train_config.grad_clip)
        self.optimizer.step()
        return policy_loss.item(), value_loss.item(), entropy.item()

    def evaluate(self) -> float:
        """
        Greedy solve rate over eval_episodes, with envs round-robining.
        """
        self.policy.eval()
        solved = 0
        for i in range(self.train_config.eval_episodes):
            episode = collect_episode(self.envs[i % len(self.envs)], self.policy, self.device, greedy=True)
            solved += episode.solved
        return solved / self.train_config.eval_episodes

    def save_checkpoint(self, numbered: bool = False) -> Path:
        """
        Full checkpoint: Model, optimizer, iteration, rng state;
        exact-resume payload in latest.pt.
        """
        # fmt: off
        payload = {
            "iteration"       : self.iteration,
            "model_state"     : self.policy.state_dict(),
            "optimizer_state" : self.optimizer.state_dict(),
            "model_config"    : asdict(self.model_config),
            "train_config"    : asdict(self.train_config),
            "reward_config"   : asdict(self.reward_config),
            "rng": {
                "torch" : torch.get_rng_state(),
                "cuda"  : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy" : np.random.get_state(),
            },
        }
        # fmt: on
        path = self.run_dir / "latest.pt"
        torch.save(payload, path)
        if numbered:
            path = self.run_dir / f"iter_{self.iteration:04d}.pt"
            torch.save(payload, path)
        return path

    @classmethod
    def resume(
        cls,
        path: str | Path,
        envs: list[OpusEnv],
        train_config: TrainConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> Trainer:
        """
        Rebuild trainer from checkpoint.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_config = ModelConfig(**payload["model_config"])
        train_config = train_config or TrainConfig(**payload["train_config"])
        reward_config = reward_config or RewardConfig(**payload["reward_config"])

        policy = OpusPolicy(envs[0].tokenizer.config, model_config)
        policy.load_state_dict(payload["model_state"])
        trainer = cls(policy, envs, train_config, reward_config, model_config)
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        # load_state_dict restores lr; overrideing with resotred
        # (identity for exact resume, where train_config comes from the same checkpoint)
        for group in trainer.optimizer.param_groups:
            group["lr"] = trainer.train_config.lr
        trainer.iteration = payload["iteration"]

        rng = payload["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        return trainer

    def _seed(self) -> None:
        seed = self.train_config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        for env in self.envs:
            env.reset(seed=seed)

    def _log_metrics(
        self, batch: RolloutBatch, policy_loss: float, value_loss: float, entropy: float, wall_time: float
    ) -> None:
        entry = {
            "iteration": self.iteration,
            "mean_reward": batch.mean_return,
            "solve_rate": batch.solve_rate,
            "mean_length": batch.mean_length,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "wall_time": wall_time,
        }
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(
            f"iter {self.iteration} | reward {batch.mean_return:+.3f} | solve {batch.solve_rate:.2f} "
            f"| len {batch.mean_length:.1f} | pl {policy_loss:+.4f} | vl {value_loss:.4f} | ent {entropy:.3f}"
        )

    def _log_eval(self, greedy_solve_rate: float) -> None:
        entry = {"iteration": self.iteration, "greedy_solve_rate": greedy_solve_rate}
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(
            f"iter {self.iteration} | greedy solve {greedy_solve_rate:.2f} over {self.train_config.eval_episodes} episodes"
        )
