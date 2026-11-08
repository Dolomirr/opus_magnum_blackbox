"""
`Trainer`: masked REINFORCE with learned value baseline, in one visible loop.

`update` is PPO: replacing its body (ratio clip, GAE, epochs per batch)
upgrades algorithm; batch format and everything outside is unaffected.
"""

import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from opus_magnum_blackbox.config import ModelConfig, PretrainConfig, RewardConfig, TrainConfig
from opus_magnum_blackbox.env import OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.profile import Timer
from opus_magnum_blackbox.rollout import RolloutBatch, collect_batched


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
        # Stage name stamped onto metrics entries while a pipeline stage runs; None outside.
        self.stage: str | None = None
        # Optional `profile.ProfiledEnvs`; when set, `run` writes `profile.jsonl`.
        self.profile = None
        # Sub-second precision: trainer + resume within one second must not share a run dir.
        # The directory is created lazily on the first write, so resumes can target an existing one.
        self.run_dir = Path(self.train_config.run_dir) / self._run_name()

    @staticmethod
    def _auto_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _run_name(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        levels = sorted({entry.name for env in self.envs for entry in env.levels})
        return f"{stamp}-{'-'.join(levels)}"

    def run(self) -> None:
        """
        Collect, update, log, evaluate and checkpoint until the iteration target is reached.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.iteration == 0:
            self._seed()
        while self.iteration < self.train_config.iterations:
            self.iteration += 1
            started = time.time()

            self.policy.eval()
            with self._timed("rollout"):
                batch = collect_batched(
                    self.envs,
                    self.policy,
                    self.train_config.episodes_per_iteration,
                    self.device,
                    gamma=self.train_config.gamma,
                )
            with self._timed("update"):
                policy_loss, value_loss, entropy = self.update(batch)
            iteration_wall = time.time() - started
            self._log_metrics(batch, policy_loss, value_loss, entropy, iteration_wall)
            self._log_profile(batch, iteration_wall)

            if self.iteration % self.train_config.eval_interval == 0:
                with self._timed("eval"):
                    solve_rate = self.evaluate()
                self._log_eval(solve_rate)

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

    def _timed(self, name: str):
        """
        Loop-phase timer; a no-op context when no profiler is attached.
        """
        return Timer(self.profile.loop, name) if self.profile is not None else nullcontext()

    def _log_profile(self, batch: RolloutBatch, iteration_wall: float) -> None:
        """
        Append the iteration's phase attribution to `profile.jsonl`.
        """
        if self.profile is None:
            return
        payload = self.profile.snapshot(
            kind="iteration",
            iteration=self.iteration,
            iteration_wall=iteration_wall,
            sim={
                "episodes": len(batch.decoder_length),
                "steps": int(batch.valid.sum().item()),
                "solve_rate": batch.solve_rate,
                "mean_length": batch.mean_length,
            },
        )
        self.profile.log(self.run_dir / "profile.jsonl", payload)

    def evaluate(self) -> float:
        """
        Greedy solve rate over eval_episodes, with envs round-robining.
        """
        self.policy.eval()
        batch = collect_batched(self.envs, self.policy, self.train_config.eval_episodes, self.device, greedy=True)
        return batch.solve_rate

    def pretrain(self, batches: list[RolloutBatch], config: PretrainConfig, stage: str | None = None) -> None:
        """
        Behavior cloning: masked NLL on known solution sequences.

        Each batch holds one level's supervised rows;
        rows are reshuffled every epoch and sliced into `config.batch_size` steps.
        Early-stops when greedy solve rate and mean loss both reach their targets.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        previous_lr = self.optimizer.param_groups[0]["lr"]
        for group in self.optimizer.param_groups:
            group["lr"] = config.lr

        stage_name = stage or "pretrain"
        for epoch in range(1, config.epochs + 1):
            losses: list[float] = []
            accuracies: list[float] = []
            for batch in batches:
                order = torch.randperm(len(batch.decoder_length))
                for start in range(0, len(order), config.batch_size):
                    loss, accuracy = self._bc_update(batch, order[start : start + config.batch_size])
                    losses.append(loss)
                    accuracies.append(accuracy)
            mean_loss = sum(losses) / len(losses)
            mean_accuracy = sum(accuracies) / len(accuracies)
            if epoch % config.eval_interval == 0 or epoch == config.epochs:
                solve_rate = self.evaluate()
                self._log_pretrain(stage_name, epoch, mean_loss, mean_accuracy, solve_rate)
                if solve_rate >= config.early_stop_solve_rate and mean_loss <= config.early_stop_loss:
                    break

        self.save_checkpoint()
        for group in self.optimizer.param_groups:
            group["lr"] = previous_lr

    def _bc_update(self, batch: RolloutBatch, rows: torch.Tensor) -> tuple[float, float]:
        """
        One gradient step over given batch rows.

        Returns loss and token accuracy, both averaged over valid positions.
        """
        self.policy.train()

        # fmt: off
        encoder_input  = batch.encoder_input[rows].to(self.device)
        encoder_length = batch.encoder_length[rows].to(self.device)
        actions        = batch.actions[rows].to(self.device)
        decoder_length = batch.decoder_length[rows].to(self.device)
        action_masks   = batch.action_masks[rows].to(self.device)
        valid          = batch.valid[rows].to(self.device)
        # fmt: on

        # Padded steps have all-False masks;
        # make PAD valid there to preserve proper distribution.
        # Those are excluded from loss by `valid`.
        action_masks = action_masks.clone()
        action_masks[~valid, self.pad_id] = True

        logits, _ = self.policy(encoder_input, encoder_length, actions, decoder_length)
        dist = self.policy.masked_distribution(logits[:, :-1], action_masks)
        logprobs = dist.log_prob(actions)
        accuracy = dist.probs.argmax(dim=-1)[valid].eq(actions[valid]).float().mean()

        loss = -logprobs[valid].mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.train_config.grad_clip)
        self.optimizer.step()
        return loss.item(), accuracy.item()

    def save_checkpoint(self, numbered: bool = False) -> Path:
        """
        Full checkpoint: Model, optimizer, iteration, rng state;
        exact-resume payload in latest.pt.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
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
        if self.stage is not None:
            entry["stage"] = self.stage
        entry["timestamp"] = datetime.now().isoformat()
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(
            f"iter {self.iteration} | reward {batch.mean_return:+.3f} | solve {batch.solve_rate:.2f} "
            f"| len {batch.mean_length:.1f} | pl {policy_loss:+.4f} | vl {value_loss:.4f} | ent {entropy:.3f}"
        )

    def _log_eval(self, greedy_solve_rate: float) -> None:
        entry = {"iteration": self.iteration, "greedy_solve_rate": greedy_solve_rate}
        if self.stage is not None:
            entry["stage"] = self.stage
        timestamp = datetime.now().isoformat()
        entry["timestamp"] = timestamp
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(
            f"iter {self.iteration} | greedy solve {greedy_solve_rate:.2f} over {self.train_config.eval_episodes} episodes | {timestamp}"
        )

    def _log_pretrain(self, stage: str, epoch: int, loss: float, accuracy: float, greedy_solve_rate: float) -> None:
        entry = {
            "iteration": epoch,
            "stage": stage,
            "loss": loss,
            "token_accuracy": accuracy,
            "greedy_solve_rate": greedy_solve_rate,
        }
        entry["timestamp"] = datetime.now().isoformat()
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(
            f"pretrain epoch {epoch} | loss {loss:.4f} | acc {accuracy:.3f} "
            f"| greedy solve {greedy_solve_rate:.2f} over {self.train_config.eval_episodes} episodes"
        )
