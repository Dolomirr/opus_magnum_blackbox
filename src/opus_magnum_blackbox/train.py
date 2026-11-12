"""
`Trainer`: masked REINFORCE with learned value baseline, in one visible loop.

`update` is PPO: replacing its body (ratio clip, GAE, epochs per batch)
upgrades algorithm; batch format and everything outside is unaffected.
"""

import json
import time
import zlib
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from opus_magnum_blackbox.config import ModelConfig, PretrainConfig, RewardConfig, TrainConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.profile import Timer
from opus_magnum_blackbox.rollout import RolloutBatch, collect_batched, replicate_envs


class Trainer:
    """
    Owns policy, optimizer, envs, configs, run directory and rng.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        policy: OpusPolicy,
        envs: list[OpusEnv],
        train_config: TrainConfig | None = None,
        reward_config: RewardConfig | None = None,
        model_config: ModelConfig | None = None,
        eval_envs: list[OpusEnv] | None = None,
        autodiscovery: int | None = None,
    ) -> None:
        self.policy = policy
        self.envs = envs
        self.eval_envs = eval_envs or []
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
        # supervision dial (1..10)
        self.autodiscovery: int | None = autodiscovery
        # Optional `profile.ProfiledEnvs`; when set, `run` writes `profile.jsonl`.
        self.profile = None
        # Stepping-independent env replicas for the lockstep collector, cached per slot count.
        self._slot_env_pool: dict[int, list[OpusEnv]] = {}
        # Single-level envs for exhaustive greedy eval, cached per env pool + entry name.
        self._entry_env_pools: dict[tuple[OpusEnv, ...], dict[str, OpusEnv]] = {}
        # Best eval-bank solve rate seen so far; drives `best_eval.pt`.
        self._best_eval_rate: float | None = None
        # Sub-second precision: trainer + resume within one second must not share a run dir.
        # The directory is created lazily on the first write, so resumes can target an existing one.
        self.run_dir = Path(self.train_config.run_dir) / self._run_name()

    @staticmethod
    def _auto_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _run_name(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        names = sorted({entry.name for env in self.envs for entry in env.levels})
        # bank resolves to one common group prefix instead of joining every variant
        common: list[str] = []
        for parts in zip(*(name.split("/") for name in names), strict=False):
            if len(set(parts)) != 1:
                break
            common.append(parts[0])
        label = "/".join(common) if common else names[0]
        # Flatten group separators: the label must stay a single path segment.
        name = f"{stamp}-{label.replace('/', '_')}"
        if self.autodiscovery is not None and self.autodiscovery > 1:
            name += f"-ad{self.autodiscovery}"
        return name

    def run(self, eval_patience: int | None = None) -> None:
        """
        Collect, update, log, evaluate and checkpoint until the iteration target is reached.

        Ends loop early after `eval_patience` many consecutive evals without
        `eval_solve_rate` improvement over best rate seen.
        Needs eval envs; None runs full iteration target).
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.iteration == 0:
            self._seed()
        # Patience bookkeeping, seeded from the best rate earlier stages reached.
        best_eval_rate = self._best_eval_rate
        stale_evals = 0
        while self.iteration < self.train_config.iterations:
            self.iteration += 1
            started = time.time()

            self.policy.eval()
            with self._timed("rollout"):
                batch = collect_batched(
                    self._slot_env_pool_for(self.train_config.episodes_per_iteration),
                    self.policy,
                    self.train_config.episodes_per_iteration,
                    self.device,
                    gamma=self.train_config.gamma,
                    explore_epsilon=self.train_config.explore_epsilon,
                )
            with self._timed("update"):
                policy_loss, value_loss, entropy = self.update(batch)
            iteration_wall = time.time() - started
            self._log_metrics(batch, policy_loss, value_loss, entropy, iteration_wall)
            self._log_profile(batch, iteration_wall)

            if self.iteration % self.train_config.eval_interval == 0:
                with self._timed("eval"):
                    solve_rate = self.evaluate()
                    eval_solve_rate = self._greedy_solve_rate(self.eval_envs) if self.eval_envs else None
                self._log_eval(solve_rate, eval_solve_rate)
                self._track_best_eval(eval_solve_rate)
                if eval_patience is not None and eval_solve_rate is not None:
                    if best_eval_rate is not None and eval_solve_rate > best_eval_rate:
                        stale_evals = 0
                        best_eval_rate = eval_solve_rate
                    else:
                        stale_evals += 1
                        best_eval_rate = eval_solve_rate if best_eval_rate is None else best_eval_rate
                    if stale_evals >= eval_patience:
                        self._log_stage_early_stop(best_eval_rate, eval_solve_rate)
                        break

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
        Exhaustive greedy solve rate over the train bank: one episode per level entry.
        """
        return self._greedy_solve_rate(self.envs)

    def _greedy_solve_rate(self, envs: list[OpusEnv]) -> float:
        """
        One deterministic greedy episode per level entry across `envs` banks.
        """
        self.policy.eval()
        entry_envs = [self._entry_env(envs, entry) for env in envs for entry in env.levels]
        batch = collect_batched(entry_envs, self.policy, len(entry_envs), self.device, greedy=True)
        return batch.solve_rate

    def _entry_env(self, pool: list[OpusEnv], entry: LevelEntry) -> OpusEnv:
        """
        Single-level env for exhaustive greedy eval, cached per pool + entry name.
        Names are unique within a bank, matching `OpusEnv`'s own level cache.
        """
        cache = self._entry_env_pools.setdefault(tuple(pool), {})
        env = cache.get(entry.name)
        if env is None:
            env = OpusEnv([entry])
            cache[entry.name] = env
        return env

    def _slot_env_pool_for(self, n: int) -> list[OpusEnv]:
        """
        Step-lock env replicas for `n` slots, built on first use.

        Replicas are seeded from (train seed, slot count) via a stable hash, so they
        draw independent level sequences and resumed runs reproduce identical streams.
        """
        pool = self._slot_env_pool.get(n)
        if pool is None:
            pool = replicate_envs(
                self.envs,
                n,
                seed=zlib.crc32(f"slots:{self.train_config.seed}:{n}".encode()),
            )
            if self.profile is not None:
                self.profile.envs.extend(pool)
            self._slot_env_pool[n] = pool
        return pool

    def _track_best_eval(self, eval_solve_rate: float | None) -> None:
        """
        Keep `best_eval.pt` at the best eval-bank rate seen.
        Overwritten on actual improvement only.
        """
        if eval_solve_rate is None or (self._best_eval_rate is not None and eval_solve_rate <= self._best_eval_rate):
            return
        self._best_eval_rate = eval_solve_rate
        torch.save(self._checkpoint_payload(), self.run_dir / "best_eval.pt")

    def pretrain(self, batches: list[RolloutBatch], config: PretrainConfig, stage: str | None = None) -> None:
        """
        Behavior cloning: masked NLL on known solution sequences.

        Each batch holds one level's supervised rows;
        rows are reshuffled every epoch and sliced into `config.batch_size` steps.
        Early-stops when train-bank greedy rate and mean loss both reach their targets;
        eval-bank rate is logged alongside when eval envs exist.
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
                eval_solve_rate = self._greedy_solve_rate(self.eval_envs) if self.eval_envs else None
                self._log_pretrain(stage_name, epoch, mean_loss, mean_accuracy, solve_rate, eval_solve_rate)
                self._track_best_eval(eval_solve_rate)
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

    def _checkpoint_payload(self) -> dict:
        # fmt: off
        return {
            "iteration"       : self.iteration,
            "model_state"     : self.policy.state_dict(),
            "optimizer_state" : self.optimizer.state_dict(),
            "model_config"    : asdict(self.model_config),
            "train_config"    : asdict(self.train_config),
            "reward_config"   : asdict(self.reward_config),
            "best_eval_rate"  : self._best_eval_rate,
            "rng": {
                "torch" : torch.get_rng_state(),
                "cuda"  : torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy" : np.random.get_state(),
            },
        }
        # fmt: on

    def save_checkpoint(self, numbered: bool = False) -> Path:
        """
        Full checkpoint: Model, optimizer, iteration, rng state;
        exact-resume payload in latest.pt.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint_payload()
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
        eval_envs: list[OpusEnv] | None = None,
    ) -> Trainer:
        """
        Rebuild trainer from checkpoint.
        Eval envs come from the caller's config, not the checkpoint payload.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model_config = ModelConfig(**payload["model_config"])
        train_config = train_config or TrainConfig(**payload["train_config"])
        reward_config = reward_config or RewardConfig(**payload["reward_config"])

        policy = OpusPolicy(envs[0].tokenizer.config, model_config)
        policy.load_state_dict(payload["model_state"])
        trainer = cls(policy, envs, train_config, reward_config, model_config, eval_envs=eval_envs)
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        # load_state_dict restores lr; overrideing with resotred
        # (identity for exact resume, where train_config comes from the same checkpoint)
        for group in trainer.optimizer.param_groups:
            group["lr"] = trainer.train_config.lr
        trainer.iteration = payload["iteration"]
        trainer._best_eval_rate = payload.get("best_eval_rate")

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
        for env in [*self.envs, *self.eval_envs]:
            env.reset(seed=seed)

    def _stamp_autodiscovery(self, entry: dict) -> None:
        """
        Pipeline supervision dial, stamped onto every metrics row when the run comes from a pipeline.
        """
        if self.autodiscovery is not None:
            entry["autodiscovery"] = self.autodiscovery

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
        self._stamp_autodiscovery(entry)
        entry["timestamp"] = datetime.now().isoformat()
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(
            f"iter {self.iteration} | reward {batch.mean_return:+.3f} | solve {batch.solve_rate:.2f} "
            f"| len {batch.mean_length:.1f} | pl {policy_loss:+.4f} | vl {value_loss:.4f} | ent {entropy:.3f}"
        )

    def _log_eval(self, greedy_solve_rate: float, eval_solve_rate: float | None = None) -> None:
        entry = {"iteration": self.iteration, "greedy_solve_rate": greedy_solve_rate}
        if eval_solve_rate is not None:
            entry["eval_solve_rate"] = eval_solve_rate
        if self.stage is not None:
            entry["stage"] = self.stage
        self._stamp_autodiscovery(entry)
        timestamp = datetime.now().isoformat()
        entry["timestamp"] = timestamp
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        rates = f"{greedy_solve_rate:.2f}" + ("" if eval_solve_rate is None else f" / eval-bank {eval_solve_rate:.2f}")
        counts = self._bank_counts()
        print(f"iter {self.iteration} | greedy solve {rates} over {counts} | {timestamp}")

    def _log_stage_early_stop(self, best_rate: float, last_rate: float) -> None:
        """
        `stage_early_stop` metrics row: eval patience exhausted at this iteration.
        """
        entry = {
            "iteration": self.iteration,
            "stage_early_stop": True,
            "best_eval_rate": best_rate,
            "eval_solve_rate": last_rate,
        }
        if self.stage is not None:
            entry["stage"] = self.stage
        self._stamp_autodiscovery(entry)
        entry["timestamp"] = datetime.now().isoformat()
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(f"iter {self.iteration} | stage early stop | best eval {best_rate:.2f} / last {last_rate:.2f}")

    def _log_stage_skip(self, stage: str, reason: str) -> None:
        """
        `stage_skip` metrics row: the pipeline skipped this stage before it ran.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "iteration": self.iteration,
            "stage": stage,
            "stage_skip": True,
            "reason": reason,
        }
        self._stamp_autodiscovery(entry)
        entry["timestamp"] = datetime.now().isoformat()
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        print(f"stage {stage} skipped | {reason}")

    def _bank_counts(self) -> str:
        """
        "N train" / "N train / M eval" level counts: exhaustive eval episodes equal bank sizes.
        """
        counts = f"{sum(len(env.levels) for env in self.envs)} train"
        if self.eval_envs:
            counts += f" / {sum(len(env.levels) for env in self.eval_envs)} eval"
        return counts

    def _log_pretrain(  # noqa: PLR0913, PLR0917
        self,
        stage: str,
        epoch: int,
        loss: float,
        accuracy: float,
        greedy_solve_rate: float,
        eval_solve_rate: float | None = None,
    ) -> None:
        entry = {
            "iteration": epoch,
            "stage": stage,
            "loss": loss,
            "token_accuracy": accuracy,
            "greedy_solve_rate": greedy_solve_rate,
        }
        if eval_solve_rate is not None:
            entry["eval_solve_rate"] = eval_solve_rate
        self._stamp_autodiscovery(entry)
        entry["timestamp"] = datetime.now().isoformat()
        with (self.run_dir / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")
        rates = f"{greedy_solve_rate:.2f}" + ("" if eval_solve_rate is None else f" / eval-bank {eval_solve_rate:.2f}")
        print(
            f"pretrain epoch {epoch} | loss {loss:.4f} | acc {accuracy:.3f} "
            f"| greedy solve {rates} over {self._bank_counts()}"
        )
