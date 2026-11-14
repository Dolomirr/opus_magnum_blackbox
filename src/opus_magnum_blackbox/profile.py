import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from opus_magnum_blackbox.codec import DecodeReport
from opus_magnum_blackbox.env import LevelEntry, OpusEnv, progress_distances
from opus_magnum_blackbox.reward import compute_reward


@dataclass
class PhaseStats:
    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds
        self.calls[name] = self.calls.get(name, 0) + 1

    def flush(self) -> tuple[dict[str, float], dict[str, int]]:
        seconds, calls = self.seconds, self.calls
        self.seconds, self.calls = {}, {}
        return seconds, calls


class Timer:
    def __init__(self, stats: PhaseStats, name: str) -> None:
        self.stats = stats
        self.name = name

    def __enter__(self) -> Timer:
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.stats.add(self.name, time.perf_counter() - self.started)


class ProfiledEnv(OpusEnv):
    def __init__(
        self,
        levels: list[LevelEntry],
        tokenizer_config=None,
        reward_config=None,
    ) -> None:
        super().__init__(levels, tokenizer_config, reward_config)
        self.prof = PhaseStats()

    def _observation(self) -> dict:
        with Timer(self.prof, "mask"):
            return super()._observation()

    def _finish(self, info: dict) -> tuple[float, dict]:
        level = self._level_cache[self._current.name]

        with Timer(self.prof, "codec_parse"):
            report: DecodeReport = self.tokenizer.decode_solution_to_level(self._history, self._spec, level)
        result = None
        if report.ok:
            with Timer(self.prof, "sim_run"):
                result = level.run()
        initial_dist, final_dist = progress_distances(level)
        with Timer(self.prof, "sim_reset"):
            level.reset()
        with Timer(self.prof, "reward"):
            reward = compute_reward(
                report,
                result,
                self._spec,
                replace(self.reward_config, cost_max=self._current.cost_max),
                initial_dist=initial_dist,
                final_dist=final_dist,
            )

        if result is not None:
            info.update(
                {
                    "status": result.status.value,
                    "reason": result.reason.name if result.reason else None,
                    "solved": result.solved,
                    "metrics": result.metrics,
                    "crash_step": result.error.step if result.error else None,
                    "total_finished": result.total_finished,
                }
            )
        else:
            info.update({"status": "static_failure", "reason": report.error, "solved": False})
        info["episode"] = {"solved": info["solved"], "reward": reward, "length": len(self._history)}
        return reward, info


class ProfiledEnvs:
    def __init__(self, envs: list[ProfiledEnv]) -> None:
        self.envs = envs
        self.loop = PhaseStats()

    def snapshot(self, **extra: float) -> dict:
        env_seconds: dict[str, float] = {}
        env_calls: dict[str, int] = {}
        for env in self.envs:
            seconds, calls = env.prof.flush()
            for name, value in seconds.items():
                env_seconds[name] = env_seconds.get(name, 0.0) + value
            for name, value in calls.items():
                env_calls[name] = env_calls.get(name, 0) + value
        loop_seconds, loop_calls = self.loop.flush()
        return {"env": env_seconds, "env_calls": env_calls, "loop": loop_seconds, "loop_calls": loop_calls, **extra}

    def log(self, path: Path, payload: dict) -> None:
        payload = {"timestamp": datetime.now().isoformat(), **payload}
        with path.open("a") as file:
            file.write(json.dumps(payload) + "\n")


def summarize(path: Path) -> dict:
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    training = [entry for entry in entries if entry.get("kind") == "iteration"]
    if not training:
        return {"entries": 0}

    sections = ("env", "loop")
    table: dict[str, dict] = {section: {} for section in sections}
    for section in sections:
        for name in sorted({key for entry in training for key in entry[section]}):
            values = [entry[section].get(name, 0.0) for entry in training]
            calls = [entry[f"{section}_calls"].get(name, 0) for entry in training]
            table[section][name] = {"mean_ms": 1000 * sum(values) / len(values), "mean_calls": sum(calls) / len(calls)}

    iteration_ms = 1000 * sum(entry["iteration_wall"] for entry in training) / len(training)
    rollout_ms = table["loop"].get("rollout", {}).get("mean_ms", 0.0)
    update_ms = table["loop"].get("update", {}).get("mean_ms", 0.0)
    eval_ms = table["loop"].get("eval", {}).get("mean_ms", 0.0)
    env_total = sum(value["mean_ms"] for value in table["env"].values())

    return {
        "entries": len(training),
        "iteration_ms": iteration_ms,
        "table": table,
        "rollup": {
            "rollout": rollout_ms,
            "update": update_ms,
            "eval": eval_ms,
            "other": iteration_ms - rollout_ms - update_ms - eval_ms,
            "env_total": env_total,
            # `env_total` is inside `rollout`; the rest of it is the sampling frame.
            "rollout_frame": rollout_ms - env_total,
        },
    }


def format_summary(report: dict) -> str:
    if not report.get("entries"):
        return "profile: no iteration entries"
    iteration_ms = report["iteration_ms"]
    lines = [f"profile over {report['entries']} iterations: {iteration_ms:.1f} ms/iteration"]

    rollup = report["rollup"]
    lines.extend(
        f"  {name:<14} {rollup[name]:8.1f} ms  {100.0 * rollup[name] / iteration_ms:5.1f}%"
        for name in ("rollout", "update", "eval", "other")
    )
    lines.append(
        f"  {'env (all)':<14} {rollup['env_total']:8.1f} ms  {100.0 * rollup['env_total'] / iteration_ms:5.1f}%"
        "  (sum of env phases below)"
    )

    lines.append("env phases (ms/iter, calls/iter):")
    for name, stats in sorted(report["table"]["env"].items(), key=lambda item: -item[1]["mean_ms"]):
        lines.append(
            f"  {name:<14} {stats['mean_ms']:8.3f} ms  {100.0 * stats['mean_ms'] / iteration_ms:5.1f}%"
            f"  {stats['mean_calls']:8.1f} calls"
        )
    return "\n".join(lines)
