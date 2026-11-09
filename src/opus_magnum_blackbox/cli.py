import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch
from opus_magnum_sym import Level

from opus_magnum_blackbox.codec import apply_solution, decode_solution, solution_to_oms
from opus_magnum_blackbox.config import ModelConfig, RewardConfig, TrainConfig
from opus_magnum_blackbox.corpus import TIER0_LEVELS
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.grammar import GrammarError
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.profile import format_summary, summarize
from opus_magnum_blackbox.rollout import collect_episode
from opus_magnum_blackbox.stages import load_pipeline, run_pipeline
from opus_magnum_blackbox.train import Trainer


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        _train(args)
    elif args.command == "run":
        _run(args)
    elif args.command == "generate":
        _generate(args)
    else:
        _eval(args)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opus-magnum-blackbox", description="Training CLI for Opus Magnum blackbox.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train on a corpus level")

    # fmt: off
    train.add_argument("--level",                        default=None, help="corpus level name (default: first tier-0 level)")
    train.add_argument("--iterations",       type=int,   default=None)
    train.add_argument("--episodes",         type=int,   default=None, help="episodes per iteration")
    train.add_argument("--lr",               type=float, default=None)
    train.add_argument("--entropy-coef",     type=float, default=None)
    train.add_argument("--eval-interval",    type=int,   default=None)
    train.add_argument("--eval-episodes",    type=int,   default=None)
    train.add_argument("--seed",             type=int,   default=None)
    train.add_argument("--device",                       default=None, help='e.g. cuda or cpu; default auto-resolves')
    train.add_argument("--run-dir",                      default=None)
    train.add_argument("--shaping-weight",   type=float, default=None, help="reward shaping on unsolved COMPLETED (fallback knob)")
    train.add_argument("--placement-reward", type=float, default=None, help="per-placement credit on failure (fallback knob)")
    train.add_argument("--alpha",            type=float, default=None, help="reward weight: cost bonus (rl-optimise knob)")
    train.add_argument("--beta",             type=float, default=None, help="reward weight: area bonus (rl-optimise knob)")
    train.add_argument("--gamma",            type=float, default=None, help="reward weight: cycles bonus (rl-optimise knob)")
    train.add_argument("--resume",                       default=None, help="checkpoint path to resume from")

    evaluate = subparsers.add_parser("eval", help="greedy-evaluate a checkpoint")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--level",                 default=None, help="corpus level name (default: first tier-0 level)")
    evaluate.add_argument("--episodes",    type=int, default=TrainConfig().eval_episodes)
    evaluate.add_argument("--device",                default=None)
    evaluate.add_argument("--dump-tokens", action="store_true",    help="print each episode's token sequence")

    generate = subparsers.add_parser("generate", help="generate .oms solution files from a checkpoint")
    generate.add_argument("checkpoint")
    generate.add_argument("--level",                 default=None,               help="corpus level name or path to a .oml file (default: first tier-0 level)")
    generate.add_argument("--count", "-n", type=int, default=1,                  help="number of independent solutions")
    generate.add_argument("--seed",        type=int, default=TrainConfig().seed, help="base seed; solution i is seeded with seed + i")
    generate.add_argument("--out-dir",               default="solutions",        help="directory for generated .oms files")
    generate.add_argument("--device",                default=None)
    # fmt: on

    run = subparsers.add_parser("run", help="execute a stage pipeline from a TOML file")
    run.add_argument("config", help="pipeline TOML file")
    run.add_argument(
        "--resume",
        default=None,
        help="existing run dir with stage_<name>.pt checkpoints; stages already completed there are skipped",
    )
    return parser


def _find_entry(name: str | None) -> LevelEntry:
    if name is None:
        return TIER0_LEVELS[0]
    path = Path(name)
    if path.suffix == ".oml" and path.exists():
        return LevelEntry(source=path, cost_max=RewardConfig().cost_max, name=path.stem)
    for entry in TIER0_LEVELS:
        if entry.name == name:
            return entry
    known = ", ".join(entry.name for entry in TIER0_LEVELS)
    sys.exit(f"unknown level {name!r}; give a corpus level name or a path to a .oml file (corpus has: {known})")


def _train_config(args: argparse.Namespace, base: TrainConfig) -> TrainConfig:
    """
    CLI values override `base`; None means unspecified.
    """
    # fmt: off
    mapping = {
        "iterations"             : args.iterations,
        "episodes_per_iteration" : args.episodes,
        "lr"                     : args.lr,
        "entropy_coef"           : args.entropy_coef,
        "eval_interval"          : args.eval_interval,
        "eval_episodes"          : args.eval_episodes,
        "seed"                   : args.seed,
        "device"                 : args.device,
        "run_dir"                : args.run_dir,
    }
    # fmt: on
    return replace(base, **{field: value for field, value in mapping.items() if value is not None})


def _reward_config(args: argparse.Namespace, base: RewardConfig) -> RewardConfig:
    """
    Fallback and rl-optimise knobs; None means unspecified.
    """
    mapping = {
        "shaping_weight": args.shaping_weight,
        "placement_reward": args.placement_reward,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
    }
    return replace(base, **{field: value for field, value in mapping.items() if value is not None})


def _train(args: argparse.Namespace) -> None:
    entry = _find_entry(args.level)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        train_config = _train_config(args, TrainConfig(**payload["train_config"]))
        reward_config = _reward_config(args, RewardConfig(**payload["reward_config"]))
        env = OpusEnv([entry], reward_config=reward_config)
        trainer = Trainer.resume(args.resume, [env], train_config=train_config, reward_config=reward_config)
    else:
        reward_config = _reward_config(args, RewardConfig())
        env = OpusEnv([entry], reward_config=reward_config)
        trainer = Trainer(OpusPolicy(env.tokenizer.config), [env], _train_config(args, TrainConfig()), reward_config)
    print(f"run directory: {trainer.run_dir}")
    trainer.run()


def _run(args: argparse.Namespace) -> None:
    pipeline = load_pipeline(args.config)
    run_dir = run_pipeline(pipeline, resume_dir=args.resume)
    print(f"run directory: {run_dir}")
    if (run_dir / "profile.jsonl").exists():
        report = summarize(run_dir / "profile.jsonl")
        print(format_summary(report))


def _eval(args: argparse.Namespace) -> None:
    env = OpusEnv([_find_entry(args.level)])
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = OpusPolicy(env.tokenizer.config, ModelConfig(**payload["model_config"]))
    policy.load_state_dict(payload["model_state"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy.to(device).eval()

    solved = 0
    rewards: list[float] = []
    for _ in range(args.episodes):
        episode = collect_episode(env, policy, device, greedy=True)
        solved += episode.solved
        rewards.append(sum(episode.rewards))
        if args.dump_tokens:
            tokens = " ".join(env.tokenizer.vocab.name(token) for token in episode.tokens)
            print(f"{'solved' if episode.solved else 'failed'} ({sum(episode.rewards):+.1f}): {tokens}")
    rate = solved / args.episodes
    print(f"solve rate: {solved}/{args.episodes} = {rate:.2f} | mean reward: {sum(rewards) / len(rewards):+.3f}")


def _generate(args: argparse.Namespace) -> None:
    entry = _find_entry(args.level)
    env = OpusEnv([entry])
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = OpusPolicy(env.tokenizer.config, ModelConfig(**payload["model_config"]))
    policy.load_state_dict(payload["model_state"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy.to(device).eval()

    level = Level.from_file(entry.source) if isinstance(entry.source, Path) else Level(entry.source)
    spec = level.spec
    vocab = env.tokenizer.vocab
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = solved = 0
    for i in range(args.count):
        # reseed both rngs so each solution is independent
        torch.manual_seed(args.seed + i)
        env.reset(seed=args.seed + i)
        episode = collect_episode(env, policy, device)

        sol = decode_solution(episode.tokens, spec, vocab)
        if isinstance(sol, GrammarError):
            print(f"[{i}] grammar error at token {sol.pos}: {sol.message}")
            continue
        report = apply_solution(sol, spec, level)
        result = level.run() if report.ok else None
        level.reset()
        if not report.ok:
            print(f"[{i}] placement failed: {report.error}")
            continue

        path = out_dir / f"{entry.name}.{i:02d}.oms"
        path.write_text(solution_to_oms(sol, report.handles, spec))
        written += 1
        solved += result is not None and result.solved
        status = result.status.value if result is not None else "unapplied"
        print(f"[{i}] {path}: {status} ({sum(episode.rewards):+.1f})")
    print(f"wrote {written}/{args.count} solutions to {out_dir}, {solved} solved")
