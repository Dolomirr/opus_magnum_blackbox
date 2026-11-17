"""
Error-taxonomy probe: greedy per-variant failure classes over a pipeline run's bank.

Reads the run's resolved `config.toml` (copied into the run dir by the pipeline),
resolves its eval bank (`--bank train` for the train bank, and the fallback when
the config declares no eval levels), runs one deterministic greedy episode per
variant with the given checkpoint, and prints failure-class counts per bank group
and hex-distance class: solved / wrong blueprint / wrong base / wrong program /
wrong ticks / invalid / other.

Legacy pre-insertion checkpoints are remapped on load (see `taxonomy.load_policy`),
which can shift greedy behavior relative to their original training-time telemetry;
checkpoints stored under the current vocabulary probe exactly.

Usage: `uv run scripts/error_taxonomy.py <run_dir> <ckpt> [--bank {eval,train}]`
"""

import argparse
import sys
import tomllib
from pathlib import Path

import torch

from opus_magnum_blackbox.corpus import resolve_entries
from opus_magnum_blackbox.env import OpusEnv
from opus_magnum_blackbox.taxonomy import error_taxonomy, format_table, load_policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Greedy per-variant error taxonomy over a pipeline run's bank.")
    parser.add_argument("run_dir", type=Path, help="pipeline run directory (its config.toml names the banks)")
    parser.add_argument("ckpt", type=Path, help="checkpoint to probe (e.g. <run_dir>/best_eval.pt)")
    parser.add_argument("--bank", choices=("eval", "train"), default="eval", help="which configured bank to probe")
    args = parser.parse_args(argv)

    config = tomllib.loads((args.run_dir / "config.toml").read_text())["run"]
    names = config["levels"]
    if args.bank == "eval" and config.get("eval_levels"):
        names = config["eval_levels"]
    entries = resolve_entries(list(names))

    features = tuple(torch.load(args.ckpt, map_location="cpu", weights_only=False).get("encoder_features", ()))
    envs = [OpusEnv([entry], encoder_features=features) for entry in entries]
    policy, payload = load_policy(args.ckpt, envs[0].tokenizer.config)
    remap = payload.get("vocab_remap")
    remap_note = f" | vocab remapped {remap['from']}->{remap['to']}" if remap else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    tables = error_taxonomy(policy, entries, device, encoder_features=features)
    print(f"{args.ckpt} (iter {payload.get('iteration')}{remap_note}) over {len(entries)} variants: {', '.join(names)}")
    for group, table in tables.items():
        solved = sum(counts.get("solved", 0) for counts in table.values())
        total = sum(sum(counts.values()) for counts in table.values())
        print(f"\n{group} ({len(entries)} variants): solved {solved}/{total}")
        print(format_table(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
