"""
Coverage probe: greedy solved/total over full banks, by class x hex-distance x bank.

The batch-analysis readout: one command per run covers the whole class-bank space,

    uv run scripts/coverage.py <run_dir> <run_dir>/best_eval.pt \
        tier2_single/train tier2_single/eval tier2_handoff/train tier2_handoff/eval

Each bank argument is a corpus resolve name (one level or a directory group).
Every variant gets one deterministic greedy episode with the given checkpoint;
the level class follows the bank rule (target 1 -> handoff, else single), and
the distance is the entrance/exit hex distance. `<run_dir>` is header context
only; banks are named explicitly so ad-hoc slices can be probed too.

Legacy pre-insertion checkpoints are remapped on load (see `taxonomy.load_policy`),
which can shift greedy behavior relative to their original training-time telemetry;
checkpoints stored under the current vocabulary probe exactly.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from opus_magnum_sym import Level, LevelSpec

from opus_magnum_blackbox.corpus import resolve_entries
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.rollout import collect_batched
from opus_magnum_blackbox.taxonomy import load_policy, pair_distance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Greedy solved/total coverage by class x distance x bank.")
    parser.add_argument("run_dir", type=Path, help="pipeline run directory (header context)")
    parser.add_argument("ckpt", type=Path, help="checkpoint to probe (e.g. <run_dir>/best_eval.pt)")
    parser.add_argument("banks", nargs="+", help="corpus resolve names, e.g. tier2_single/train tier2_handoff/eval")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = None
    print(f"{args.ckpt} coverage over {args.run_dir} | banks: {', '.join(args.banks)}")
    for bank in args.banks:
        entries = resolve_entries([bank])
        envs = [OpusEnv([entry]) for entry in entries]
        if policy is None:
            policy, payload = load_policy(args.ckpt, envs[0].tokenizer.config)
            policy.to(device)
            remap = payload.get("vocab_remap")
            note = f" | vocab remapped {remap['from']}->{remap['to']}" if remap else ""
            print(f"checkpoint iter {payload.get('iteration')}{note}")
        batch = collect_batched(envs, policy, len(envs), device, greedy=True)
        _print_bank(bank, entries, batch.episode_solved.tolist())
    return 0


def _print_bank(bank: str, entries: list[LevelEntry], solved: list[bool]) -> None:
    """
    Per-bank section: class x distance rows (ascending), per-class and bank totals.
    """
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])  # (class, distance) -> [solved, total]
    for entry, is_solved in zip(entries, solved, strict=True):
        spec = _entry_spec(entry)
        kind = "handoff" if spec.target == 1 else "single"
        cell = counts[(kind, pair_distance(spec))]
        cell[0] += int(is_solved)
        cell[1] += 1
    bank_solved = sum(cell[0] for cell in counts.values())
    bank_total = sum(cell[1] for cell in counts.values())
    print(f"\n{bank}: solved {bank_solved}/{bank_total} ({bank_solved / bank_total:.3f})" if bank_total else f"\n{bank}: empty")
    for kind, distance in sorted(counts):
        cell_solved, cell_total = counts[(kind, distance)]
        print(f"  {kind:<8} d{distance:<3} {cell_solved}/{cell_total}")
    for kind in sorted({kind for kind, _ in counts}):
        kind_solved = sum(cell[0] for (cell_kind, _), cell in counts.items() if cell_kind == kind)
        kind_total = sum(cell[1] for (cell_kind, _), cell in counts.items() if cell_kind == kind)
        print(f"  {kind:<8} --  {kind_solved}/{kind_total}")


def _entry_spec(entry: LevelEntry) -> LevelSpec:
    level = Level.from_file(entry.source) if isinstance(entry.source, Path) else Level(entry.source)
    return level.spec


if __name__ == "__main__":
    sys.exit(main())
