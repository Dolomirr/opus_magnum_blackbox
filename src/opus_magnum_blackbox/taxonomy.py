"""
Error taxonomy over greedy episodes: classify every bank variant's outcome
against its reference solution, per hex-distance class.

Classes (the stage-4 analysis buckets):

- `solved`: the simulation reached the target.
- `wrong_blueprint`: the placed blueprint multiset differs from the reference.
- `wrong_base`: blueprints match, but some arm stands elsewhere or faces elsewhere
  (rotation, column, row multiset differs).
- `wrong_program`: placements match, but the (handle, tick) pairs differ — actions
  scheduled on the wrong ticks or handles.
- `wrong_ticks`: the (handle, tick) skeleton matches, but the action types on it
  differ.
- `invalid`: the greedy sequence does not decode under the grammar.
- `other`: unsolved yet structurally equal to the reference — nothing left to
  attribute (should not happen on verified banks).

Both sides are canonicalized before comparison: placements compare as multisets,
and program handles are relabelled by sorted manipulator position, so a
reordered-but-equivalent solution still counts as matching.
"""

from collections import defaultdict
from pathlib import Path

import torch
from opus_magnum_sym import Level, LevelSpec

from opus_magnum_blackbox.codec import decode_solution
from opus_magnum_blackbox.config import ModelConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.families import hex_distance
from opus_magnum_blackbox.grammar import GrammarError, Solution
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import RolloutBatch, collect_batched
from opus_magnum_blackbox.solutions import load_solutions
from opus_magnum_blackbox.train import _remap_vocab_ids
from opus_magnum_blackbox.vocab import Vocab, insertion_permutation

FAILURE_CLASSES = (
    "solved",
    "wrong_blueprint",
    "wrong_base",
    "wrong_program",
    "wrong_ticks",
    "invalid",
    "other",
)
# fmt: off
type ClassCounts = dict[str, int]           # class -> episode count
type DistanceTable = dict[int, ClassCounts] # distance -> class -> count
type GroupTables = dict[str, DistanceTable] # bank group -> distance table
# fmt: on


def pair_distance(spec: LevelSpec) -> int:
    """
    Hex distance between the spec's fixed entrance and exit.
    """
    positions = {entrypoint.kind.name.lower(): entrypoint.pos for entrypoint in spec.entrypoints}
    return hex_distance(positions["entrance"], positions["exit"])


def classify_episode(
    solved: bool,
    greedy_tokens: list[int],
    reference_tokens: list[int],
    vocab: Vocab,
    spec: LevelSpec,
) -> str:
    """
    Taxonomy class of one greedy episode against its reference sequence.
    """
    if solved:
        return "solved"
    greedy = decode_solution(list(greedy_tokens), spec, vocab)
    reference = decode_solution(list(reference_tokens), spec, vocab)
    if isinstance(greedy, GrammarError) or isinstance(reference, GrammarError):
        return "invalid"
    return _difference_class(greedy, reference)


def _difference_class(greedy: Solution, reference: Solution) -> str:
    """
    First structural difference from the reference, in check order: blueprint
    multiset, base placements, program skeleton, actions on the skeleton.
    """
    greedy_placements = _placement_keys(greedy)
    reference_placements = _placement_keys(reference)
    if sorted(name for name, _, _, _ in greedy_placements) != sorted(name for name, _, _, _ in reference_placements):
        return "wrong_blueprint"
    if sorted(greedy_placements) != sorted(reference_placements):
        return "wrong_base"
    greedy_pairs, greedy_triplets = _program_keys(greedy)
    reference_pairs, reference_triplets = _program_keys(reference)
    if greedy_pairs != reference_pairs:
        return "wrong_program"
    if greedy_triplets != reference_triplets:
        return "wrong_ticks"
    return "other"


def batch_taxonomy(batch: RolloutBatch, entries: list[LevelEntry], vocab: Vocab) -> GroupTables:
    """
    Classify a collected greedy batch row by row and count classes per bank
    group (the entries' first two name segments) and hex-distance class.
    """
    specs = {entry.name: _entry_spec(entry) for entry in entries}
    references = {entry.name: _reference_tokens(entry) for entry in entries}
    tables: GroupTables = {}
    for entry in entries:
        tables.setdefault(entry_group(entry.name), defaultdict(lambda: defaultdict(int)))
    lengths = batch.decoder_length.tolist()
    for row, name in enumerate(batch.level_names):
        tokens = batch.actions[row, : lengths[row]].tolist()
        solved = bool(batch.episode_solved[row])
        classification = classify_episode(solved, tokens, references[name], vocab, specs[name])
        distance = pair_distance(specs[name])
        tables[entry_group(name)][distance][classification] += 1
    return {
        group: {distance: dict(counts) for distance, counts in sorted(per_distance.items())}
        for group, per_distance in tables.items()
    }


def error_taxonomy(policy: OpusPolicy, entries: list[LevelEntry], device, encoder_features: tuple[str, ...] = ()) -> GroupTables:
    """
    One deterministic greedy episode per entry, classified against its reference.
    """
    envs = [OpusEnv([entry], encoder_features=encoder_features) for entry in entries]
    batch = collect_batched(envs, policy, len(envs), device, greedy=True)
    return batch_taxonomy(batch, entries, envs[0].tokenizer.vocab)


def load_policy(path: str | Path, tokenizer_config) -> tuple[OpusPolicy, dict]:
    """
    Policy rebuilt from a checkpoint payload (model weights + stored model config).

    Checkpoints from before the `INSERTED_BUILTINS` vocab insertion are remapped
    onto current ids — the same permutation `Trainer.warm_start` applies; inserted
    builtins get small fresh values from seed 0. The remap is recorded in the
    returned payload as `vocab_remap`.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = payload["model_config"]
    if not isinstance(model_config, ModelConfig):
        model_config = ModelConfig(**model_config)
    policy = OpusPolicy(tokenizer_config, model_config, encoder_features=tuple(payload.get("encoder_features", ())))
    state = dict(payload["model_state"])
    vocab = Vocab(tokenizer_config.vocab)
    old_vocab_size = int(state["policy_head.weight"].shape[0])
    if old_vocab_size != len(vocab):
        embed_rows = int(state["embed.weight"].shape[0])
        bias_size = int(state["policy_head.bias"].shape[0])
        if embed_rows != old_vocab_size + 1 or bias_size != old_vocab_size:
            raise ValueError(
                f"checkpoint {path}: inconsistent vocab-sized shapes "
                f"(embed {embed_rows}, head {old_vocab_size}, bias {bias_size})"
            )
        state = _remap_vocab_ids(state, vocab, insertion_permutation(old_vocab_size, vocab), seed=0)
        payload["vocab_remap"] = {"from": old_vocab_size, "to": len(vocab)}
    policy.load_state_dict(state)
    return policy, payload


def format_table(table: DistanceTable) -> str:
    """
    Fixed-width grid: class rows, ascending distance columns, per-class totals.
    """
    distances = sorted(table)
    classes = [name for name in FAILURE_CLASSES if any(table[distance].get(name) for distance in distances)]
    header = "class".ljust(16) + "".join(f"d{distance:<4}" for distance in distances) + "total"
    lines = [header]
    for name in classes:
        cells = "".join(f"{table[distance].get(name, 0):<5}" for distance in distances)
        lines.append(name.ljust(16) + cells + str(sum(table[distance].get(name, 0) for distance in distances)))
    return "\n".join(lines)


def entry_group(name: str) -> str:
    """
    Bank group of a corpus level name: its first two path segments.
    """
    parts = name.split("/")
    return "/".join(parts[:2]) if len(parts) > 1 else name


def _entry_spec(entry: LevelEntry) -> LevelSpec:
    level = Level.from_file(entry.source) if isinstance(entry.source, Path) else Level(entry.source)
    return level.spec


def _reference_tokens(entry: LevelEntry) -> list[int]:
    if not entry.solutions:
        raise ValueError(f"{entry.name}: no reference .oms sibling to classify against")
    return load_solutions(entry)[0]


def _placement_keys(sol: Solution) -> list[tuple[str, int, int, int]]:
    """
    Placement identity keys: (blueprint or entrypoint shape, rotation, column, row).
    """
    return [
        (
            placement.builtin.name if placement.builtin is not None else placement.template.name,
            placement.rotation,
            placement.q,
            placement.r,
        )
        for placement in sol.placements
    ]


def _program_keys(sol: Solution) -> tuple[list[tuple[int, int]], list[tuple[int, int, object]]]:
    """
    Program skeleton and full triplets, handles relabelled by sorted manipulator
    position so reordered-but-equivalent layouts compare equal.
    """
    placements = sol.placements
    ordinals = [
        index
        for index, placement in enumerate(placements)
        if placement.builtin is not None and placement.builtin.manipulator
    ]
    ordered = sorted(
        ordinals,
        key=lambda index: (
            placements[index].q,
            placements[index].r,
            placements[index].rotation,
            placements[index].builtin.name,
        ),
    )
    remap = {old: new for new, old in enumerate(ordered)}
    pairs = sorted((remap[action.handle], action.tick) for action in sol.actions)
    triplets = sorted((remap[action.handle], action.tick, action.action) for action in sol.actions)
    return pairs, triplets
