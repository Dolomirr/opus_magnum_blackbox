"""Error taxonomy: hand-computed failure classes, scripted-policy probe over a fixture checkpoint."""

from dataclasses import asdict, replace

import pytest
import torch
from opus_magnum_sym import Hex
from test_train import TINY, _ScriptPolicy

from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.families import (
    anywhere_reference_oms,
    anywhere_reference_solution,
    anywhere_reference_tokens,
    anywhere_spec,
)
from opus_magnum_blackbox.grammar import Solution
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import RolloutBatch
from opus_magnum_blackbox.taxonomy import (
    batch_taxonomy,
    classify_episode,
    error_taxonomy,
    format_table,
    load_policy,
    pair_distance,
)
from opus_magnum_blackbox.vocab import INSERTED_BUILTINS, Vocab, insertion_permutation


def _bank(tmp_path):
    """Two anywhere-family entries with on-disk reference .oms files: d1 (ring 1) and d3."""
    specs = (anywhere_spec(Hex(0, 0), Hex(0, 1)), anywhere_spec(Hex(0, 0), Hex(0, 3)))
    entries = []
    for index, (spec, name) in enumerate(zip(specs, ("bank/eval/v000", "bank/eval/v001"), strict=True)):
        oms = tmp_path / f"ref{index}.oms"
        oms.write_text(anywhere_reference_oms(spec))
        entries.append(LevelEntry(source=spec, cost_max=50, name=name, solutions=(oms,)))
    return entries, specs


def _scripted(env: OpusEnv, scripts: dict) -> _ScriptPolicy:
    """`_ScriptPolicy` keyed by padded-encoder bytes, one script per spec."""
    tokens = {}
    for spec, sequence in scripts.items():
        padded, _ = env.tokenizer.encode_level_spec(spec)
        tokens[padded.tobytes()] = sequence
    return _ScriptPolicy(len(env.tokenizer.vocab), tokens)


def _drop_first_triplet(tokens: list[int], vocab: Vocab) -> list[int]:
    """Program without its first (handle, tick, action) triplet: grammar-valid, but the skeleton differs."""
    start = tokens.index(vocab.special_id("[START_PROGRAM]"))
    return tokens[: start + 1] + tokens[start + 4 :]


def _duplicate_first_tick(tokens: list[int], vocab: Vocab) -> list[int]:
    """Second triplet re-using the first's tick: undecodable (a tick may be used at most once per handle)."""
    start = tokens.index(vocab.special_id("[START_PROGRAM]"))
    out = list(tokens)
    out[start + 5] = out[start + 2]
    return out


def _rotated_arm_tokens(spec) -> list[int]:
    """Reference sequence with the arm's rest rotation turned one step: same cell and blueprint, wrong base facing."""
    sol = anywhere_reference_solution(spec)
    arm = next(placement for placement in sol.placements if placement.builtin is not None and placement.builtin.manipulator)
    placements = [
        replace(placement, rotation=(placement.rotation + 1) % 6) if placement is arm else placement
        for placement in sol.placements
    ]
    return OpusTokenizer().encode_solution(placements, sol.actions, spec)


# --- pure classification ---


def test_classify_episode_hand_computed():
    vocab = Vocab()
    spec = anywhere_spec(Hex(0, 0), Hex(0, 1))
    reference = anywhere_reference_tokens(spec)
    sol = anywhere_reference_solution(spec)
    arm = next(placement for placement in sol.placements if placement.builtin is not None and placement.builtin.manipulator)
    tokenizer = OpusTokenizer()

    assert pair_distance(spec) == 1
    assert classify_episode(True, reference, reference, vocab, spec) == "solved"  # solved wins over every difference
    assert classify_episode(False, reference, reference, vocab, spec) == "other"  # unsolved yet identical: unattributable
    assert classify_episode(False, _duplicate_first_tick(reference, vocab), reference, vocab, spec) == "invalid"

    l2 = vocab.builtin(vocab.builtin_id("arm_l2"))
    swapped = Solution(
        placements=[
            replace(placement, builtin=l2, template=l2.template) if placement is arm else placement
            for placement in sol.placements
        ],
        actions=sol.actions,
        submitted=True,
        length=0,
    )
    blueprint_tokens = tokenizer.encode_solution(swapped.placements, swapped.actions, spec)
    assert classify_episode(False, blueprint_tokens, reference, vocab, spec) == "wrong_blueprint"

    rotated = Solution(
        placements=[
            replace(placement, rotation=(placement.rotation + 1) % 6) if placement is arm else placement
            for placement in sol.placements
        ],
        actions=sol.actions,
        submitted=True,
        length=0,
    )
    base_tokens = tokenizer.encode_solution(rotated.placements, rotated.actions, spec)
    assert classify_episode(False, base_tokens, reference, vocab, spec) == "wrong_base"

    assert classify_episode(False, _drop_first_triplet(reference, vocab), reference, vocab, spec) == "wrong_program"

    first, second = sol.actions[0], sol.actions[1]
    assert first.action != second.action
    swapped_actions = [replace(first, action=second.action), replace(second, action=first.action), *sol.actions[2:]]
    tick_tokens = tokenizer.encode_solution(sol.placements, swapped_actions, spec)
    assert classify_episode(False, tick_tokens, reference, vocab, spec) == "wrong_ticks"


# --- scripted-policy probe over a fixture bank ---


def test_probe_fixture_checkpoint_matches_hand_computed(tmp_path):
    entries, (spec_a, spec_b) = _bank(tmp_path)
    env = OpusEnv(entries[:1])
    references = {spec: anywhere_reference_tokens(spec) for spec in (spec_a, spec_b)}

    correct = _scripted(env, references)
    tables = error_taxonomy(correct, entries, "cpu")
    assert tables == {
        "bank/eval": {
            pair_distance(spec_a): {"solved": 1},
            pair_distance(spec_b): {"solved": 1},
        }
    }

    broken = _scripted(
        env,
        {
            spec_a: _drop_first_triplet(references[spec_a], env.tokenizer.vocab),  # no grab: unsolved, skeleton differs
            spec_b: _rotated_arm_tokens(spec_b),
        },
    )
    tables = error_taxonomy(broken, entries, "cpu")
    assert tables == {
        "bank/eval": {
            pair_distance(spec_a): {"wrong_program": 1},
            pair_distance(spec_b): {"wrong_base": 1},
        }
    }


def test_load_policy_roundtrip(tmp_path):
    entries, _ = _bank(tmp_path)
    env = OpusEnv(entries[:1])
    source = OpusPolicy(env.tokenizer.config, TINY)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": source.state_dict(), "model_config": asdict(TINY), "iteration": 7}, path)

    policy, payload = load_policy(path, env.tokenizer.config)
    assert payload["iteration"] == 7
    assert policy.model_config == TINY
    for name, tensor in policy.state_dict().items():
        assert torch.equal(tensor, source.state_dict()[name]), name


def test_load_policy_remaps_pre_insertion_checkpoint(tmp_path):
    """
    A checkpoint from before the `INSERTED_BUILTINS` insertion (vocab 163 -> 168)
    loads by index permutation, exactly as `Trainer.warm_start` remaps it; old
    token rows move intact and the remap lands in the payload.
    """
    entries, _ = _bank(tmp_path)
    env = OpusEnv(entries[:1])
    vocab = env.tokenizer.vocab
    old_size = len(vocab) - len(INSERTED_BUILTINS)
    source = OpusPolicy(env.tokenizer.config, TINY)
    perm = torch.tensor(insertion_permutation(old_size, vocab))
    shrunk = dict(source.state_dict())
    shrunk["embed.weight"] = torch.cat([source.state_dict()["embed.weight"][perm], source.state_dict()["embed.weight"][-1:]])
    shrunk["policy_head.weight"] = source.state_dict()["policy_head.weight"][perm]
    shrunk["policy_head.bias"] = source.state_dict()["policy_head.bias"][perm]
    path = tmp_path / "old_ckpt.pt"
    torch.save({"model_state": shrunk, "model_config": asdict(TINY), "iteration": 3}, path)

    policy, payload = load_policy(path, env.tokenizer.config)
    assert payload["vocab_remap"] == {"from": old_size, "to": len(vocab)}
    for name in ("arm", "arm_piston"):  # pre-insertion rows move intact
        row = source.state_dict()["embed.weight"][vocab.builtin_id(name)]
        assert torch.equal(policy.state_dict()["embed.weight"][vocab.builtin_id(name)], row), name


def test_format_table_layout():
    table = {1: {"solved": 3, "wrong_base": 1}, 2: {"wrong_blueprint": 2}}
    lines = format_table(table).splitlines()
    assert lines[0].split() == ["class", "d1", "d2", "total"]
    assert next(line for line in lines if line.startswith("solved")).split() == ["solved", "3", "0", "3"]
    assert next(line for line in lines if line.startswith("wrong_blueprint")).split() == ["wrong_blueprint", "0", "2", "2"]
    assert next(line for line in lines if line.startswith("wrong_base")).split() == ["wrong_base", "1", "0", "1"]


def test_probe_missing_reference_raises(tmp_path):
    """A bank entry without a reference .oms cannot be classified: loud error, not a silent skip."""
    spec = anywhere_spec(Hex(0, 0), Hex(0, 1))
    entry = LevelEntry(source=spec, cost_max=50, name="bank/eval/v000")
    batch = RolloutBatch(
        encoder_input=torch.zeros(1, 8, dtype=torch.long),
        encoder_length=torch.ones(1, dtype=torch.long),
        actions=torch.zeros(1, 4, dtype=torch.long),
        decoder_length=torch.ones(1, dtype=torch.long),
        action_masks=torch.zeros(1, 4, 16, dtype=torch.bool),
        old_logprobs=torch.zeros(1, 4),
        old_values=torch.zeros(1, 4),
        returns=torch.zeros(1, 4),
        valid=torch.zeros(1, 4, dtype=torch.bool),
        episode_solved=torch.zeros(1, dtype=torch.bool),
        level_names=["bank/eval/v000"],
        mean_return=0.0,
        solve_rate=0.0,
        mean_length=1.0,
    )
    with pytest.raises(ValueError, match="no reference"):
        batch_taxonomy(batch, [entry], OpusEnv([entry]).tokenizer.vocab)
