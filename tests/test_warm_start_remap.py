"""
Warm-start vocab permutation: pre-insertion checkpoints remap exactly.

The arm_l2..arm_l6 insertion shifted every post-builtin token id by +5. A
checkpoint from before it loads into the current policy as the exact index
permutation of embedding rows and policy-head columns, with fresh small values
on the inserted ids — so greedy behavior on old tokens is bit-identical.
"""

import json
from dataclasses import asdict

import pytest
import torch
from conftest import make_spec
from test_vocab import PRE_INSERTION_BUILTINS

from opus_magnum_blackbox import vocab as vocab_module
from opus_magnum_blackbox.codec import OpusTokenizer
from opus_magnum_blackbox.config import TrainConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import collect_episode
from opus_magnum_blackbox.train import Trainer
from opus_magnum_blackbox.vocab import INSERTED_BUILTINS, Vocab, insertion_permutation

ENTRY = LevelEntry(source=make_spec(), cost_max=100, name="tier0_like")


def _old_world(monkeypatch, tmp_path, seed=5):
    """
    Build an old-vocabulary env and random policy, save its checkpoint, then
    restore the current registry; returns (checkpoint path, policy, env).
    """
    by_name = {b.name: b for b in vocab_module.BUILTINS}
    monkeypatch.setattr(vocab_module, "BUILTINS", tuple(by_name[name] for name in PRE_INSERTION_BUILTINS))
    env = OpusEnv([ENTRY])
    torch.manual_seed(seed)
    policy = OpusPolicy(env.tokenizer.config)
    path = tmp_path / "pre_insertion.pt"
    torch.save({"iteration": 1, "model_state": policy.state_dict(), "model_config": asdict(policy.model_config)}, path)
    monkeypatch.undo()
    return path, policy, env


def _trainer(env, tmp_path) -> Trainer:
    config = TrainConfig(run_dir=str(tmp_path / "run"), device="cpu")
    return Trainer(OpusPolicy(env.tokenizer.config), [env], config)


def _greedy_episode(env, policy):
    return collect_episode(env, policy, "cpu", greedy=True)


def test_warm_start_remaps_checkpoint_exactly(monkeypatch, tmp_path):
    path, old_policy, _ = _old_world(monkeypatch, tmp_path)
    vocab = Vocab()
    old_size = len(vocab) - len(INSERTED_BUILTINS)
    perm = insertion_permutation(old_size, vocab)

    trainer = _trainer(OpusEnv([ENTRY]), tmp_path)
    trainer.warm_start(path)
    new_state = trainer.policy.state_dict()
    old_state = old_policy.state_dict()

    # Non-vocab parameters are untouched by the remap.
    for key, value in old_state.items():
        if key in ("embed.weight", "policy_head.weight", "policy_head.bias"):
            continue
        assert torch.equal(new_state[key], value), key

    # Vocab-sized parameters follow the permutation; the model-internal BOS row maps to BOS.
    assert torch.equal(new_state["embed.weight"][torch.tensor([*perm, len(vocab)])], old_state["embed.weight"])
    head_index = torch.tensor(perm)
    assert torch.equal(new_state["policy_head.weight"][head_index], old_state["policy_head.weight"])
    assert torch.equal(new_state["policy_head.bias"][head_index], old_state["policy_head.bias"])

    # Inserted ids hold fresh small deterministic values from the train seed.
    fresh_ids = torch.tensor([vocab.builtin_id(name) for name in INSERTED_BUILTINS])
    d_model = trainer.policy.model_config.d_model
    generator = torch.Generator().manual_seed(trainer.train_config.seed)
    expected_embed = torch.randn(len(INSERTED_BUILTINS), d_model, generator=generator) * 0.02
    expected_weight = torch.randn(len(INSERTED_BUILTINS), d_model, generator=generator) * 0.02
    expected_bias = torch.randn(len(INSERTED_BUILTINS), generator=generator) * 0.02
    assert torch.equal(new_state["embed.weight"][fresh_ids], expected_embed)
    assert torch.equal(new_state["policy_head.weight"][fresh_ids], expected_weight)
    assert torch.equal(new_state["policy_head.bias"][fresh_ids], expected_bias)


def test_remapped_checkpoint_greedy_episode_is_bit_identical(monkeypatch, tmp_path):
    path, old_policy, old_env = _old_world(monkeypatch, tmp_path)
    episode_old = _greedy_episode(old_env, old_policy)
    perm = insertion_permutation(old_policy.vocab_size, Vocab())

    env = OpusEnv([ENTRY])
    trainer = _trainer(env, tmp_path)
    trainer.warm_start(path)
    episode_new = _greedy_episode(env, trainer.policy)

    assert episode_new.solved == episode_old.solved
    assert episode_new.terminated == episode_old.terminated
    assert episode_new.dead_end == episode_old.dead_end
    assert episode_new.rewards == episode_old.rewards
    assert episode_new.tokens == [perm[token] for token in episode_old.tokens]
    old_encoder = episode_old.encoder_input[: episode_old.encoder_length]
    new_encoder = episode_new.encoder_input[: episode_new.encoder_length]
    assert new_encoder.tolist() == [perm[int(token)] for token in old_encoder]


def test_warm_start_metrics_row_records_remap(monkeypatch, tmp_path):
    path, _policy, _env = _old_world(monkeypatch, tmp_path)
    trainer = _trainer(OpusEnv([ENTRY]), tmp_path)
    trainer.warm_start(path)
    trainer._log_warm_start(str(path))

    row = json.loads((trainer.run_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert row["warm_start"] == str(path)
    assert row["vocab_remap"] == {
        "from": len(Vocab()) - len(INSERTED_BUILTINS),
        "to": len(Vocab()),
        "inserted": list(INSERTED_BUILTINS),
    }


def test_warm_start_rejects_incompatible_vocabs(tmp_path):
    policy = OpusPolicy(OpusTokenizer().config)
    vocab_size = policy.vocab_size
    config = asdict(policy.model_config)
    path = tmp_path / "incompatible.pt"

    inconsistent = dict(policy.state_dict())
    inconsistent["policy_head.weight"] = inconsistent["policy_head.weight"][3:]
    torch.save({"iteration": 1, "model_state": inconsistent, "model_config": config}, path)
    trainer = _trainer(OpusEnv([ENTRY]), tmp_path)
    with pytest.raises(ValueError, match="inconsistent vocab-sized shapes"):
        trainer.warm_start(path)

    wrong_total = dict(policy.state_dict())
    wrong_total["embed.weight"] = wrong_total["embed.weight"][: vocab_size - 2]
    wrong_total["policy_head.weight"] = wrong_total["policy_head.weight"][3:]
    wrong_total["policy_head.bias"] = wrong_total["policy_head.bias"][3:]
    torch.save({"iteration": 1, "model_state": wrong_total, "model_config": config}, path)
    with pytest.raises(ValueError, match="does not match the pre-insertion vocabulary"):
        trainer.warm_start(path)

    mismatch = dict(policy.state_dict())
    torch.save(
        {"iteration": 1, "model_state": mismatch, "model_config": {**config, "d_model": 32}}, tmp_path / "other.pt"
    )
    with pytest.raises(ValueError, match="does not match"):
        trainer.warm_start(tmp_path / "other.pt")
