"""Stage-1/2 data path and behavior-cloning trainer: `.oms` files to validated token sequences, supervised batches, BC smoke."""

import json

import pytest
import torch
from conftest import known_solution_tokens, make_spec
from opus_magnum_sym import Level, StepActionType

from opus_magnum_blackbox.codec import apply_solution, decode_solution
from opus_magnum_blackbox.config import ModelConfig, PretrainConfig, TrainConfig
from opus_magnum_blackbox.corpus import LEVELS_DIR
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.grammar import Action, GrammarError, Placement, spec_model
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import Episode, build_batch, build_supervised
from opus_magnum_blackbox.solutions import SolutionLoadError, augment, load_solutions
from opus_magnum_blackbox.train import Trainer

TIER0_OML = LEVELS_DIR / "tier0_typed_delivery.oml"
TIER0_OMS = LEVELS_DIR / "tier0_typed_delivery.oms"

TINY = ModelConfig(d_model=32, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64)


def tier0_entry() -> LevelEntry:
    return LevelEntry(source=TIER0_OML, cost_max=50, name="tier0_typed_delivery", solutions=(TIER0_OMS,))


def test_encode_solution_matches_known_tokens(tokenizer, vocab, spec):
    model = spec_model(spec)
    arm = vocab.builtin(vocab.builtin_id("arm"))
    placements = [
        Placement(
            builtin=None,
            slot=model.slots.index("entrance_typed"),
            template=model.templates[0],
            rotation=1,
            q=3,
            r=2,
            pos=0,
        ),
        Placement(
            builtin=None, slot=model.slots.index("exit_typed"), template=model.templates[1], rotation=2, q=2, r=3, pos=0
        ),
        Placement(builtin=arm, slot=None, template=arm.template, rotation=0, q=3, r=3, pos=0),
    ]
    actions = [
        Action(handle=0, tick=tick, action=step, pos=0)
        for tick, step in [
            (1, StepActionType.GRAB),
            (2, StepActionType.ROTATE_CLW),
            (3, StepActionType.ROTATE_CLW),
            (4, StepActionType.ROTATE_CLW),
            (5, StepActionType.ROTATE_CLW),
            (6, StepActionType.RELEASE),
        ]
    ]
    assert tokenizer.encode_solution(placements, actions, spec) == known_solution_tokens(vocab)


def test_loaded_solution_is_known_and_solves(vocab):
    (tokens,) = load_solutions(tier0_entry())
    assert tokens == known_solution_tokens(vocab)

    level = Level.from_file(TIER0_OML)
    decoded = decode_solution(tokens, level.spec, vocab)
    assert not isinstance(decoded, GrammarError)
    assert apply_solution(decoded, level.spec, level).ok
    assert level.run().solved


def test_loaded_tokens_mask_valid_at_prefixes(masker):
    (tokens,) = load_solutions(tier0_entry())
    spec = Level.from_file(TIER0_OML).spec
    for position, token in enumerate(tokens):
        assert masker.mask(tokens[:position], spec)[token]


def test_augmented_variants_solve_and_are_deterministic(vocab, spec):
    tokens = known_solution_tokens(vocab)
    variants = augment(tokens, spec, seed=0, max_variants=6)
    assert variants
    assert len(variants) <= 6
    assert all(variant != tokens for variant in variants)
    for variant in variants:
        decoded = decode_solution(variant, spec, vocab)
        assert not isinstance(decoded, GrammarError)
        level = Level(spec)
        assert apply_solution(decoded, spec, level).ok
        assert level.run().solved
    assert augment(tokens, spec, seed=0, max_variants=6) == variants


def test_unsupported_command_raises(tmp_path):
    path = tmp_path / "unsupported.oms"
    path.write_text("# only place/action are supported\ncomp 1 2 ARM\n")
    entry = LevelEntry(source=make_spec(), cost_max=50, name="unsupported", solutions=(path,))
    with pytest.raises(SolutionLoadError, match="unsupported command 'comp'"):
        load_solutions(entry)


def _supervised_episodes(tokenizer, masker, vocab, spec, count: int) -> list[Episode]:
    tokens = known_solution_tokens(vocab)
    padded, encoder_length = tokenizer.encode_level_spec(spec)
    return [
        Episode(
            encoder_input=padded,
            encoder_length=encoder_length,
            tokens=tokens,
            masks=[masker.mask(tokens[:t], spec) for t in range(len(tokens))],
            logprobs=[0.0] * len(tokens),
            values=[0.0] * len(tokens),
            rewards=[0.0] * len(tokens),
            returns=[0.0] * len(tokens),
            solved=True,
            terminated=True,
        )
        for _ in range(count)
    ]


def test_build_supervised_aligns_with_build_batch(tokenizer, masker, vocab, spec):
    episodes = _supervised_episodes(tokenizer, masker, vocab, spec, count=2)
    rl = build_batch(episodes, vocab.pad_id)
    bc = build_supervised([episode.tokens for episode in episodes], spec, tokenizer, masker)

    assert bc.actions.shape == rl.actions.shape
    assert bc.action_masks.shape == rl.action_masks.shape
    for key in ("encoder_input", "encoder_length", "actions", "decoder_length", "action_masks", "valid"):
        assert torch.equal(getattr(bc, key), getattr(rl, key))
    for key in ("old_logprobs", "old_values", "returns"):
        assert torch.equal(getattr(bc, key), torch.zeros_like(getattr(rl, key)))
    assert bc.mean_return == bc.solve_rate == 0.0
    assert bc.mean_length == rl.mean_length


def make_bc_setup(tmp_path, **config_overrides):
    env = OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")])
    config = TrainConfig(
        iterations=1,
        episodes_per_iteration=2,
        eval_interval=10,
        eval_episodes=2,
        checkpoint_interval=10,
        device="cpu",
        run_dir=str(tmp_path),
        **config_overrides,
    )
    torch.manual_seed(0)
    policy = OpusPolicy(env.tokenizer.config, TINY)
    trainer = Trainer(policy, [env], config)
    batch = build_supervised([known_solution_tokens(env.tokenizer.vocab)], make_spec(), env.tokenizer, env.masker)
    return trainer, batch


def read_pretrain_entries(run_dir) -> list[dict]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if json.loads(line).get("stage") == "pretrain"]


def test_pretrain_smoke_loss_decreases_and_resumes(tmp_path):
    trainer, batch = make_bc_setup(tmp_path)
    config = PretrainConfig(epochs=3, batch_size=32, eval_interval=1, early_stop_solve_rate=1.0, early_stop_loss=0.0)
    trainer.pretrain([batch], config)

    entries = read_pretrain_entries(trainer.run_dir)
    assert [entry["iteration"] for entry in entries] == [1, 2, 3]
    losses = [entry["loss"] for entry in entries]
    assert losses[0] > losses[1] > losses[2]

    checkpoint = trainer.run_dir / "latest.pt"
    payload = torch.load(checkpoint, weights_only=False)
    resumed = Trainer.resume(checkpoint, [OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")])])
    for key, value in payload["model_state"].items():
        assert torch.equal(resumed.policy.state_dict()[key], value)
    # BC stage lr must not leak into the restored RL optimizer state.
    assert resumed.optimizer.param_groups[0]["lr"] == resumed.train_config.lr
