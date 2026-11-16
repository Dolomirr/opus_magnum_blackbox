"""Stage C2 derived encoder features: spec side channel, env observation, model projection, batch plumbing."""

import pytest
import torch
from opus_magnum_sym import Hex

from opus_magnum_blackbox.config import ModelConfig, TrainConfig, encoder_feature_dim
from opus_magnum_blackbox.env import LevelEntry, OpusEnv, spec_encoder_features
from opus_magnum_blackbox.families import anywhere_spec
from opus_magnum_blackbox.model import OpusPolicy
from opus_magnum_blackbox.rollout import build_batch, build_supervised, collect_episode
from opus_magnum_blackbox.train import Trainer

TINY = ModelConfig(d_model=32, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64)

NEAR = (Hex(0, 0), Hex(1, 0))  # d=1: every ring reaches the pair
FAR = (Hex(0, 0), Hex(5, 5))  # d=10: no ring can span the pair
MID = (Hex(0, 0), Hex(2, 1))  # d=3: only ring 2 spans the pair in-board (the ring-3 base falls off-board)


def _entry(spec, name="feature_probe") -> LevelEntry:
    return LevelEntry(source=spec, cost_max=50, name=name)


def test_feature_dim_and_validation():
    assert encoder_feature_dim(()) == 0
    assert encoder_feature_dim(("distance",)) == 1
    assert encoder_feature_dim(("ring_reach",)) == 3
    assert encoder_feature_dim(("distance", "ring_reach")) == 4
    with pytest.raises(ValueError, match="unknown encoder features"):
        encoder_feature_dim(("radius",))
    with pytest.raises(ValueError, match="duplicate"):
        encoder_feature_dim(("distance", "distance"))


def test_spec_features_hand_checked():
    for pair, expected in [
        (NEAR, [0.1, 1.0, 1.0, 1.0]),
        (FAR, [1.0, 0.0, 0.0, 0.0]),
        (MID, [0.3, 0.0, 1.0, 0.0]),
    ]:
        spec = anywhere_spec(*pair)
        assert spec_encoder_features(spec, ("distance", "ring_reach")).tolist() == pytest.approx(expected)
        assert spec_encoder_features(spec, ("distance",)).tolist() == pytest.approx([expected[0]])
        assert spec_encoder_features(spec, ("ring_reach",)).tolist() == expected[1:]


def test_env_observations_byte_identical_when_off():
    entry = _entry(anywhere_spec(*NEAR))
    default_obs, _ = OpusEnv([entry]).reset()
    explicit_obs, _ = OpusEnv([entry], encoder_features=()).reset()
    assert set(default_obs) == set(explicit_obs)  # no feature key either way
    for key, value in default_obs.items():
        assert (value == explicit_obs[key]).all() if hasattr(value, "shape") else value == explicit_obs[key]
    assert set(OpusEnv([entry]).observation_space.keys()) == set(OpusEnv([entry], encoder_features=()).observation_space.keys())


def test_observation_space_and_obs_carry_features_only_when_enabled():
    entry = _entry(anywhere_spec(*NEAR))
    assert "encoder_features" not in OpusEnv([entry]).observation_space.spaces

    env = OpusEnv([entry], encoder_features=("distance", "ring_reach"))
    assert env.observation_space.spaces["encoder_features"].shape == (4,)
    expected = spec_encoder_features(anywhere_spec(*NEAR), ("distance", "ring_reach"))
    obs, _ = env.reset()
    assert obs["encoder_features"].dtype.name == "float32"
    assert (obs["encoder_features"] == expected).all()
    for _ in range(5):  # spec-fixed: constant across the episode
        obs, *_ = env.step(int(obs["action_mask"].argmax()))
        assert (obs["encoder_features"] == expected).all()


def test_model_accepts_and_ignores_per_config():
    tokenizer_config = OpusEnv([_entry(anywhere_spec(*NEAR))]).tokenizer.config
    torch.manual_seed(0)
    off_policy = OpusPolicy(tokenizer_config, TINY)
    encoder_input = torch.randint(0, off_policy.vocab_size, (2, 8))
    encoder_length = torch.full((2,), 8, dtype=torch.long)
    decoder_history = torch.randint(0, off_policy.vocab_size, (2, 3))
    decoder_length = torch.full((2,), 3, dtype=torch.long)
    features = torch.rand(2, 4)

    plain = off_policy(encoder_input, encoder_length, decoder_history, decoder_length)
    ignored = off_policy(encoder_input, encoder_length, decoder_history, decoder_length, encoder_features=features)
    assert torch.equal(plain[0], ignored[0])  # off: accepted and ignored
    assert torch.equal(plain[1], ignored[1])

    on_policy = OpusPolicy(tokenizer_config, TINY, encoder_features=("distance", "ring_reach"))
    assert on_policy.encoder_feature_proj is not None
    with pytest.raises(ValueError, match="encoder features"):
        on_policy(encoder_input, encoder_length, decoder_history, decoder_length)
    with_features = on_policy(encoder_input, encoder_length, decoder_history, decoder_length, encoder_features=features)
    with_other = on_policy(
        encoder_input, encoder_length, decoder_history, decoder_length, encoder_features=torch.zeros(2, 4)
    )
    assert with_features[0].shape == plain[0].shape
    assert not torch.equal(with_features[0], with_other[0])  # the projection feeds the encoder


def test_supervised_and_collected_batches_carry_features():
    spec = anywhere_spec(*MID)
    features = spec_encoder_features(spec, ("distance", "ring_reach"))
    tokens = [0, 1, 2]  # batch content is irrelevant here, shape and broadcast matter
    batch = build_supervised([tokens, tokens], spec, OpusEnv([_entry(spec)]).tokenizer, OpusEnv([_entry(spec)]).masker)
    assert batch.encoder_features is None  # off: field stays None
    batch = build_supervised(
        [tokens, tokens], spec, OpusEnv([_entry(spec)]).tokenizer, OpusEnv([_entry(spec)]).masker, encoder_features=features
    )
    assert batch.encoder_features.shape == (2, 4)
    assert (batch.encoder_features == features).all()

    env = OpusEnv([_entry(spec)], encoder_features=("distance", "ring_reach"))
    policy = OpusPolicy(env.tokenizer.config, TINY, encoder_features=("distance", "ring_reach"))
    episode = collect_episode(env, policy, torch.device("cpu"), greedy=True)
    assert (episode.encoder_features == features).all()
    stacked = build_batch([episode, episode], env.tokenizer.vocab.pad_id)
    assert stacked.encoder_features.shape == (2, 4)
    assert (stacked.encoder_features == features).all()


def test_checkpoint_round_trips_feature_config(tmp_path):
    spec = anywhere_spec(*MID)
    names = ("distance", "ring_reach")
    env = OpusEnv([_entry(spec)], encoder_features=names)
    config = TrainConfig(
        iterations=1, episodes_per_iteration=2, eval_interval=10, eval_episodes=2, checkpoint_interval=10,
        device="cpu", run_dir=str(tmp_path),
    )
    torch.manual_seed(0)
    trainer = Trainer(OpusPolicy(env.tokenizer.config, TINY, encoder_features=names), [env], config)
    trainer.run()

    payload = torch.load(trainer.run_dir / "latest.pt", weights_only=False)
    assert payload["encoder_features"] == ["distance", "ring_reach"]
    resumed = Trainer.resume(trainer.run_dir / "latest.pt", [env])
    assert resumed.policy.encoder_features == names
    for key, value in trainer.policy.state_dict().items():
        assert torch.equal(resumed.policy.state_dict()[key], value)
