"""`OpusPolicy` contract: shapes, causality, BOS isolation, masking, encoder padding."""

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed")

from opus_magnum_blackbox.config import ModelConfig
from opus_magnum_blackbox.model import OpusPolicy

TINY = ModelConfig(d_model=32, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64)


@pytest.fixture
def policy(tokenizer) -> OpusPolicy:
    torch.manual_seed(0)
    return OpusPolicy(tokenizer.config, TINY)


def make_batch(tokenizer, lengths=(3, 5)):
    """Random padded batch of two episodes with distinct decoder lengths."""
    vocab_size = len(tokenizer.vocab)
    gen = torch.Generator().manual_seed(0)
    return (
        torch.randint(0, vocab_size, (2, tokenizer.config.max_encoder_len), generator=gen),
        torch.tensor([tokenizer.config.max_encoder_len // 2] * 2),
        torch.randint(0, vocab_size, (2, 5), generator=gen),
        torch.tensor(lengths),
    )


def test_forward_shapes(policy, tokenizer):
    enc_in, enc_len, dec_hist, dec_len = make_batch(tokenizer)
    logits, values = policy.forward(enc_in, enc_len, dec_hist, dec_len)
    width = dec_len.max().item() + 1  # BOS-shifted decoder input, cropped to the used prefix
    assert logits.shape == (2, width, len(tokenizer.vocab))
    assert values.shape == (2, width)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


def test_forward_crops_padded_decoder_tail(policy, tokenizer):
    """Output width is max(decoder_length) + 1 even when the history is padded wider."""
    enc_in, enc_len, dec_hist, dec_len = make_batch(tokenizer, lengths=(2, 3))
    logits, values = policy.forward(enc_in, enc_len, dec_hist, dec_len)
    assert logits.shape == (2, 4, len(tokenizer.vocab))
    assert values.shape == (2, 4)


def test_padded_batch_matches_unpadded_rows(policy, tokenizer):
    """
    Next-token outputs gathered at decoder_length from a padded batch equal
    per-row unpadded batch-1 forwards (Stage 2.5 prefix-crop equivalence).
    """
    enc_in, enc_len, dec_hist, dec_len = make_batch(tokenizer)
    logits, values = policy.forward(enc_in, enc_len, dec_hist, dec_len)
    for row, length in enumerate(dec_len.tolist()):
        row_logits, row_values = policy.forward(
            enc_in[row : row + 1],
            enc_len[row : row + 1],
            dec_hist[row : row + 1, :length],
            dec_len[row : row + 1],
        )
        assert torch.allclose(logits[row, length], row_logits[0, length], atol=1e-4, rtol=0.0)
        assert torch.allclose(values[row, length], row_values[0, length], atol=1e-4, rtol=0.0)


def test_causality(policy, tokenizer):
    """Mutating history token j changes logits only at positions > j."""
    enc_in, enc_len, dec_hist, dec_len = make_batch(tokenizer)
    logits, _ = policy.forward(enc_in, enc_len, dec_hist, dec_len)

    j = 2
    mutated = dec_hist.clone()
    mutated[0, j] = (mutated[0, j] + 1) % len(tokenizer.vocab)
    mutated_logits, _ = policy.forward(enc_in, enc_len, mutated, dec_len)

    assert torch.allclose(logits[0, : j + 1], mutated_logits[0, : j + 1])
    assert not torch.allclose(logits[0, j + 1 :], mutated_logits[0, j + 1 :])


def test_bos_position_uses_only_encoder(policy, tokenizer):
    """Logits/values at position 0 are invariant to the whole history but react to the encoder."""
    enc_in, enc_len, dec_hist, dec_len = make_batch(tokenizer)
    logits, values = policy.forward(enc_in, enc_len, dec_hist, dec_len)

    fresh_hist = torch.randint(0, len(tokenizer.vocab), dec_hist.shape, generator=torch.Generator().manual_seed(1))
    logits2, values2 = policy.forward(enc_in, enc_len, fresh_hist, dec_len)
    assert torch.allclose(logits[0, 0], logits2[0, 0])
    assert torch.allclose(values[0, 0], values2[0, 0])

    mutated_enc = enc_in.clone()
    mutated_enc[0, 0] = (mutated_enc[0, 0] + 1) % len(tokenizer.vocab)
    logits3, _ = policy.forward(mutated_enc, enc_len, dec_hist, dec_len)
    assert not torch.allclose(logits[0, 0], logits3[0, 0])


def test_masked_distribution(policy, tokenizer):
    vocab_size = len(tokenizer.vocab)
    logits = torch.randn(3, 4, vocab_size, generator=torch.Generator().manual_seed(2))
    mask = torch.ones(3, 4, vocab_size, dtype=torch.bool)
    mask[0, :, : vocab_size // 2] = False
    mask[1, :, 1:] = False  # single valid token

    dist = policy.masked_distribution(logits, mask)
    probs = dist.probs
    assert (probs[~mask] == 0).all()
    assert (probs[mask] > 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(3, 4))

    actions = dist.sample()
    assert mask.gather(-1, actions.unsqueeze(-1)).all()


def test_encoder_padding_invariance(policy, tokenizer):
    """Outputs are invariant to encoder tokens beyond encoder_length."""
    enc_in, enc_len, dec_hist, dec_len = make_batch(tokenizer)
    logits, values = policy.forward(enc_in, enc_len, dec_hist, dec_len)

    mutated_enc = enc_in.clone()
    tail = torch.arange(enc_in.shape[1]).unsqueeze(0) >= enc_len.unsqueeze(1)
    mutated_enc[tail] = (mutated_enc[tail] + 7) % len(tokenizer.vocab)
    mutated_logits, mutated_values = policy.forward(mutated_enc, enc_len, dec_hist, dec_len)

    assert torch.allclose(logits, mutated_logits)
    assert torch.allclose(values, mutated_values)
