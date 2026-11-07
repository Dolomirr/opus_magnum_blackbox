"""
`OpusPolicy`: transformer policy network env's observations.

BOS is embedding index `vocab_size`, only model-internal;
never appears in actual token streams.

Decoder input is shifted action sequence: position `t` holds BOS for `t=0`,
or action token `a_{t-1}`.
Output position `t` is the policy distribution for action `a_t`
and estimate value of state `s_t`.
"""

import torch
from torch import nn

from opus_magnum_blackbox.config import ModelConfig, TokenizerConfig
from opus_magnum_blackbox.vocab import Vocab


class OpusPolicy(nn.Module):
    """
    Encoder-decoder transformer with policy and value heads.
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        model_config: ModelConfig | None = None,
    ) -> None:
        super().__init__()

        self.model_config = model_config or ModelConfig()
        self.vocab_size = len(Vocab(tokenizer_config.vocab))
        self.max_encoder_len = tokenizer_config.max_encoder_len
        self.max_decoder_len = tokenizer_config.max_decoder_len

        cfg = self.model_config

        # model internal BOS.
        self.embed = nn.Embedding(self.vocab_size + 1, cfg.d_model)
        self.encoder_pos = nn.Embedding(tokenizer_config.max_encoder_len, cfg.d_model)
        # decoder input is shifted, so its one position longer than history
        self.decoder_pos = nn.Embedding(tokenizer_config.max_decoder_len + 1, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.nhead,
            cfg.dim_feedforward,
            cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            cfg.encoder_layers,
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            cfg.d_model,
            cfg.nhead,
            cfg.dim_feedforward,
            cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            cfg.decoder_layers,
        )

        self.policy_head = nn.Linear(cfg.d_model, self.vocab_size)
        self.value_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        encoder_input: torch.Tensor,
        encoder_length: torch.Tensor,
        decoder_history: torch.Tensor,
        decoder_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Per-position logits `[B, T + 1, vocab_size]` and values `[B, T + 1]`.

        Output position `t` is the distribution for action `a_t` and the value
        of state `s_t`: at rollout time the next-token position to gather is
        `t = decoder_length`.

        Args:
            encoder_input:    [B, E] padded spec token ids
            encoder_length:   [B] true spec token counts
            decoder_history:  [B, T] padded action token ids
            decoder_length:   [B] tokens emitted so far

        Returns:
            Tuple:
                - [B, T + 1, vocab_size]
                - [B, T + 1]

        """
        enc_width = encoder_input.shape[1]
        dec_width = decoder_history.shape[1] + 1
        device = encoder_input.device
        enc_positions = torch.arange(enc_width, device=device)
        dec_positions = torch.arange(dec_width, device=device)

        enc_pad = enc_positions.unsqueeze(0) >= encoder_length.unsqueeze(1)
        memory = self.encoder(self.embed(encoder_input) + self.encoder_pos(enc_positions), src_key_padding_mask=enc_pad)

        # BOS-shift: position 0 holds BOS, position i holds history token i-1.
        bos = torch.full((decoder_history.shape[0], 1), self.vocab_size, dtype=torch.long, device=device)
        dec_input = torch.cat((bos, decoder_history), dim=1)
        # Position i is a valid input through i = decoder_length (holds a_{t-1}); beyond that is padding.
        dec_pad = dec_positions.unsqueeze(0) > decoder_length.unsqueeze(1)
        causal = torch.triu(torch.ones(dec_width, dec_width, dtype=torch.bool, device=device), diagonal=1)
        hidden = self.decoder(
            self.embed(dec_input) + self.decoder_pos(dec_positions),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=dec_pad,
            memory_key_padding_mask=enc_pad,
        )
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def masked_distribution(self, logits: torch.Tensor, action_mask: torch.Tensor) -> torch.distributions.Categorical:
        """
        Categorical mask for valid tokens.
        Invalid logits become `-inf` before softmax.
        """
        return torch.distributions.Categorical(
            logits=logits.masked_fill(
                ~action_mask.bool(),
                float("-inf"),
            ),
        )
