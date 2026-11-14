"""CLI eval: bank/group names resolve through the corpus and the rate is exhaustive per variant."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from conftest import make_spec

from opus_magnum_blackbox import cli
from opus_magnum_blackbox.config import ModelConfig
from opus_magnum_blackbox.env import LevelEntry, OpusEnv
from opus_magnum_blackbox.model import OpusPolicy

TINY = ModelConfig(d_model=32, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64)


def _checkpoint(tmp_path: Path, env: OpusEnv) -> Path:
    torch.manual_seed(0)
    policy = OpusPolicy(env.tokenizer.config, TINY)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": policy.state_dict(), "model_config": asdict(TINY)}, path)
    return path


def test_eval_group_name_reports_exhaustive_rate(tmp_path, monkeypatch, capsys):
    entries = [
        LevelEntry(source=make_spec(), cost_max=50, name="bank/v00"),
        LevelEntry(source=make_spec(steps=5), cost_max=50, name="bank/v01"),
    ]
    monkeypatch.setattr(cli, "resolve_entries", lambda names: entries)
    checkpoint = _checkpoint(tmp_path, OpusEnv(entries))
    argv = ["eval", str(checkpoint), "--level", "bank", "--device", "cpu"]

    assert cli.main(argv) == 0
    first = capsys.readouterr().out
    assert "exhaustive solve rate" in first
    assert "/2 = " in first  # denominator is the group's variant count, not an episode count

    cli.main(argv)
    assert capsys.readouterr().out == first  # greedy per variant: deterministic and repeatable


def test_eval_unknown_level_exits_with_corpus_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "resolve_entries",
        lambda names: (_ for _ in ()).throw(ValueError("unknown levels: ['nope']; corpus has: []")),
    )
    checkpoint = _checkpoint(tmp_path, OpusEnv([LevelEntry(source=make_spec(), cost_max=50, name="bank/v00")]))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["eval", str(checkpoint), "--level", "nope", "--device", "cpu"])
    assert "unknown levels" in str(excinfo.value.code)


def test_generate_temperature_changes_sampling_but_stays_valid(tmp_path, monkeypatch):
    """
    `generate --temperature 2` fattens the sampling tail: different solutions than T = 1
    at a fixed seed, deterministic per temperature. Only grammar-valid, applicable
    sequences are written — the file exists only after decode + placement succeeded.
    """
    entry = LevelEntry(source=make_spec(), cost_max=50, name="tier0_test")
    monkeypatch.setattr(cli, "_find_entry", lambda name: entry)
    checkpoint = _checkpoint(tmp_path, OpusEnv([entry]))

    def generate(temperature: float, out_dir: Path) -> list[str]:
        argv = ["generate", str(checkpoint), "--count", "4", "--seed", "5", "--device", "cpu", "--out-dir", str(out_dir)]
        if temperature != 1.0:
            argv += ["--temperature", str(temperature)]
        assert cli.main(argv) == 0
        return [path.read_text() for path in sorted(out_dir.glob("tier0_test.*.oms"))]

    cold = generate(1.0, tmp_path / "cold")
    hot = generate(2.0, tmp_path / "hot")
    cold_again = generate(1.0, tmp_path / "cold-again")

    assert len(cold) >= 1
    assert len(hot) >= 1
    assert cold == cold_again  # fixed seed: reproducible
    assert cold != hot  # temperature widens the sampling tail
