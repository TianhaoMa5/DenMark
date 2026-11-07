"""Regression coverage for complete-block semantic generation and scoring."""

from types import SimpleNamespace

import pytest
import torch

from denmark.baselines.block_semantic.cli import detection_ids
from denmark.baselines.block_semantic.embedding import (
    alimark_alignment_score,
    cadmark_evidence,
    cadmark_partition,
    cadmark_pivots,
)
from denmark.baselines.block_semantic.model import METHODS, BlockConfig, BlockSampler, BlockWatermark
from denmark.baselines.block_semantic.swan import Swan


class Tokenizer:
    all_special_ids = [9]

    def decode(self, ids, **kwargs):
        return "".join(chr(ord("a") + i) for i in ids if i != 9)

    def convert_ids_to_tokens(self, ids):
        return list(self.decode(ids))

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(ch) - ord("a") for ch in text]}


class DiffusionModel:
    def __init__(self):
        self.calls = []

    def __call__(self, ids):
        offset = 2 * (len(self.calls) % 2)
        self.calls.append(ids.clone())
        logits = torch.full((*ids.shape, 10), -100.0)
        for index in range(len(ids)):
            logits[index, :, index + offset] = 100.0
        return SimpleNamespace(logits=logits)


def embed(texts):
    alphabet = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1]], dtype=torch.float32)
    return torch.stack([
        alphabet[[min(ord(ch) - ord("a"), 4) for ch in text if ch.isalpha()]].mean(dim=0)
        for text in texts
    ])


@pytest.mark.parametrize("method", METHODS)
def test_full_blocks_generate_and_detect(method):
    cfg = BlockConfig(
        gen_length=4, block_size=2, steps=2, num_candidates=4,
        candidate_batch_size=2, temperature=0, remasking="ar",
    )
    model = DiffusionModel()
    tokenizer = Tokenizer()
    sampler = BlockSampler(cfg, model, tokenizer, 9, "cpu")
    swan = Swan(["bb"], lambda text: text, lambda graph, target: float(graph == target))
    watermark = BlockWatermark(
        method, cfg, tokenizer, embed=embed, embedding_dim=2,
        swan=swan if method == "swan" else None,
    )
    text, ids, trace = watermark.generate(
        sampler, [4], seed=42, swan_prompt=lambda target: [0, *tokenizer(target)["input_ids"]]
    )
    assert len(model.calls) == 4
    assert len(ids) == 4
    assert text == tokenizer.decode(ids)
    assert [row["n_valid_candidates"] for row in trace] == [4, 4]
    prefix_length = 3 if method == "swan" else 1
    assert model.calls[2][0, prefix_length:prefix_length + 2].tolist() == ids[:2]

    result = watermark.detect(ids, prompt_ids=[4], sampler=sampler)
    assert torch.isfinite(torch.tensor(result["score"]))
    assert result["n_active_blocks"] == 2
    if method == "cad-mark":
        assert len(model.calls) == 8
        assert model.calls[6][0, 1:3].tolist() == ids[:2]
    elif method == "swan":
        assert ids == [1, 1, 1, 1]
        assert result["green_blocks"] == 2
        assert all(row["accepted"] for row in trace)
    elif method == "alimark":
        assert result["n_segmentations"] == 4
    assert watermark.detect([9, 9], prompt_ids=[4], sampler=sampler)["n_active_blocks"] == 0


def test_cadmark_uses_sequential_boundaries_and_shared_soft_evidence():
    reference = torch.tensor([[-1.0, -0.2], [-0.5, 0.2], [0.5, 0.6], [1.0, 0.8]])
    bits = torch.tensor([True, False])
    survivors, medians = cadmark_partition(reference, bits)
    assert survivors.tolist() == [2]
    assert medians.tolist() == pytest.approx([0.0, 0.7])
    hit, variance = cadmark_evidence(reference, reference[2], bits)
    miss, _ = cadmark_evidence(reference, reference[0], bits)
    assert hit > miss
    assert variance > 0
    tied = torch.zeros_like(reference)
    assert cadmark_partition(tied, bits)[1].numel() == 0
    assert cadmark_evidence(tied, tied[0], bits) == (0.0, 0.0)


def test_cadmark_pivots_keep_context_and_requested_width():
    context = torch.tensor([1.0, 0.0])
    assert torch.equal(cadmark_pivots(context, torch.tensor([[0.0, 1.0]])), context[None])
    assert torch.allclose(cadmark_pivots(context, torch.eye(2)), torch.eye(2))


def test_alimark_recovers_alignment_after_one_inserted_block():
    secret = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
    observed = torch.cat([secret[:1], torch.tensor([[1, 0]]), secret[1:]])
    assert alimark_alignment_score(secret, secret) == 1.0
    assert alimark_alignment_score(observed, secret) == pytest.approx(0.8)


def test_detection_uses_explicit_attack_field_over_stale_tokens():
    row = {"text": "aaa", "token_ids": [0, 0, 0], "text_gpt_rewrite_document": "bb"}
    assert detection_ids(row, Tokenizer(), False) == [0, 0, 0]
    assert detection_ids(row, Tokenizer(), False, "text_gpt_rewrite_document") == [1, 1]


def test_cli_round_trips_generated_jsonl(tmp_path, monkeypatch):
    import json
    import sys

    import transformers

    from denmark import __main__ as cli
    from denmark.core import model

    tokenizer = Tokenizer()
    encoder = SimpleNamespace(config=SimpleNamespace(hidden_size=2))
    encoder.to = lambda _: encoder
    encoder.eval = lambda: encoder
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **kw: encoder)
    monkeypatch.setattr(model, "load_generator_model", lambda *a, **kw: DiffusionModel())
    monkeypatch.setattr(model, "encode_texts", lambda texts, *a: embed(texts))
    monkeypatch.delitem(sys.modules, "denmark.baselines.block_semantic.cli", raising=False)

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"sample_id":"example","prompt_ids":[4]}\n')
    generated = tmp_path / "generated.jsonl"
    scores = tmp_path / "scores.jsonl"
    options = [
        "--method", "semfield-pl", "--model", "test-generator", "--encoder", "test-encoder",
        "--device", "cpu", "--gen_length", "4", "--block_size", "2", "--steps", "2",
        "--num_candidates", "4", "--candidate_batch_size", "2", "--temperature", "0",
        "--remasking", "ar", "--mask_id", "9",
    ]
    for mode, source, output in [
        ("generate", prompts, generated), ("detect", generated, scores),
    ]:
        monkeypatch.setattr(sys, "argv", [
            "denmark", "baseline", "block-semantic", mode, *options,
            "--input", str(source), "--output", str(output),
        ])
        cli.main()
    row = json.loads(scores.read_text())
    assert row["sample_id"] == "example"
    assert row["prompt_ids"] == [4]
    assert len(row["token_ids"]) == 4
    assert row["detection"]["score"] >= 0
