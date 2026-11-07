"""Generate and detect semantic watermark baselines on fixed token blocks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from denmark.baselines.block_best_of_k.model import _sample_candidate_microbatch
from denmark.baselines.block_semantic.embedding import (
    alimark_alignment_score,
    cadmark_evidence,
    cadmark_partition,
    cadmark_pivots,
    gaussian_direction,
    samark_score,
    select_alimark,
    select_samark,
    semfield_scores,
)
from denmark.baselines.block_semantic.swan import Swan
from denmark.baselines.pmark.model import (
    build_pmark_pivots,
    build_pmark_secret_bits,
    pmark_soft_matches,
    select_pmark_candidate,
)
from denmark.core.model import safe_decode
from denmark.core.scoring import block_decode


METHODS = ("swan", "pmark", "alimark", "samark", "cad-mark", "semfield", "semfield-pl")


@dataclass(frozen=True)
class BlockConfig:
    gen_length: int = 300
    block_size: int = 25
    steps: int | None = None
    num_candidates: int = 16
    candidate_batch_size: int = 4
    temperature: float = 0.7
    remasking: str = "low_confidence"
    generator_family: str = "llada"

    @property
    def num_blocks(self) -> int:
        return self.gen_length // self.block_size

    @property
    def steps_per_block(self) -> int:
        steps = self.gen_length if self.steps is None else self.steps
        return steps // self.num_blocks

    def __post_init__(self) -> None:
        if self.gen_length <= 0 or self.block_size <= 0 or self.gen_length % self.block_size:
            raise ValueError("block_size must divide positive gen_length")
        if self.steps is not None and (self.steps <= 0 or self.steps % self.num_blocks):
            raise ValueError("steps must be positive and divisible by the number of blocks")
        if min(self.num_candidates, self.candidate_batch_size) <= 0:
            raise ValueError("candidate counts must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.generator_family not in {"llada", "mmada"}:
            raise ValueError("fixed-block sampling supports llada and mmada")
        if self.remasking not in {"low_confidence", "random", "ar"}:
            raise ValueError("invalid remasking policy")


class BlockSampler:
    def __init__(self, config: BlockConfig, model, tokenizer, mask_id: int, device: str):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.device = device

    def sample(
        self, prompt_ids: list[int], committed: list[int]
    ) -> tuple[list[list[int]], list[str]]:
        cfg = self.config
        start = len(prompt_ids) + len(committed)
        end = start + cfg.block_size
        canvas = torch.full(
            (1, len(prompt_ids) + cfg.gen_length), self.mask_id,
            dtype=torch.long, device=self.device,
        )
        canvas[0, :start] = torch.tensor(prompt_ids + committed, dtype=torch.long, device=self.device)
        chunks = []
        for offset in range(0, cfg.num_candidates, cfg.candidate_batch_size):
            chunks.append(_sample_candidate_microbatch(
                canvas, self.model, self.mask_id, start, end, cfg.steps_per_block,
                cfg.temperature, cfg.remasking,
                min(cfg.candidate_batch_size, cfg.num_candidates - offset), cfg.generator_family,
            ))
        tokens = torch.cat(chunks).tolist()
        texts = [block_decode(block, 0, len(block), self.tokenizer) for block in tokens]
        return tokens, texts


class BlockWatermark:
    def __init__(
        self,
        method: str,
        config: BlockConfig,
        tokenizer,
        *,
        embed: Callable[[list[str]], torch.Tensor] | None = None,
        embedding_dim: int | None = None,
        num_channels: int = 2,
        pivot_seed: int = 42,
        secret_seed: int = 0,
        context_window: int = 3,
        swan: Swan | None = None,
    ):
        if method not in METHODS:
            raise ValueError(f"unsupported block watermark method {method!r}")
        self.method = method
        self.config = config
        self.tokenizer = tokenizer
        self.embed = embed
        self.context_window = context_window
        self.swan = swan
        self.secret_seed = secret_seed
        if method == "swan":
            if swan is None:
                raise ValueError("SWAN requires an AMR backend")
            return
        if embed is None or not embedding_dim:
            raise ValueError("this method requires an embedding encoder")
        if method.startswith("semfield"):
            self.direction = gaussian_direction(embedding_dim, secret_seed)
            return
        if method == "cad-mark" and (
            context_window < 1 or num_channels >= config.num_candidates.bit_length()
        ):
            raise ValueError("CAD-Mark requires a positive window and at least 2**channels candidates")
        self.pivots = build_pmark_pivots(embedding_dim, num_channels, pivot_seed)
        # AliMark searches longer prefixes after testing split/merge alternatives.
        self.bits = build_pmark_secret_bits(3 * config.num_blocks, num_channels, secret_seed)

    def _encode(self, texts: list[str]) -> torch.Tensor:
        embeddings = self.embed(texts).detach().float().cpu()
        if not torch.isfinite(embeddings).all():
            raise ValueError("encoder returned non-finite embeddings")
        return F.normalize(embeddings, dim=-1)

    def _cad_pivots(self, prefix: list[str], prompt_ids: list[int]) -> torch.Tensor:
        context = " ".join(filter(None, prefix[-self.context_window:]))
        if not context:
            context = block_decode(prompt_ids, 0, len(prompt_ids), self.tokenizer)
        embedding = self._encode([context])[0]
        return cadmark_pivots(embedding, self.pivots)

    @torch.no_grad()
    def generate(
        self,
        sampler: BlockSampler,
        prompt_ids: list[int],
        *,
        seed: int,
        swan_prompt: Callable[[str], list[int]] | None = None,
    ) -> tuple[str, list[int], list[dict]]:
        if self.method == "swan" and swan_prompt is None:
            raise ValueError("SWAN generation requires a template-conditioned prompt builder")
        rng = random.Random(seed)
        sign_rng = random.Random(seed ^ self.secret_seed)
        committed = []
        prefix = []
        previous_embeddings = []
        diagnostics = []
        aggregate = None
        polarity = 1.0
        if self.method == "samark":
            prompt_text = block_decode(prompt_ids, 0, len(prompt_ids), self.tokenizer)
            vocabulary = set(prompt_text.lower().split())
        for block_id in range(self.config.num_blocks):
            target = rng.choice(self.swan.bank) if self.swan else None
            proposal_prompt = swan_prompt(target) if target is not None else prompt_ids
            tokens, texts = sampler.sample(proposal_prompt, committed)
            valid = [index for index, text in enumerate(texts) if text]
            selected = 0
            details = {}
            if valid:
                candidates = [texts[index] for index in valid]
                if self.swan:
                    index, accepted = self.swan.select(candidates, target)
                    details["accepted"] = accepted
                else:
                    embeddings = self._encode(candidates)
                    if self.method.startswith("semfield"):
                        totals = embeddings if aggregate is None else embeddings + aggregate
                        scores = semfield_scores(totals, self.direction)
                        first_locked = self.method == "semfield-pl" and aggregate is None
                        index = int((scores.abs() if first_locked else polarity * scores).argmax())
                        aggregate = totals[index]
                        if first_locked and scores[index] < 0:
                            polarity = -1.0
                    elif self.method == "cad-mark":
                        pivots = self._cad_pivots(prefix, prompt_ids)
                        survivors, medians = cadmark_partition(
                            embeddings @ pivots.T, self.bits[block_id]
                        )
                        index = rng.choice(survivors.tolist())
                        details["channels_used"] = len(medians)
                    else:
                        projections = embeddings @ self.pivots.T
                        if self.method == "pmark":
                            index, _, _ = select_pmark_candidate(
                                projections, self.bits[block_id], [True] * len(candidates), rng=rng
                            )
                        elif self.method == "alimark":
                            index = select_alimark(projections, self.bits[block_id], rng)
                        else:
                            if not previous_embeddings:
                                signs = torch.tensor(sign_rng.choices(
                                    [True, False], k=self.pivots.shape[0]
                                ))
                            index = select_samark(
                                embeddings, projections, signs, candidates, " ".join(prefix),
                                vocabulary, previous_embeddings, rng,
                            )
                            previous_embeddings.append(embeddings[index])
                            vocabulary.update(candidates[index].lower().split())
                selected = valid[index]
            committed.extend(tokens[selected])
            prefix.append(texts[selected])
            diagnostics.append({
                "block_id": block_id, "selected_candidate_index": selected,
                "n_valid_candidates": len(valid), **details,
            })
        return safe_decode(self.tokenizer, committed, skip_special_tokens=True).strip(), committed, diagnostics

    @torch.no_grad()
    def detect(
        self,
        token_ids: list[int],
        *,
        prompt_ids: list[int] | None = None,
        sampler: BlockSampler | None = None,
    ) -> dict:
        """Return a raw high-is-watermarked score for calibration on matched negatives."""
        if self.method == "cad-mark" and (not prompt_ids or sampler is None):
            raise ValueError("CAD-Mark detection requires the original prompt and generator")
        size = self.config.block_size
        ids = token_ids[:self.config.gen_length]
        chunks = [ids[start:start + size] for start in range(0, len(ids), size)]
        texts = [block_decode(block, 0, len(block), self.tokenizer) for block in chunks]
        active = [index for index, text in enumerate(texts) if text]
        result = {"score": 0.0, "n_active_blocks": len(active)}
        if not active:
            return result
        if self.swan:
            return self.swan.score([texts[index] for index in active])
        if self.method == "alimark":
            return {**result, **self._detect_alimark([chunks[index] for index in active])}

        embeddings = self._encode([texts[index] for index in active])
        if self.method.startswith("semfield"):
            score = float(semfield_scores(embeddings.sum(dim=0), self.direction))
            result["score"] = abs(score) if self.method == "semfield-pl" else score
        elif self.method == "cad-mark":
            numerator = variance = 0.0
            for block_id, embedding in zip(active, embeddings):
                _, reference_texts = sampler.sample(prompt_ids, ids[:block_id * size])
                reference = [text for text in reference_texts if text]
                if len(reference) < 2:
                    continue
                pivots = self._cad_pivots(texts[:block_id], prompt_ids)
                value, weight = cadmark_evidence(
                    self._encode(reference) @ pivots.T, embedding @ pivots.T, self.bits[block_id]
                )
                numerator += value
                variance += weight
            result["score"] = numerator / math.sqrt(variance) if variance > 0 else 0.0
        else:
            projections = embeddings @ self.pivots.T
            if self.method == "pmark":
                matches = pmark_soft_matches(projections, self.bits[active])
                result["score"] = float((matches - 0.5).sum()) / math.sqrt(0.25 * matches.numel())
            else:
                result["score"] = samark_score(projections)
        return result

    def _detect_alimark(self, chunks: list[list[int]]) -> dict:
        base = tuple(block_decode(block, 0, len(block), self.tokenizer) for block in chunks)
        variants = [base]
        for i, block in enumerate(chunks):
            if i + 1 < len(chunks):
                merged = block + chunks[i + 1]
                text = block_decode(merged, 0, len(merged), self.tokenizer)
                variants.append(base[:i] + (text,) + base[i + 2:])
            midpoint = len(block) // 2
            left = block_decode(block, 0, midpoint, self.tokenizer)
            right = block_decode(block, midpoint, len(block), self.tokenizer)
            if left and right:
                variants.append(base[:i] + (left, right) + base[i + 1:])
        variants = list(dict.fromkeys(variants))
        unique = list(dict.fromkeys(text for variant in variants for text in variant))
        signals = dict(zip(unique, (self._encode(unique) @ self.pivots.T) > 0))
        score = max(
            alimark_alignment_score(torch.stack([signals[text] for text in variant]), self.bits)
            for variant in variants
        )
        return {"score": score, "n_segmentations": len(variants)}
