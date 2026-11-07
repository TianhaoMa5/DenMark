"""Candidate selectors and detectors for embedding-based block baselines."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from denmark.baselines.pmark.model import pmark_soft_matches


def select_alimark(projections: torch.Tensor, targets: torch.Tensor, rng: random.Random) -> int:
    """Choose uniformly among candidates with the most matching bits."""
    matches = ((projections > 0) == targets).sum(dim=1)
    return rng.choice(torch.where(matches == matches.max())[0].tolist())


def alimark_alignment_score(bits: torch.Tensor, secret_bits: torch.Tensor) -> float:
    """Best normalized block edit similarity across plausible secret-prefix lengths."""
    n, channels = bits.shape
    if not n:
        return 0.0
    lower = max(1, math.ceil(0.5 * n))
    upper = min(len(secret_bits), math.ceil(1.5 * n))
    if upper < lower:
        raise ValueError("AliMark secret sequence is too short for alignment")
    hamming = (bits[:, None] != secret_bits[None, :upper]).sum(dim=2).tolist()
    previous = [j * channels for j in range(upper + 1)]
    for i, costs in enumerate(hamming, 1):
        current = [i * channels]
        for j, substitution in enumerate(costs, 1):
            current.append(min(
                previous[j] + channels,
                current[j - 1] + channels,
                previous[j - 1] + substitution,
            ))
        previous = current
    return 1.0 - min(previous[j] / (max(n, j) * channels) for j in range(lower, upper + 1))


def _ngrams(text: str) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return set(zip(words, words[1:], words[2:], words[3:]))


def select_samark(
    embeddings: torch.Tensor,
    projections: torch.Tensor,
    targets: torch.Tensor,
    candidates: list[str],
    context: str,
    vocabulary: set[str],
    previous_embeddings: list[torch.Tensor],
    rng: random.Random,
) -> int:
    """Filter repetition, maximize satisfied signs, then sample by tanh score."""
    indices = torch.arange(len(candidates))
    context_ngrams = _ngrams(context)
    candidate_ngrams = [_ngrams(candidate) for candidate in candidates]
    overlap = torch.tensor([
        len(grams & context_ngrams) / max(1, len(grams)) for grams in candidate_ngrams
    ])
    if (overlap < 0.5).any():
        indices = indices[overlap < 0.5]

    scores = torch.tanh(30.0 * projections * (2.0 * targets.float() - 1.0)).sum(dim=1)
    if previous_embeddings:
        redundancy = (embeddings @ torch.stack(previous_embeddings).T).amax(dim=1)
        diverse = indices[redundancy[indices] < 0.85]
        if len(diverse):
            indices = diverse
        scores += 0.3 * (1.0 - redundancy)

    candidate_vocabularies = [set(candidate.lower().split()) for candidate in candidates]
    scores += 0.15 * torch.tensor([
        len(words - vocabulary) / max(1, len(words)) for words in candidate_vocabularies
    ])
    matches = ((projections[indices] > 0) == targets).sum(dim=1)
    survivors = indices[matches == matches.max()]
    weights = torch.softmax(80.0 * scores[survivors], dim=0)
    return rng.choices(survivors.tolist(), weights=weights.tolist(), k=1)[0]


def samark_score(projections: torch.Tensor) -> float:
    """Infer document signs from the raw channel sums before transforming."""
    signs = torch.where(projections.sum(dim=0) >= 0, 1.0, -1.0)
    values = torch.tanh(30.0 * projections * signs).flatten()
    std = float(values.std()) if values.numel() > 1 else 0.0
    return float(values.mean()) * math.sqrt(values.numel()) / std if std >= 1e-12 else 0.0


def gaussian_direction(dim: int, seed: int) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).standard_normal(dim)).float()


def semfield_scores(aggregate: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return F.normalize(aggregate, dim=-1) @ direction


def cadmark_pivots(context: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Use the normalized context as the first Gram–Schmidt pivot."""
    if float(context.norm()) == 0:
        return directions
    basis = [F.normalize(context, dim=0)]
    # Project the keyed directions off the context and each accepted pivot.
    for direction in directions:
        if len(basis) == len(directions):
            break
        for pivot in basis:
            direction = direction - (direction @ pivot) * pivot
        norm = direction.norm()
        if norm > 1e-6:
            basis.append(direction / norm)
    return torch.stack(basis)


def cadmark_partition(
    projections: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct successive medians and keep the longest satisfiable prefix."""
    survivors = torch.arange(len(projections))
    medians = []
    for channel, bit in enumerate(targets):
        values = projections[survivors, channel]
        if len(values) < 2 or values.min() == values.max():
            break
        boundary = torch.quantile(values, 0.5)
        keep = (values >= boundary) == bit
        if not keep.any():
            break
        medians.append(boundary)
        survivors = survivors[keep]
    return survivors, torch.stack(medians) if medians else projections.new_empty(0)


def cadmark_evidence(
    reference: torch.Tensor, observed: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float]:
    _, medians = cadmark_partition(reference, targets)
    channels = len(medians)
    if not channels:
        return 0.0, 0.0
    weight = float(reference[:, 0].std(unbiased=False))
    matches = pmark_soft_matches(observed[:channels] - medians, targets[:channels])
    return weight * float((matches - 0.5).sum()), 0.25 * channels * weight ** 2
