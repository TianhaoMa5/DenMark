"""Use a separately installed SWAN checkout for AMR parsing and S2match."""

from __future__ import annotations

from importlib import import_module, util
import math
from functools import lru_cache
from pathlib import Path
from typing import Callable


class Swan:
    def __init__(
        self,
        bank: list[str],
        parse: Callable[[str], str],
        similarity: Callable[[str, str], float],
        *,
        accept_threshold: float = 0.75,
        detect_threshold: float = 0.65,
        null_hit_rate: float = 0.25,
    ):
        if not bank:
            raise ValueError("SWAN requires a nonempty AMR template bank")
        if not (0 <= accept_threshold <= 1 and 0 <= detect_threshold <= 1):
            raise ValueError("SWAN thresholds must be in [0, 1]")
        if not 0 < null_hit_rate < 1:
            raise ValueError("SWAN null hit rate must be in (0, 1)")
        self.bank = bank
        self.parse = parse
        self.similarity = similarity
        self.accept_threshold = accept_threshold
        self.detect_threshold = detect_threshold
        self.null_hit_rate = null_hit_rate

    def select(self, candidates: list[str], target: str) -> tuple[int, bool]:
        for index, candidate in enumerate(candidates):
            graph = self.parse(candidate)
            if graph and self.similarity(graph, target) >= self.accept_threshold:
                return index, True
        return len(candidates) - 1, False

    def score(self, blocks: list[str]) -> dict:
        hits = 0
        for block in blocks:
            graph = self.parse(block)
            hits += bool(graph and any(
                self.similarity(graph, template) >= self.detect_threshold for template in self.bank
            ))
        n = len(blocks)
        p = self.null_hit_rate
        score = (hits - n * p) / math.sqrt(n * p * (1 - p)) if n else 0.0
        return {"score": score, "n_active_blocks": n, "green_blocks": hits}


def load_swan(repo: Path, bank: Path, **settings) -> Swan:
    module_path = repo.resolve() / "utils" / "amr_utils.py"
    if not module_path.is_file():
        raise ValueError("--swan_repo must point to an amazon-science/SWAN checkout")
    if not bank.is_file():
        raise ValueError("--amr_bank must point to an existing JSON file")
    spec = util.spec_from_file_location("_denmark_swan_amr", module_path)
    amr = util.module_from_spec(spec)
    spec.loader.exec_module(amr)

    # Upstream reloads the same GloVe file on every graph comparison.
    s2match = import_module("s2match")
    if not hasattr(s2match.load_vecs, "cache_info"):
        s2match.load_vecs = lru_cache(maxsize=1)(s2match.load_vecs)

    def parse(text: str) -> str:
        graph = amr.parse_amr(text)
        if not graph:
            return ""
        normalized = amr.normalize_amr_variables(graph)
        return amr.normalize_amr_variables(amr.generate_template_amr(normalized))

    return Swan(sorted(amr.load_amr_bank(str(bank))), parse, amr.compute_s2match_score, **settings)
