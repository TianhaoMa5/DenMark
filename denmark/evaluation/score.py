#!/usr/bin/env python3
"""Evaluate cached DenMark raw scans with disjoint calibration and ROC pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from denmark.evaluation.metrics import (
    DEFAULT_LENGTH_BIN_WIDTH,
    calibrated_scan_scores_by_length_bin,
    length_bin_index,
    score_matrix,
    summarize_roc,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_ids(rows: list[dict], field: str, label: str) -> set[str]:
    values = [str(row[field]) for row in rows if row.get(field) is not None]
    if len(values) != len(rows):
        raise ValueError(f"{label}: every row must contain {field!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}: duplicate {field!r} values")
    return set(values)


def token_lengths(rows: list[dict], field: str, label: str) -> list[int]:
    values: list[int] = []
    for index, row in enumerate(rows):
        if row.get(field) is None:
            raise ValueError(f"{label}: row {index} has no {field!r}")
        value = int(row[field])
        if value < 1:
            raise ValueError(f"{label}: row {index} has invalid token length {value}")
        values.append(value)
    return values


def calibration_path(directory: Path, pattern: str, bin_index: int) -> Path:
    try:
        relative = pattern.format(bin=bin_index)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("calibration-pattern must contain a valid {bin} field") from exc
    return directory / relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--calibration-pattern", default="bin_{bin:03d}.jsonl")
    parser.add_argument("--negative-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scan-min", type=int, default=12)
    parser.add_argument("--scan-max", type=int, default=37)
    parser.add_argument("--score-field", default="raw_scores_by_unit_size")
    parser.add_argument("--source-id-field", default="source_id")
    parser.add_argument("--token-length-field", default="token_length")
    parser.add_argument("--length-bin-width", type=int, default=DEFAULT_LENGTH_BIN_WIDTH)
    parser.add_argument("--expected-calibration-per-bin", type=int, default=10_000)
    parser.add_argument(
        "--fpr",
        type=float,
        nargs="+",
        default=(0.005, 0.01, 0.05),
        help="FPR targets for strict empirical thresholds without interpolation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scan_min <= 0 or args.scan_max < args.scan_min:
        raise ValueError("invalid scan range")
    unit_sizes = list(range(args.scan_min, args.scan_max + 1))
    positives = read_jsonl(args.positive_jsonl)
    negatives = read_jsonl(args.negative_jsonl)
    if not positives or not negatives:
        raise ValueError("positive and negative files must be non-empty")
    if args.length_bin_width <= 0:
        raise ValueError("length-bin-width must be positive")
    if args.expected_calibration_per_bin <= 0:
        raise ValueError("expected-calibration-per-bin must be positive")

    negative_ids = unique_ids(negatives, args.source_id_field, "held-out negatives")
    positive_lengths = token_lengths(
        positives, args.token_length_field, "positives"
    )
    negative_lengths = token_lengths(
        negatives, args.token_length_field, "held-out negatives"
    )
    required_bins = sorted(
        {
            length_bin_index(length, args.length_bin_width)
            for length in (*positive_lengths, *negative_lengths)
        }
    )

    calibration_by_bin: dict[int, np.ndarray] = {}
    calibration_inputs: dict[str, object] = {}
    for bin_index in required_bins:
        path = calibration_path(args.calibration_dir, args.calibration_pattern, bin_index)
        rows = read_jsonl(path)
        if len(rows) != args.expected_calibration_per_bin:
            raise ValueError(
                f"length bin {bin_index}: expected {args.expected_calibration_per_bin} "
                f"calibration rows, found {len(rows)}"
            )
        calibration_ids = unique_ids(
            rows, args.source_id_field, f"calibration bin {bin_index}"
        )
        overlap = calibration_ids & negative_ids
        if overlap:
            raise ValueError(
                f"calibration bin {bin_index} and held-out negatives overlap on "
                f"{len(overlap)} source IDs"
            )
        lengths = token_lengths(
            rows, args.token_length_field, f"calibration bin {bin_index}"
        )
        wrong_bin = [
            length
            for length in lengths
            if length_bin_index(length, args.length_bin_width) != bin_index
        ]
        if wrong_bin:
            raise ValueError(
                f"calibration bin {bin_index} contains {len(wrong_bin)} rows "
                "with mismatched token lengths"
            )
        calibration_by_bin[bin_index] = score_matrix(
            rows, unit_sizes, args.score_field
        )
        calibration_inputs[str(bin_index)] = {
            "path": str(path),
            "sha256": sha256(path),
            "count": len(rows),
        }

    positive_scores = calibrated_scan_scores_by_length_bin(
        score_matrix(positives, unit_sizes, args.score_field),
        positive_lengths,
        calibration_by_bin,
        args.length_bin_width,
    )
    negative_scores = calibrated_scan_scores_by_length_bin(
        score_matrix(negatives, unit_sizes, args.score_field),
        negative_lengths,
        calibration_by_bin,
        args.length_bin_width,
    )
    result = {
        "method": "DenMark",
        "statistic": "per-size empirical p-value + Bonferroni; score=-log(p_scan)",
        "scan_unit_sizes": unit_sizes,
        "calibration_mode": "token_length_binned",
        "length_bin_width": args.length_bin_width,
        "required_length_bins": required_bins,
        "calibration_count_per_bin": args.expected_calibration_per_bin,
        "heldout_negative_count": len(negatives),
        "calibration_heldout_id_overlap": 0,
        "metrics": summarize_roc(positive_scores, negative_scores, args.fpr),
        "inputs": {
            "positive": {"path": str(args.positive_jsonl), "sha256": sha256(args.positive_jsonl)},
            "calibration_by_bin": calibration_inputs,
            "heldout_negative": {
                "path": str(args.negative_jsonl),
                "sha256": sha256(args.negative_jsonl),
            },
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
