#!/usr/bin/env python3
"""Build one source-disjoint C4 calibration pool for a token-length bin.

Run this tokenizer-heavy command on a compute node. Each invocation writes one
10,000-row bin and an audit file. Held-out ROC source documents are excluded in
full before any crop is sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator

from denmark.data.negatives import DATASET_CONFIG, DATASET_NAME, DATASET_REVISION
from denmark.evaluation.metrics import length_bin_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Short backbone label for row IDs")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or local path")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--bin", type=int, required=True, dest="bin_index")
    parser.add_argument("--heldout-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-per-bin", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def heldout_source_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    for row in read_jsonl(path):
        if row.get("source_index") is None:
            raise ValueError("every held-out row must contain source_index")
        index = int(row["source_index"])
        if index in indices:
            raise ValueError(f"duplicate held-out source_index: {index}")
        indices.add(index)
    if not indices:
        raise ValueError("held-out manifest is empty")
    return indices


def main() -> None:
    args = parse_args()
    if args.n_per_bin <= 0:
        raise ValueError("n-per-bin must be positive")
    lower, upper = length_bin_bounds(args.bin_index)
    partial = args.output.with_suffix(".partial.jsonl")
    audit_path = args.output.with_suffix(".audit.json")
    if any(path.exists() for path in (args.output, partial, audit_path)):
        raise FileExistsError("output exists; preserve it and choose a new output path")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    excluded = heldout_source_indices(args.heldout_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    rng = random.Random(args.seed + args.bin_index)
    stream = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=args.dataset_revision,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    selected_sources: set[int] = set()
    text_hashes: set[str] = set()
    histogram: dict[int, int] = {}
    with partial.open("x", encoding="utf-8") as handle:
        for source_index, source in enumerate(stream):
            if source_index in excluded:
                continue
            text = " ".join(
                str(source.get("text") or "").replace("NEWLINE_CHAR", " ").split()
            )
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            target_length = lower + count % (upper - lower + 1)
            if len(token_ids) < target_length:
                continue
            crop_start = rng.randint(0, len(token_ids) - target_length)
            cropped = tokenizer.decode(
                token_ids[crop_start : crop_start + target_length],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            cropped = " ".join(cropped.split())
            roundtrip_length = len(
                tokenizer(cropped, add_special_tokens=False)["input_ids"]
            )
            if roundtrip_length != target_length:
                continue
            text_hash = hashlib.sha256(cropped.encode()).hexdigest()
            if text_hash in text_hashes:
                continue

            text_hashes.add(text_hash)
            selected_sources.add(source_index)
            row = {
                "id": f"{args.base}:{args.bin_index}:{source_index}",
                "source_id": f"c4-realnewslike:{source_index}",
                "source_index": source_index,
                "text": cropped,
                "token_length": target_length,
                "tokenizer": args.base,
                "tokenizer_name": args.tokenizer,
                "tokenizer_revision": args.tokenizer_revision,
                "length_bin": args.bin_index,
                "crop_start": crop_start,
                "text_sha256": text_hash,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            histogram[target_length] = histogram.get(target_length, 0) + 1
            if count % 1_000 == 0:
                print(json.dumps({"bin": args.bin_index, "rows": count}), flush=True)
            if count == args.n_per_bin:
                break

    if count != args.n_per_bin:
        raise RuntimeError(
            f"incomplete pool: {count}/{args.n_per_bin}; partial output preserved"
        )
    overlap = selected_sources & excluded
    if overlap:
        raise RuntimeError(f"calibration/held-out source overlap: {len(overlap)}")
    partial.rename(args.output)

    audit = {
        "protocol": "length_binned_c4_calibration",
        "base": args.base,
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": args.dataset_revision,
        "tokenizer_name": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "bin": args.bin_index,
        "bounds_inclusive": [lower, upper],
        "rows": count,
        "length_histogram": dict(sorted(histogram.items())),
        "heldout_source_count": len(excluded),
        "source_overlap": 0,
        "seed": args.seed,
        "output_sha256": file_sha256(args.output),
        "heldout_sha256": file_sha256(args.heldout_jsonl),
        "cross_bin_sources_may_repeat": True,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
