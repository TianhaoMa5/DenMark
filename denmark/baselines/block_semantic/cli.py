"""Generate and score complete-block adaptations of sentence watermark baselines."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from denmark.baselines.block_best_of_k.generate import (
    build_chat_prompt,
    seed_generation_rng,
    stable_attempt_seed,
)
from denmark.baselines.block_semantic.model import METHODS, BlockConfig, BlockSampler, BlockWatermark
from denmark.baselines.block_semantic.swan import load_swan
from denmark.baselines.pmark.generate import encoder_embedding_dim
from denmark.core.model import encode_texts, load_generator_model, resolve_mask_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("generate", "detect"):
        command = modes.add_parser(mode)
        command.add_argument("--method", required=True, choices=METHODS)
        command.add_argument("--input", type=Path, required=True, help="Input JSONL")
        command.add_argument("--output", type=Path, required=True, help="Output JSONL")
        command.add_argument("--model", required=True, help="Generator model/tokenizer name or path")
        command.add_argument("--encoder", help="HF encoder name or path, except for SWAN")
        command.add_argument("--device", default="cuda")
        command.add_argument("--generator_family", choices=("llada", "mmada"), default="llada")
        command.add_argument("--mask_id", type=int)
        command.add_argument("--gen_length", type=int, default=300)
        command.add_argument("--block_size", type=int, default=25)
        command.add_argument("--steps", type=int)
        command.add_argument("--num_candidates", type=int, default=16)
        command.add_argument("--candidate_batch_size", type=int, default=4)
        command.add_argument("--temperature", type=float, default=0.7)
        command.add_argument("--remasking", choices=("low_confidence", "random", "ar"), default="low_confidence")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--pivot_seed", type=int, default=42)
        command.add_argument("--secret_seed", type=int, default=0)
        command.add_argument("--num_channels", type=int, default=2)
        command.add_argument("--context_window", type=int, default=3)
        command.add_argument("--swan_repo", type=Path)
        command.add_argument("--amr_bank", type=Path)
        command.add_argument("--swan_accept_threshold", type=float, default=0.75)
        command.add_argument("--swan_detect_threshold", type=float, default=0.65)
        command.add_argument("--swan_null_hit_rate", type=float, default=0.25)
    modes.choices["detect"].add_argument("--retokenize", action="store_true")
    modes.choices["detect"].add_argument("--text_field", help="Retokenize this exact text field")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.method == "swan" and (not args.swan_repo or not args.amr_bank):
        parser.error("SWAN requires --swan_repo and --amr_bank")
    if args.method != "swan" and not args.encoder:
        parser.error(f"{args.method} requires --encoder")
    if min(args.seed, args.pivot_seed, args.secret_seed) < 0:
        parser.error("seeds must be non-negative")
    return args


def prompt_ids(row: dict, tokenizer) -> list[int]:
    for name in ("prompt_ids", "input_ids"):
        if name in row:
            return row[name]
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("expected a nonempty prompt, prompt_ids or input_ids")
    return build_chat_prompt(prompt, "", tokenizer)[1]


def swan_prompt(row: dict, template: str, tokenizer) -> list[int]:
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("SWAN generation requires a nonempty text prompt")
    instruction = (
        f"{prompt}\n\n"
        "Continue the response in the same style and follow the semantic structure "
        "of the following abstract meaning representation in the next passage. "
        "Use context-appropriate words in place of abstract concepts. "
        f"Output only the response text.\n\n{template}"
    )
    return build_chat_prompt(instruction, "", tokenizer)[1]


def detection_ids(
    row: dict, tokenizer, retokenize: bool, text_field: str | None = None
) -> list[int]:
    if text_field:
        if not isinstance(row.get(text_field), str):
            raise ValueError(f"expected a string in {text_field!r}")
        return tokenizer(row[text_field], add_special_tokens=False)["input_ids"]
    for name in ("attacked_text", "attack_text"):
        if isinstance(row.get(name), str):
            return tokenizer(row[name], add_special_tokens=False)["input_ids"]
    if not retokenize and "token_ids" in row:
        return row["token_ids"]
    for name in ("text", "generation_text_skip_special", "generation", "completion", "output"):
        if isinstance(row.get(name), str):
            return tokenizer(row[name], add_special_tokens=False)["input_ids"]
    raise ValueError("expected completion text or token_ids")


def main() -> None:
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer

    config = BlockConfig(
        gen_length=args.gen_length, block_size=args.block_size, steps=args.steps,
        num_candidates=args.num_candidates, candidate_batch_size=args.candidate_batch_size,
        temperature=args.temperature, remasking=args.remasking, generator_family=args.generator_family,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    method_options = {}
    if args.method == "swan":
        method_options["swan"] = load_swan(
            args.swan_repo, args.amr_bank, accept_threshold=args.swan_accept_threshold,
            detect_threshold=args.swan_detect_threshold, null_hit_rate=args.swan_null_hit_rate,
        )
    else:
        encoder = AutoModel.from_pretrained(args.encoder).to(args.device).eval()
        encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder)
        method_options.update(
            embed=lambda texts: encode_texts(texts, encoder, encoder_tokenizer, args.device),
            embedding_dim=encoder_embedding_dim(encoder),
        )
    watermark = BlockWatermark(
        args.method, config, tokenizer, num_channels=args.num_channels, pivot_seed=args.pivot_seed,
        secret_seed=args.secret_seed, context_window=args.context_window, **method_options,
    )
    sampler = None
    if args.mode == "generate" or args.method == "cad-mark":
        model = load_generator_model(
            args.model, generator_family=args.generator_family, device_map=args.device,
            torch_dtype=torch.float32 if args.device == "cpu" else torch.bfloat16,
        )
        sampler = BlockSampler(
            config, model, tokenizer, resolve_mask_id(tokenizer, args.generator_family, args.mask_id),
            args.device,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    )
    try:
        with args.input.open(encoding="utf-8") as source, staged:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                    seed = stable_attempt_seed(args.seed, args.mode, row.get("sample_id", line_number))
                    seed_generation_rng(seed)
                    if args.mode == "generate":
                        prompt = prompt_ids(row, tokenizer)
                        started = time.perf_counter()
                        text, ids, diagnostics = watermark.generate(
                            sampler, prompt, seed=seed,
                            swan_prompt=lambda target: swan_prompt(row, target, tokenizer),
                        )
                        result = {
                            **row, "text": text, "token_ids": ids, "prompt_ids": prompt,
                            "block_size": config.block_size,
                            "generation_seconds": time.perf_counter() - started,
                            "block_diagnostics": diagnostics,
                        }
                    else:
                        ids = detection_ids(row, tokenizer, args.retokenize, args.text_field)
                        prompt = prompt_ids(row, tokenizer) if args.method == "cad-mark" else None
                        result = {
                            **row, "detection": watermark.detect(ids, prompt_ids=prompt, sampler=sampler)
                        }
                    result["method"] = args.method
                    staged.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{args.input}:{line_number}: {exc}") from exc
        Path(staged.name).replace(args.output)
    finally:
        Path(staged.name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
