# Semantic watermark baselines on diffusion blocks

These LLaDA and MMada adaptations share the Block Best-of-K proposal decoder.
Each generation step samples `K` complete fixed-token blocks from the committed
prefix, selects one using the method's rule, and appends its original token IDs.
The proposal budget is per complete block. Microbatching controls how many
proposals occupy device memory together. Wholly special-token blocks have no
semantic evidence.

| CLI method | Generation and detection | Reference |
| --- | --- | --- |
| `swan` | Condition proposals on an AMR template, accept the first candidate that passes S2match, and detect using the AMR bank hit rate. | [SWAN](https://aclanthology.org/2026.acl-long.1681/) |
| `pmark` | Apply the existing prior-mode multi-channel half-space filter and its soft-match detector. | [PMark](https://arxiv.org/abs/2509.21057) |
| `alimark` | Maximize sign matches to a bit stream. The detector tries a single adjacent-block merge or a midpoint split, then finds the best block edit alignment over secret prefixes. | [AliMark](https://arxiv.org/abs/2605.29434) |
| `samark` | Draw one set of signs per response, filter for diversity, and use tanh scores to sample among the best sign matches. The detector infers the signs from the scored text. | [SAMark](https://arxiv.org/abs/2605.25796) |
| `cad-mark` | Bisect the candidate pool along context-dependent orthogonal pivots. The detector draws fresh candidates from each observed prefix, reconstructs the boundaries, and weights the soft matches by candidate diversity. | [CAD-Mark](https://aclanthology.org/2026.emnlp-main.544/) |
| `semfield`, `semfield-pl` | Maximize the normalized running aggregate's projection onto a keyed Gaussian direction. The PL variant locks polarity on the first active block and detects with the absolute projection. | [SemField](https://arxiv.org/abs/2610.11848) |

## Run on JSONL

Use a text `prompt` or pretokenized `prompt_ids` per row. An optional
`sample_id` keeps the per-row sampling seed stable across shards. The following
example uses the original LLaDA-8B generator and MPNet embeddings.

```json
{"sample_id":"example-1","prompt":"Explain why the ocean appears blue."}
```

```bash
python -m denmark baseline block-semantic generate \
  --method semfield --input prompts.jsonl --output runs/semfield.jsonl \
  --model GSAI-ML/LLaDA-8B-Instruct --encoder sentence-transformers/all-mpnet-base-v2

python -m denmark baseline block-semantic detect \
  --method semfield --input runs/semfield.jsonl --output runs/semfield.scores.jsonl \
  --model GSAI-ML/LLaDA-8B-Instruct --encoder sentence-transformers/all-mpnet-base-v2
```

Detection uses saved completion `token_ids` by default. Use `--retokenize`
for rewritten text stored in `text`, or `--text_field FIELD` for a different
field. Rows with `attacked_text` or `attack_text` are tokenized from that field.
For CAD-Mark, each row also needs the original prompt or prompt IDs. Match the
generator, denoising settings, and candidate count between generation and
detection. CAD-Mark loads the generator during detection. The other methods
only load its tokenizer.

Keep the block size, token budget, encoder, and method keys consistent across
generation and detection. For comparisons, use the raw `detection.score`
with matched unwatermarked negatives and a disjoint calibration set. The paper
thresholds and null distributions depend on sentence segmentation and model
settings.

## Details of the block adaptations

- SWAN uses one randomly chosen template for all `K` proposals in a block and
  falls back to the last nonempty candidate when none passes. For generation,
  supply the original text `prompt` so the adapter can add the template
  instruction before the committed response. Detection uses the entire bank.
- AliMark uses normalized edit similarity before empirical calibration. Its
  restructuring step splits by token count and merges neighboring blocks.
- CAD-Mark uses the midpoint sample median at each surviving split. If a split
  is degenerate or the valid pool is exhausted, it keeps the last nonempty
  subset and scores only the earlier usable channels. For the first block,
  the context pivot comes from the prompt. A later pivot uses the latest
  `--context_window` observed blocks, with a default of three.

## SWAN setup

In a separate environment, follow the upstream setup for the parser, spaCy,
S2match patch, and GloVe. Set `GLOVE_VECTORS_PATH` to an absolute path.
Install DenMark in the same environment so the CLI can import the AMR backend.
Pass `--swan_repo /path/to/SWAN --amr_bank /path/to/your/amr_bank.json` to both
commands and omit `--encoder`. The upstream checkout has its own license.
[SWAN setup](https://github.com/amazon-science/SWAN#setup)
