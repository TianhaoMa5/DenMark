# Reproducibility protocol

This document defines the data preparation, detection, and quality-evaluation
protocol used to reproduce the paper.

## Positive samples

Generate up to 300 watermarked outputs for every method, backbone, and dataset.
Before applying an attack, retain outputs that satisfy all of the following:

- at least 150 tokens under the matching generator tokenizer;
- word-level 4-gram repetition ratio at most 0.2;
- non-empty and non-degenerate text;
- unique output text.

Apply every attack to this fixed retained positive set. Do not remove an
attacked output merely because the attack changes its token length.

## Calibration and ROC negatives

For each backbone, first prepare 10,000 unique C4 RealNewsLike passages using
that backbone's tokenizer. Their lengths are approximately uniform over the
integer range 150 through 300, and they form the held-out empirical
ROC-negative pool. Separately build 10,000 calibration crops for every required
25-token length bin. Bin `b` contains lengths `25b` through `25b+24`.

- DenMark calibrates each response only against the 10,000 rows in its matching
  length bin; missing bins are errors and never fall back to a global pool.
- The 10,000 held-out negatives are shared by all methods for that backbone.
- Every calibration bin and the held-out pool must have zero source-ID overlap.

The matching length-bin calibration pool defines the per-size empirical
p-values. It is never used to choose or evaluate a point on the final ROC
curve. Detection retokenizes and scores the complete post-attack response;
attacked outputs are neither length-filtered again nor truncated to 300 tokens.

## Detection metrics

Use the detector ranking statistic recorded for each method. For DenMark this
is `-log(p_scan)`. Report rank AUC with half credit for ties and the highest
attainable TPR with empirical FPR no greater than the target. Do not interpolate
or split tied scores. Legacy `roc_tpr_*` fields now use this paper protocol;
previous result files must be recomputed, not relabeled.

The main paper uses FPR targets 0.5%, 1%, and 5%. Some diagnostics also include
0.1%; diagnostic columns must not silently replace the paper protocol.

## Attacked text

Delete cached generation token IDs from attack outputs or explicitly force
re-tokenization. Detection must use the attacked string, not the pre-attack
tokens. Preserve stable source IDs so every attacked output can be joined back
to the corresponding unattacked positive.

## Quality

Perplexity is completion-only. Report

```text
mean(log PPL_watermarked) - mean(log PPL_clean)
```

using a clean reference from the same backbone and dataset. Keep raw per-sample
values and source IDs so paired and unpaired summaries can be audited.

## Required provenance

Every released result should include:

- input paths or immutable dataset identifiers;
- model and tokenizer identifiers or revisions;
- encoder identifier or checksum;
- complete generation and watermark configuration;
- direction and message key identifiers (not secret deployment keys);
- filter counts and rejection reasons;
- calibration and held-out pool sizes and source-ID overlap count;
- attack model, prompt version, temperature, and cache key definition;
- code commit and package versions.
