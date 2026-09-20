"""Pure NumPy calibration and ROC utilities for DenMark scan scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_LENGTH_BIN_WIDTH = 25


def length_bin_index(
    token_length: int,
    bin_width: int = DEFAULT_LENGTH_BIN_WIDTH,
) -> int:
    """Return the fixed-width calibration bin for a positive token length."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if token_length < 1:
        raise ValueError("token_length must be positive")
    return int(token_length) // int(bin_width)


def length_bin_bounds(
    bin_index: int,
    bin_width: int = DEFAULT_LENGTH_BIN_WIDTH,
) -> tuple[int, int]:
    """Return inclusive token-length bounds for one calibration bin."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if bin_index < 0:
        raise ValueError("bin_index must be non-negative")
    lower = max(1, int(bin_index) * int(bin_width))
    return lower, int(bin_index) * int(bin_width) + int(bin_width) - 1


def score_matrix(
    records: Sequence[Mapping[str, object]],
    unit_sizes: Sequence[int],
    field: str = "raw_scores_by_unit_size",
) -> np.ndarray:
    """Extract a dense ``[num_records, num_unit_sizes]`` raw-score matrix."""
    rows: list[list[float]] = []
    for index, record in enumerate(records):
        values = record.get(field)
        if not isinstance(values, Mapping):
            raise ValueError(f"record {index} has no mapping field {field!r}")
        try:
            rows.append([float(values[str(size)]) for size in unit_sizes])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"record {index} is missing a finite raw score for one or more unit sizes"
            ) from exc
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(records), len(unit_sizes)):
        raise ValueError(f"unexpected score matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("raw scan scores must all be finite")
    return matrix


def calibrated_scan_scores(
    evaluation: np.ndarray,
    calibration: np.ndarray,
) -> np.ndarray:
    """Calibrate per-size scores, Bonferroni-correct, and return ``-log(p)``.

    ``evaluation`` and ``calibration`` must have one column per candidate unit
    size. Evaluation rows are assumed to be disjoint from the calibration pool,
    so no leave-one-out correction is applied.
    """
    evaluation = np.asarray(evaluation, dtype=np.float64)
    calibration = np.asarray(calibration, dtype=np.float64)
    if evaluation.ndim != 2 or calibration.ndim != 2:
        raise ValueError("evaluation and calibration scores must be two-dimensional")
    if evaluation.shape[1] != calibration.shape[1]:
        raise ValueError("evaluation and calibration must use the same unit-size grid")
    if calibration.shape[0] == 0 or calibration.shape[1] == 0:
        raise ValueError("calibration matrix must be non-empty")
    if not np.isfinite(evaluation).all() or not np.isfinite(calibration).all():
        raise ValueError("scan score matrices must contain only finite values")

    n_calibration = calibration.shape[0]
    p_values = np.empty_like(evaluation)
    for column in range(calibration.shape[1]):
        sorted_null = np.sort(calibration[:, column])
        count_ge = n_calibration - np.searchsorted(
            sorted_null,
            evaluation[:, column],
            side="left",
        )
        p_values[:, column] = (1.0 + count_ge) / (n_calibration + 1.0)
    corrected = np.minimum(1.0, calibration.shape[1] * np.min(p_values, axis=1))
    return -np.log(corrected)


def calibrated_scan_scores_by_length_bin(
    evaluation: np.ndarray,
    token_lengths: Sequence[int] | np.ndarray,
    calibration_by_bin: Mapping[int, np.ndarray],
    bin_width: int = DEFAULT_LENGTH_BIN_WIDTH,
) -> np.ndarray:
    """Calibrate each row only against its matching token-length bin.

    Missing bins are errors. Falling back to a global or neighboring null pool
    would silently change the paper protocol.
    """
    evaluation = np.asarray(evaluation, dtype=np.float64)
    lengths = np.asarray(token_lengths, dtype=np.int64)
    if evaluation.ndim != 2:
        raise ValueError("evaluation scores must be two-dimensional")
    if lengths.ndim != 1 or lengths.shape[0] != evaluation.shape[0]:
        raise ValueError("token_lengths must contain one value per evaluation row")
    if np.any(lengths < 1):
        raise ValueError("token_lengths must be positive")
    if not np.isfinite(evaluation).all():
        raise ValueError("evaluation scores must contain only finite values")

    bins = np.asarray(
        [length_bin_index(int(length), bin_width) for length in lengths],
        dtype=np.int64,
    )
    scores = np.empty(evaluation.shape[0], dtype=np.float64)
    for bin_index in np.unique(bins):
        if int(bin_index) not in calibration_by_bin:
            raise KeyError(f"missing calibration pool for length bin {int(bin_index)}")
        rows = np.flatnonzero(bins == bin_index)
        scores[rows] = calibrated_scan_scores(
            evaluation[rows],
            np.asarray(calibration_by_bin[int(bin_index)], dtype=np.float64),
        )
    return scores


def rank_auc(positive: Iterable[float], negative: Iterable[float]) -> float:
    """Mann-Whitney AUC with half credit for tied scores."""
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.sort(np.asarray(list(negative), dtype=np.float64))
    if pos.size == 0 or neg.size == 0:
        raise ValueError("positive and negative score arrays must be non-empty")
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    return float(np.sum(left + 0.5 * (right - left)) / (pos.size * neg.size))


def roc_interpolated_tpr(
    positive: Iterable[float],
    negative: Iterable[float],
    target_fpr: float,
) -> float:
    """Return linearly interpolated empirical-ROC TPR at an exact FPR."""
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("positive and negative score arrays must be non-empty")
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must lie in [0, 1]")

    thresholds = np.unique(np.concatenate([pos, neg]))[::-1]
    fpr = np.concatenate(
        ([0.0], np.asarray([(neg >= threshold).mean() for threshold in thresholds]), [1.0])
    )
    tpr = np.concatenate(
        ([0.0], np.asarray([(pos >= threshold).mean() for threshold in thresholds]), [1.0])
    )
    # Duplicate FPR values form vertical ROC segments. Keep the highest TPR at
    # each FPR before interpolating horizontally.
    unique_fpr = np.unique(fpr)
    upper_tpr = np.asarray([tpr[fpr == value].max() for value in unique_fpr])
    return float(np.interp(target_fpr, unique_fpr, upper_tpr))


def empirical_tpr(
    positive: Iterable[float],
    negative: Iterable[float],
    target_fpr: float,
) -> float:
    """Highest attainable TPR with empirical FPR no greater than the target."""
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    if not pos.size or not neg.size:
        raise ValueError("positive and negative score arrays must be non-empty")
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("scores must be finite")
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must lie in [0, 1]")
    if target_fpr == 1.0:
        return 1.0
    # Tied scores are indivisible under a deterministic threshold.
    allowed = int(np.floor(target_fpr * neg.size))
    threshold = np.sort(neg)[neg.size - allowed - 1]
    return float((pos > threshold).mean())


def summarize_roc(
    positive: Iterable[float],
    negative: Iterable[float],
    fprs: Sequence[float] = (0.005, 0.01, 0.05),
) -> dict[str, object]:
    """Summarize rank AUC and deterministic empirical-threshold TPRs."""
    pos = list(positive)
    neg = list(negative)
    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "tpr": {str(fpr): empirical_tpr(pos, neg, fpr) for fpr in fprs},
        "tpr_method": "empirical_threshold_no_interpolation",
        "auc": rank_auc(pos, neg),
    }


__all__ = [
    "DEFAULT_LENGTH_BIN_WIDTH",
    "empirical_tpr",
    "calibrated_scan_scores",
    "calibrated_scan_scores_by_length_bin",
    "length_bin_bounds",
    "length_bin_index",
    "rank_auc",
    "roc_interpolated_tpr",
    "score_matrix",
    "summarize_roc",
]
