"""Empirical scanning, calibration separation, ties, and ROC interpolation."""

import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from denmark.baselines.semantic_detect import calibrated_scan
from denmark.core.calibration import (
    CalibratedDetectorSuite,
    empirical_right_tail_p,
    empirical_right_tail_p_loo,
)
from denmark.evaluation.metrics import (
    empirical_tpr,
    calibrated_scan_scores,
    calibrated_scan_scores_by_length_bin,
    length_bin_bounds,
    length_bin_index,
    rank_auc,
    roc_interpolated_tpr,
    score_matrix,
)


def test_paper_tpr_does_not_interpolate_ties():
    assert empirical_tpr([2, 2], [2, 1], 0.25) == 0.0
    assert empirical_tpr([3, 2], [2, 1], 0.25) == 0.5
    assert empirical_tpr([3, 2], [2, 1], 0.5) == 1.0
    assert empirical_tpr([0], [1], 1.0) == 1.0


def test_umr_scores_full_response_by_default(monkeypatch):
    from denmark.baselines.umr import detect
    monkeypatch.setattr('sys.argv', ['detect', '--umr_root', '.', '--bitmap_path',
        'bitmap', '--tokenizer', 'test', '--vocab_size', '1000',
        '--positive_jsonl', 'pos', '--negative_jsonl', 'neg', '--output_json', 'out'])
    args = detect.parse_args()
    assert args.max_token_len is None
    seen = []
    def score(ids, bitmap, **kwargs):
        seen.append(len(ids))
        return {'z_score': 1.0}
    monkeypatch.setattr(detect, 'score_umr_tokens', score)
    tokenizer = lambda *a, **kw: {'input_ids': list(range(450))}
    detect.score_rows([{'text': 'expanded response'}], tokenizer, None, args, positive=True)
    assert seen == [450]
    args.max_token_len = 300
    detect.score_rows([{'text': 'expanded response'}], tokenizer, None, args, positive=True)
    assert seen == [450, 300]


def test_denmark_detector_retokenizes_and_keeps_full_response():
    from denmark.evaluation.detect import make_item

    tokenizer = lambda *args, **kwargs: {"input_ids": list(range(450))}
    item = make_item(
        {"text": "expanded response", "watermarked_token_ids": [1, 2, 3]},
        0,
        tokenizer,
    )
    assert item["token_len"] == 450
    assert len(item["token_ids"]) == 450


def test_score_matrix_uses_string_unit_size_keys():
    records = [
        {"raw_scores_by_unit_size": {"12": 1.0, "13": 2.0}},
        {"raw_scores_by_unit_size": {"12": 3.0, "13": 4.0}},
    ]
    assert score_matrix(records, [12, 13]).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_calibrated_scan_is_per_size_then_bonferroni():
    calibration = np.asarray([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]])
    evaluation = np.asarray([[3.0, 0.0], [1.5, 1.5], [-1.0, -1.0]])
    scores = calibrated_scan_scores(evaluation, calibration)
    # Row 0 has min p=1/4, corrected by two sizes to 1/2.
    assert scores[0] == pytest.approx(-np.log(0.5))
    # Row 2 is no stronger than any calibration sample and clips to p=1.
    assert scores[2] == pytest.approx(0.0)


def test_length_bin_boundaries_are_25_tokens_wide():
    assert length_bin_index(150) == 6
    assert length_bin_index(174) == 6
    assert length_bin_index(175) == 7
    assert length_bin_index(324) == 12
    assert length_bin_index(325) == 13
    assert length_bin_bounds(6) == (150, 174)
    assert length_bin_bounds(13) == (325, 349)


def test_length_binned_calibration_routes_without_global_fallback():
    evaluation = np.asarray([[5.0, 5.0], [5.0, 5.0]])
    calibration = {
        6: np.zeros((4, 2)),
        13: np.full((4, 2), 10.0),
    }
    scores = calibrated_scan_scores_by_length_bin(
        evaluation,
        [150, 325],
        calibration,
    )
    assert scores[0] > scores[1]
    with pytest.raises(KeyError, match="length bin 7"):
        calibrated_scan_scores_by_length_bin(
            np.asarray([[1.0]]),
            [175],
            {6: np.zeros((4, 1))},
        )


def test_rank_auc_gives_half_credit_to_ties():
    assert rank_auc([1.0, 2.0], [0.0, 2.0]) == pytest.approx(0.625)


def test_roc_interpolation_handles_vertical_tied_segments():
    value = roc_interpolated_tpr([2.0, 2.0], [2.0, 1.0], 0.25)
    assert value == pytest.approx(0.5)


def test_scan_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        calibrated_scan_scores(np.asarray([[np.nan]]), np.asarray([[0.0]]))


class FakeRawDetector:
    scan_block_sizes = [12, 13]

    def raw(self, detector, text, token_ids, block_size=None):
        assert detector == "scan_raw"
        return SimpleNamespace(raw=float(token_ids[block_size - 12]))


def test_empirical_loo_excludes_only_self_and_keeps_other_ties():
    pool = [3.0, 3.0, 1.0]
    assert empirical_right_tail_p(3.0, pool) == pytest.approx(3.0 / 4.0)
    assert empirical_right_tail_p_loo(3.0, pool) == pytest.approx(2.0 / 3.0)


def test_suite_uses_loo_only_for_negative_objects():
    negatives = [
        {"text": "n0", "token_ids": [3.0, 3.0]},
        {"text": "n1", "token_ids": [2.0, 2.0]},
        {"text": "n2", "token_ids": [1.0, 1.0]},
    ]
    suite = CalibratedDetectorSuite(FakeRawDetector(), negatives, detectors=["calibrated_scan"])
    negative = suite.score_item(negatives[0], "calibrated_scan")
    positive = suite.score_item({"text": "p", "token_ids": [4.0, 4.0]}, "calibrated_scan")
    assert negative["p_value"] == pytest.approx(2.0 / 3.0)
    assert positive["p_value"] == pytest.approx(0.5)


def test_suite_supports_disjoint_calibration_and_roc_negatives():
    calibration = [
        {"text": "c0", "token_ids": [3.0, 3.0]},
        {"text": "c1", "token_ids": [2.0, 2.0]},
        {"text": "c2", "token_ids": [1.0, 1.0]},
    ]
    heldout = [{"text": "n0", "token_ids": [100.0, 100.0]}]
    suite = CalibratedDetectorSuite(
        FakeRawDetector(),
        heldout,
        detectors=["calibrated_scan"],
        calibration_items=calibration,
    )
    positive = suite.score_item({"text": "p", "token_ids": [4.0, 4.0]}, "calibrated_scan")
    assert positive["p_value"] == pytest.approx(0.5)
    assert suite.neg_scores["calibrated_scan"] == pytest.approx([0.69314718056])


class CalibratedScanTests(unittest.TestCase):
    def test_negative_scores_are_out_of_fold_and_positive_uses_full_calibration(self):
        result = calibrated_scan(
            [25],
            {25: [100.0, 0.0, 100.0, 0.0]},
            {25: [100.0]},
            [0, 1, 0, 1],
        )
        self.assertEqual(
            [record["n_calibration"] for record in result["negative_records"]],
            [2, 2, 2, 2],
        )
        self.assertEqual(result["positive_records"][0]["n_calibration"], 4)
        self.assertEqual(result["negative_fold_ids"], [0, 1, 0, 1])
        # A high held-out negative sees only the two low scores in the other
        # fold, proving it was not calibrated on itself or its prompt group.
        self.assertAlmostEqual(
            result["negative_records"][0]["best_single_size_p"],
            1.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
