import json
import sys

from denmark.evaluation import score


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def raw_row(source_id, token_length, value):
    return {
        "source_id": source_id,
        "token_length": token_length,
        "raw_scores_by_unit_size": {"12": value},
    }


def test_score_cli_uses_matching_length_bins(tmp_path, monkeypatch):
    positives = tmp_path / "positive.jsonl"
    negatives = tmp_path / "negative.jsonl"
    calibration = tmp_path / "calibration"
    output = tmp_path / "result.json"
    calibration.mkdir()

    write_jsonl(positives, [raw_row("positive", 150, 3.0)])
    write_jsonl(negatives, [raw_row("heldout", 175, -1.0)])
    write_jsonl(
        calibration / "bin_006.jsonl",
        [raw_row("cal-6-a", 150, 0.0), raw_row("cal-6-b", 174, 1.0)],
    )
    write_jsonl(
        calibration / "bin_007.jsonl",
        [raw_row("cal-7-a", 175, 0.0), raw_row("cal-7-b", 199, 1.0)],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--positive-jsonl",
            str(positives),
            "--calibration-dir",
            str(calibration),
            "--negative-jsonl",
            str(negatives),
            "--output-json",
            str(output),
            "--scan-min",
            "12",
            "--scan-max",
            "12",
            "--expected-calibration-per-bin",
            "2",
        ],
    )
    score.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["calibration_mode"] == "token_length_binned"
    assert result["required_length_bins"] == [6, 7]
    assert result["calibration_count_per_bin"] == 2
    assert set(result["inputs"]["calibration_by_bin"]) == {"6", "7"}


def test_score_cli_rejects_calibration_heldout_overlap(tmp_path, monkeypatch):
    positives = tmp_path / "positive.jsonl"
    negatives = tmp_path / "negative.jsonl"
    calibration = tmp_path / "calibration"
    output = tmp_path / "result.json"
    calibration.mkdir()

    write_jsonl(positives, [raw_row("positive", 150, 3.0)])
    write_jsonl(negatives, [raw_row("shared", 150, -1.0)])
    write_jsonl(
        calibration / "bin_006.jsonl",
        [raw_row("shared", 150, 0.0), raw_row("cal-6-b", 174, 1.0)],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--positive-jsonl",
            str(positives),
            "--calibration-dir",
            str(calibration),
            "--negative-jsonl",
            str(negatives),
            "--output-json",
            str(output),
            "--scan-min",
            "12",
            "--scan-max",
            "12",
            "--expected-calibration-per-bin",
            "2",
        ],
    )

    try:
        score.main()
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("source overlap was not rejected")
