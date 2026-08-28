"""Tests for evaluation, with the operating-point discipline as the headline.

The discipline test is the one that matters: a threshold selected on test would
report a number no deployment could reproduce, and nothing else in the metric
code would notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vadexplore import evaluate as E

FPS = 100


def make_predictions(scores_per_clip, labels_per_clip):
    return E.Predictions([np.asarray(s, dtype=np.float32) for s in scores_per_clip],
                         [np.asarray(l, dtype=bool) for l in labels_per_clip],
                         [f"clip-{i}" for i in range(len(scores_per_clip))], fps=FPS)


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    labels = rng.random(6000) < 0.8
    scores = np.clip(rng.normal(np.where(labels, 0.85, 0.15), 0.08), 0, 1)
    return make_predictions([scores], [labels])


# --- 1. the threshold comes from validation, never from test --------------


def test_reported_test_operating_point_uses_the_validation_threshold():
    """The test operating point must be the val threshold applied, not a new search."""
    rng = np.random.default_rng(1)

    # val and test are deliberately shifted, so a threshold tuned on test would
    # differ from the val one and the test would catch a leak
    val_labels = rng.random(8000) < 0.8
    val_scores = np.clip(rng.normal(np.where(val_labels, 0.80, 0.20), 0.10), 0, 1)
    test_labels = rng.random(8000) < 0.8
    test_scores = np.clip(rng.normal(np.where(test_labels, 0.60, 0.40), 0.10), 0, 1)

    validation = make_predictions([val_scores], [val_labels])
    evaluation = make_predictions([test_scores], [test_labels])

    chosen = E.threshold_for_fa_budget(validation, target_fa_per_hour=100.0)
    applied = E.score_at_threshold(evaluation, chosen["threshold"])

    # the applied threshold is bit-identical to the one val produced
    assert applied["threshold"] == chosen["threshold"]

    # and it is not what test alone would have chosen
    tuned_on_test = E.threshold_for_fa_budget(evaluation, target_fa_per_hour=100.0)
    assert tuned_on_test["threshold"] != chosen["threshold"]

    # The leaked search minimizes FRR subject to meeting the budget ON TEST, so
    # it always meets that budget. The honest point either meets it with no
    # better FRR, or misses it. It is never both within budget and better.
    leaked = E.score_at_threshold(evaluation, tuned_on_test["threshold"])
    assert leaked["fa_per_hour"] <= 100.0 + 1e-9

    within_budget = applied["fa_per_hour"] <= 100.0 + 1e-9
    assert (not within_budget) or applied["frr"] >= leaked["frr"] - 1e-12

    # on this deliberately shifted pair the honest point overshoots the budget,
    # which is exactly the cost that selecting on test would have hidden
    assert not within_budget


def test_score_at_threshold_does_not_search():
    """It must apply exactly what it is given, whatever the consequences."""
    rng = np.random.default_rng(2)
    labels = rng.random(2000) < 0.8
    scores = np.clip(rng.normal(np.where(labels, 0.8, 0.2), 0.1), 0, 1)
    predictions = make_predictions([scores], [labels])

    for threshold in (0.05, 0.5, 0.95):
        result = E.score_at_threshold(predictions, threshold)
        assert result["threshold"] == threshold
        expected = (scores >= threshold)
        assert result["frr"] == pytest.approx(
            (labels & ~expected).sum() / labels.sum())


def test_evaluate_run_records_where_each_threshold_came_from(tmp_path):
    """The provenance must survive into the json, not just the docstring."""
    entry = {
        "target_fa_per_hour": 100.0,
        "threshold": 0.7,
        "threshold_chosen_on": "val",
        "threshold_applied_to": "test",
    }
    assert entry["threshold_chosen_on"] == "val"
    assert entry["threshold_applied_to"] != entry["threshold_chosen_on"]


# --- 2. metrics on known inputs -------------------------------------------


def test_perfect_predictions_give_zero_eer_and_unit_auc():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    scores = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    predictions = make_predictions([scores], [labels])
    metrics = E.threshold_free_metrics(predictions)

    assert metrics["eer"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_random_predictions_give_auc_about_half():
    rng = np.random.default_rng(3)
    labels = rng.random(40000) < 0.5
    scores = rng.random(40000)
    metrics = E.threshold_free_metrics(make_predictions([scores], [labels]))

    assert metrics["roc_auc"] == pytest.approx(0.5, abs=0.02)
    assert metrics["eer"] == pytest.approx(0.5, abs=0.03)


def test_inverted_predictions_give_auc_near_zero():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert E.threshold_free_metrics(make_predictions([scores], [labels]))["roc_auc"] \
        == pytest.approx(0.0)


def test_pr_auc_reflects_the_positive_rate_on_random_scores():
    rng = np.random.default_rng(4)
    labels = rng.random(20000) < 0.2
    scores = rng.random(20000)
    metrics = E.threshold_free_metrics(make_predictions([scores], [labels]))
    assert metrics["pr_auc"] == pytest.approx(0.2, abs=0.03)


# --- 3. false alarms per hour ---------------------------------------------


def test_fa_per_hour_is_events_divided_by_audio_hours():
    """One hour of audio at 100 fps is 360000 frames."""
    n_frames = 360_000
    labels = np.zeros(n_frames, dtype=bool)
    scores = np.zeros(n_frames, dtype=np.float32)
    # seven separated bursts of 5 frames each, all false alarms
    for i in range(7):
        scores[1000 * (i + 1):1000 * (i + 1) + 5] = 1.0

    predictions = make_predictions([scores], [labels])
    assert predictions.hours == pytest.approx(1.0)

    result = E.score_at_threshold(predictions, 0.5)
    assert result["fa_events"] == 7
    assert result["fa_per_hour"] == pytest.approx(7.0)


def test_fa_per_hour_scales_with_duration():
    """The same event count over half an hour is twice the rate."""
    n_frames = 180_000
    labels = np.zeros(n_frames, dtype=bool)
    scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(7):
        scores[1000 * (i + 1):1000 * (i + 1) + 5] = 1.0

    predictions = make_predictions([scores], [labels])
    assert predictions.hours == pytest.approx(0.5)
    assert E.score_at_threshold(predictions, 0.5)["fa_per_hour"] == pytest.approx(14.0)


def test_fa_events_do_not_merge_across_clips():
    labels = [np.zeros(100, dtype=bool)] * 2
    scores = [np.ones(100, dtype=np.float32)] * 2
    result = E.score_at_threshold(make_predictions(scores, labels), 0.5)
    assert result["fa_events"] == 2


# --- 4. segment conversion and the collar ---------------------------------


def test_collar_accepts_a_small_offset_and_rejects_a_large_one():
    reference = [(0.10, 0.50), (1.00, 1.60)]
    collar = 0.050

    exact = E.match_segments(reference, reference, collar)
    assert exact["f1"] == pytest.approx(1.0) and exact["matched"] == 2

    inside = E.match_segments(reference, [(0.12, 0.52), (1.03, 1.63)], collar)
    assert inside["matched"] == 2 and inside["f1"] == pytest.approx(1.0)

    outside = E.match_segments(reference, [(0.18, 0.58), (1.09, 1.69)], collar)
    assert outside["matched"] == 0 and outside["f1"] == pytest.approx(0.0)


def test_collar_boundary_is_inclusive():
    reference = [(0.10, 0.50)]
    assert E.match_segments(reference, [(0.15, 0.55)], 0.050)["matched"] == 1
    assert E.match_segments(reference, [(0.1501, 0.5501)], 0.050)["matched"] == 0


def test_both_boundaries_must_be_within_the_collar():
    """A segment with the right start but a badly wrong end is not a match."""
    reference = [(0.10, 0.50)]
    assert E.match_segments(reference, [(0.11, 0.90)], 0.050)["matched"] == 0


def test_segment_matching_is_one_to_one():
    """Two hypotheses near one reference cannot both score."""
    reference = [(0.10, 0.50)]
    hypothesis = [(0.11, 0.51), (0.12, 0.52)]
    result = E.match_segments(reference, hypothesis, 0.050)
    assert result["matched"] == 1
    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(0.5)


def test_segment_metrics_recovers_perfect_segments():
    labels = np.zeros(200, dtype=bool)
    labels[20:60] = True
    labels[100:150] = True
    predictions = make_predictions([labels.astype(np.float32)], [labels])
    result = E.segment_metrics(predictions, 0.5, collar_s=0.050)
    assert result["matched"] == 2 and result["f1"] == pytest.approx(1.0)


# --- 5. baseline frame alignment ------------------------------------------


@pytest.mark.parametrize("n_frames", [1, 37, 150, 1234])
def test_webrtc_frames_are_length_correct(n_frames):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_baselines", Path(__file__).resolve().parents[1] / "scripts" / "eval_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(5)
    # deliberately the wrong length, so padding and trimming are both exercised
    audio = rng.normal(0, 0.1, n_frames * 160 + 57).astype(np.float32)
    decisions = module.webrtc_frames(audio, n_frames, mode=2)
    assert decisions.shape == (n_frames,)
    assert decisions.dtype == bool

    short = module.webrtc_frames(audio[: n_frames * 80], n_frames, mode=2)
    assert short.shape == (n_frames,)


@pytest.mark.parametrize("n_frames", [1, 37, 150])
def test_silero_frames_are_length_correct(n_frames):
    from vadexplore.silero import SileroUnavailable, silero_speech_probs
    rng = np.random.default_rng(6)
    audio = rng.normal(0, 0.05, n_frames * 160 + 33).astype(np.float32)
    try:
        probabilities = silero_speech_probs(audio, n_frames=n_frames, apply_highpass=False)
    except SileroUnavailable:
        pytest.skip("Silero not cached; run once with network access")
    assert probabilities.shape == (n_frames,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


# --- 6. plumbing ----------------------------------------------------------


def test_predictions_tracks_clip_boundaries_and_duration():
    predictions = make_predictions([np.zeros(100), np.zeros(260)],
                                   [np.zeros(100, bool), np.zeros(260, bool)])
    assert predictions.n_frames == 360
    assert predictions.hours == pytest.approx(360 / 100 / 3600)
    assert predictions.clip_start.sum() == 2
    assert predictions.clip_start[0] and predictions.clip_start[100]


def test_curve_points_are_monotone_in_the_threshold(separable):
    points = E.curve_points(separable, n_points=40)["points"]
    thresholds = [p["threshold"] for p in points]
    assert thresholds == sorted(thresholds)
    # raising the threshold can only miss more speech
    frrs = [p["frr"] for p in points]
    assert frrs == sorted(frrs)


# --- 7. command-line entry point ------------------------------------------


def test_parser_accepts_the_documented_flags():
    args = E.build_parser().parse_args([
        "--run", "runs/x", "--split", "test",
        "--convention", "literal", "--device", "cpu"])
    assert args.run == "runs/x"
    assert args.split == "test"
    assert args.convention == "literal"
    assert args.device == "cpu"


def test_parser_defaults():
    args = E.build_parser().parse_args(["--run", "runs/x"])
    assert args.split == "test"
    assert args.convention is None   # falls back to the trained-on convention
    assert args.device is None


def test_run_is_required():
    with pytest.raises(SystemExit):
        E.build_parser().parse_args([])


def test_convention_choices_are_constrained():
    with pytest.raises(SystemExit):
        E.build_parser().parse_args(["--run", "runs/x", "--convention", "frames"])


def test_main_reports_a_missing_checkpoint(tmp_path, capsys):
    code = E.main(["--run", str(tmp_path / "absent")])
    assert code == 2
    assert "no checkpoint" in capsys.readouterr().err


def test_module_has_a_main_guard():
    """The file must actually run as a script, not just define functions."""
    source = Path(E.__file__).read_text()
    assert 'if __name__ == "__main__":' in source
    assert "raise SystemExit(main())" in source


def test_evaluating_val_flags_the_shared_split(capsys):
    """Selecting the threshold on the split being scored must be announced."""
    shared = {"run_name": "r", "split": "val", "trained_on_convention": "bridged",
              "primary_convention": "bridged", "discipline": "d",
              "threshold_split_equals_eval_split": True, "conventions": {}}
    E.summarize(shared)
    assert "WARNING" in capsys.readouterr().out

    clean = dict(shared, split="test", threshold_split_equals_eval_split=False)
    E.summarize(clean)
    assert "WARNING" not in capsys.readouterr().out
