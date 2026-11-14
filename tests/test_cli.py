"""Checks for the CLI plumbing (no model training, no audio hardware).

Run with:  py -m pytest tests/ -q
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voiceauth import cli, config


def test_speaker_path_is_filesystem_safe():
    p = cli.speaker_path("Example / User..")
    assert p.parent == cli.SPEAKER_DIR
    assert p.suffix == ".pkl"
    assert "/" not in p.name and " " not in p.name
    assert cli.speaker_path("...").name == "speaker.pkl"


def test_resample_preserves_duration_and_target_rate():
    from scipy.signal import resample_poly
    y = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
    out = cli._resample_16k(y, 44100)
    assert abs(len(out) - 16000) <= 2
    # a 44.1 kHz sine must survive resampling at nearly full amplitude
    assert 0.9 < float(np.max(np.abs(out))) <= 1.01
    # already-16k input is passed through untouched
    assert cli._resample_16k(np.zeros(100, np.float32), 16000).shape == (100,)


def test_normalise_is_affine():
    assert cli.normalise(1.0, (1.0, 2.0)) == 0.0
    assert cli.normalise(3.0, (1.0, 2.0)) == 1.0
    assert cli.normalise(0.0, (0.0, 1.0)) == 0.0


def test_default_threshold_is_tnorm_scale():
    thr, source = cli.default_threshold()
    # T units: a few standard deviations, not the ~0.3 raw-LLR value
    assert 1.0 < thr < 8.0
    assert isinstance(source, str) and source


def test_published_tnorm_threshold_matches_report():
    report = config.OUTPUTS / "report.txt"
    if not report.exists():
        return  # eval not run in this checkout
    text = report.read_text(encoding="utf-8")
    assert f"threshold=+{cli.PUBLISHED_TNORM_THRESHOLD:.5f}" in text


def test_calibration_has_tnorm_fields():
    if not cli.CALIBRATION.exists():
        return  # calibration is machine state, built by `calibrate`
    obj = json.loads(cli.CALIBRATION.read_text(encoding="utf-8"))
    assert "t_threshold" in obj
    assert 0.0 < obj["eer"] < 0.2
    assert obj["cohort_size"] >= 2
    # the shipped cohort must exist for scoring to work
    assert cli.COHORT_PATH.exists()


def test_trial_t_score_ranks_the_right_speaker_first():
    """The speaker's own model must beat another cohort model on their audio."""
    if not cli.COHORT_PATH.exists():
        return
    from voiceauth import gmm_ubm
    obj = gmm_ubm.load(cli.COHORT_PATH)
    names, models = obj["speakers"], obj["models"]
    files = sorted((config.DATA_FEAT / "eval" / names[0]).glob("*.npy"))
    if not files:
        return
    X = np.load(files[0])
    ubm = cli.load_ubm()
    own = cli.trial_t_score(models[0], X, ubm, models)
    other = cli.trial_t_score(models[1], X, ubm, models)
    assert own["cohort_n"] == len(models)
    assert np.isfinite(own["raw"]) and own["cohort_sd"] > 0
    assert own["t"] > other["t"]


def test_parser_exposes_expected_commands():
    p = cli.build_parser()
    for cmd in ("register", "login", "enroll", "verify", "list", "delete",
                "selftest", "calibrate", "devices"):
        assert cmd in p._subparsers._group_actions[0].choices
