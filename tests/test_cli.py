"""Checks for the CLI plumbing (no model training, no audio hardware).

Run with:  py -m pytest tests/ -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_missing_cohort_explains_how_to_build_it():
    """calibration.json ships; cohort_models.pkl is gitignored by design.

    A fresh clone therefore has no cohort, and must fail with an instruction
    rather than a traceback. Asserting `COHORT_PATH.exists()` here instead
    would make the suite fail on every clean checkout.
    """
    if cli.COHORT_PATH.exists():
        return
    with pytest.raises(SystemExit) as exc:
        cli.load_cohort_models()
    assert "calibrate" in str(exc.value)


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


def test_baseline_cohort_spans_speakers_and_excludes_target():
    """The T-norm cohort must not be a block from one speaker."""
    sys.path.insert(0, str(config.ROOT / "scripts"))
    from baselines import cohort_by_speaker

    test = {f"spk{i}": [np.zeros((30, config.FEAT_DIM)) + i for _ in range(5)]
            for i in range(6)}
    coh = cohort_by_speaker(test, exclude="spk0", n=10)
    assert len(coh) == 5                      # one per other speaker only
    for x in coh:
        assert float(x[0, 0]) != 0.0          # spk0's zeros never appear
    assert len({float(x[0, 0]) for x in coh}) == 5


def test_baselines_agree_with_published_eer():
    """scripts/baselines.py must not contradict outputs/report.txt."""
    p = config.OUTPUTS / "baselines.json"
    r = config.OUTPUTS / "report.txt"
    if not p.exists() or not r.exists():
        return
    obj = json.loads(p.read_text(encoding="utf-8"))
    kv = dict(line.split("=", 1) for line in
              r.read_text(encoding="utf-8").splitlines() if "=" in line)
    pub = float(kv["EER"].rstrip("%")) / 100.0
    row = [s for s in obj.get("systems", [])
           if str(s.get("system", "")).startswith("GMM-UBM")
           and s.get("norm") == "T-norm"]
    if not row:
        return
    assert abs(row[0]["eer"] - pub) < 2e-3, (
        f"baselines say {row[0]['eer']:.4f}, report.txt says {pub:.4f}")


def test_parser_exposes_expected_commands():
    p = cli.build_parser()
    for cmd in ("register", "login", "record", "enroll", "verify", "list",
                "delete", "selftest", "calibrate", "calibrate-user", "devices"):
        assert cmd in p._subparsers._group_actions[0].choices


def test_login_pools_requested_takes_into_one_trial(monkeypatch):
    """--takes N must concatenate N captures and score exactly one trial."""
    from types import SimpleNamespace

    frames = np.zeros((120, config.FEAT_DIM), dtype=np.float32)
    seen: list[int] = []

    monkeypatch.setattr(cli, "load_speaker",
                        lambda name: {"name": name, "created": "2026-01-01"})
    monkeypatch.setattr(cli, "_countdown", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_capture_take",
                        lambda label, secs, dev, attempts=3: frames.copy())
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    monkeypatch.setattr(cli, "print_trial", lambda spk, r: None)

    def fake_score(spk, X):
        seen.append(int(X.shape[0]))
        return {"accept": True}

    monkeypatch.setattr(cli, "score_trial", fake_score)

    args = SimpleNamespace(user="me", audio=None, takes=3, attempts=2,
                           seconds=1.0, device=None, json=False)
    assert cli.cmd_login(args) == 0
    assert seen == [360]        # three 120-frame captures, one decision


def test_login_denies_after_all_attempts(monkeypatch):
    from types import SimpleNamespace

    frames = np.zeros((120, config.FEAT_DIM), dtype=np.float32)
    decisions: list[int] = []

    monkeypatch.setattr(cli, "load_speaker",
                        lambda name: {"name": name, "created": "2026-01-01"})
    monkeypatch.setattr(cli, "_countdown", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_capture_take",
                        lambda label, secs, dev, attempts=3: frames.copy())
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    monkeypatch.setattr(cli, "print_trial", lambda spk, r: None)

    def reject(spk, X):
        decisions.append(1)
        return {"accept": False}

    monkeypatch.setattr(cli, "score_trial", reject)

    args = SimpleNamespace(user="me", audio=None, takes=1, attempts=3,
                           seconds=1.0, device=None, json=False)
    assert cli.cmd_login(args) == 2
    assert len(decisions) == 3


def test_record_parser_requires_an_output_path():
    p = cli.build_parser()
    try:
        p.parse_args(["record"])
    except SystemExit:
        pass
    else:
        raise AssertionError("record without --out should be a usage error")
    a = p.parse_args(["record", "--out", "take.wav", "--seconds", "6"])
    assert a.out == "take.wav" and a.seconds == 6.0


def test_choose_threshold_splits_separable_scores():
    thr, margin = cli._choose_threshold([5.0, 6.0], [1.0, 2.0])
    assert thr == 3.5           # midpoint of worst genuine and best impostor
    assert margin == 3.0


def test_choose_threshold_refuses_overlapping_scores():
    # impostor matches the genuine user: no threshold can separate them
    thr, margin = cli._choose_threshold([4.0, 5.0], [4.5, 7.0])
    assert thr is None
    assert margin < 0


def test_choose_threshold_needs_both_sides():
    assert cli._choose_threshold([], [1.0])[0] is None
    assert cli._choose_threshold([1.0], [])[0] is None
