"""Sanity checks for the voiceauth package.

Deliberately small and dependency-light: each test fails loudly if the
math or the shapes break. Run with:  py -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voiceauth import config, features, gmm_ubm


# ---------------- features ----------------

def test_mel_filterbank_shape_and_nonnegative():
    fb = features.mel_filterbank(40, 512, 16000, 300, 8000)
    assert fb.shape == (40, 257)
    assert (fb >= 0).all()
    # every filter must actually cover some bins
    assert (fb.sum(axis=1) > 0).all()


def test_frame_signal_count():
    y = np.zeros(16000, dtype=np.float32)
    frames = features.frame_signal(y)
    expected = 1 + (16000 - config.WIN_LEN) // config.HOP_LEN
    assert frames.shape == (expected, config.WIN_LEN)


def test_deltas_of_constant_is_zero():
    feats = np.ones((50, 19), dtype=np.float32)
    d = features.deltas(feats)
    assert d.shape == feats.shape
    assert np.allclose(d, 0.0, atol=1e-6)


def test_deltas_of_linear_ramp_is_constant():
    # deltas of a straight line are the slope, everywhere except the edges
    t = np.arange(60, dtype=np.float32)[:, None]
    feats = np.tile(t, (1, 19))
    d = features.deltas(feats)
    interior = d[config.DELTA_WIN: -config.DELTA_WIN]
    assert np.allclose(interior, 1.0, atol=1e-5)


def test_extract_from_array_shape():
    rng = np.random.default_rng(0)
    y = rng.normal(size=16000).astype(np.float32) * 0.1
    X = features.extract_from_array(y, 16000)
    assert X.ndim == 2
    assert X.shape[1] == config.FEAT_DIM
    assert np.isfinite(X).all()


def test_extract_rejects_wrong_sample_rate():
    y = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="expected"):
        features.extract_from_array(y, 8000)


# ---------------- GMM math ----------------

def test_comp_logpdf_matches_naive():
    rng = np.random.default_rng(1)
    T, D, K = 20, 4, 3
    X = rng.normal(size=(T, D))
    means = rng.normal(size=(K, D))
    covars = np.abs(rng.normal(size=(K, D))) + 0.5

    fast = gmm_ubm._comp_logpdf(X, means, covars)

    twopi = 2.0 * np.pi
    naive = np.empty((T, K))
    for t in range(T):
        for k in range(K):
            diff = X[t] - means[k]
            naive[t, k] = -0.5 * (np.sum(np.log(twopi * covars[k]))
                                  + np.sum(diff * diff / covars[k]))

    # float64 path: exact up to machine epsilon
    assert np.allclose(fast, naive, rtol=0, atol=1e-12)


def test_map_adapt_with_zero_relevance_returns_ubm():
    rng = np.random.default_rng(2)
    K, D = 4, 3
    ubm = {
        "weights": np.full(K, 1.0 / K),
        "means": rng.normal(size=(K, D)),
        "covars": np.abs(rng.normal(size=(K, D))) + 0.5,
    }
    feats = [rng.normal(size=(50, D))]
    # r -> inf means alpha -> 0, so the model must collapse back to the UBM
    adapted = gmm_ubm.map_adapt(ubm, feats, r=1e12)
    assert np.allclose(adapted["means"], ubm["means"], atol=1e-6)


def test_map_adapt_moves_means_toward_data():
    rng = np.random.default_rng(3)
    K, D = 4, 3
    ubm = {
        "weights": np.full(K, 1.0 / K),
        "means": rng.normal(size=(K, D)),
        "covars": np.abs(rng.normal(size=(K, D))) + 0.5,
    }
    feats = [ubm["means"][0] + 0.01 * rng.normal(size=(200, D))]
    adapted = gmm_ubm.map_adapt(ubm, feats, r=1.0)
    # at least one component must have shifted from the UBM
    assert not np.allclose(adapted["means"], ubm["means"], atol=1e-4)


def test_score_is_zero_when_model_equals_ubm():
    rng = np.random.default_rng(4)
    K, D = 3, 4
    ubm = {
        "weights": np.full(K, 1.0 / K),
        "means": rng.normal(size=(K, D)),
        "covars": np.abs(rng.normal(size=(K, D))) + 0.5,
    }
    X = rng.normal(size=(40, D))
    s = gmm_ubm.score(ubm, ubm, X)
    assert abs(s) < 1e-9


# ---------------- metrics ----------------

def test_compute_eer_perfect_separation():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import evaluate as ev

    scores = np.array([5.0, 4.0, 3.0, -1.0, -2.0, -3.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    eer, thr, far, frr = ev.compute_eer(scores, labels)
    assert eer == 0.0
    assert thr == scores[2]


def test_min_dcf_bounds():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import evaluate as ev

    # perfect separation -> 0
    s = np.array([5.0, 4.0, 3.0, -1.0, -2.0, -3.0])
    y = np.array([1, 1, 1, 0, 0, 0])
    assert ev.min_dcf(s, y) == pytest.approx(0.0, abs=1e-9)

    # inverted separation -> worse than trivial, capped at 1.0
    assert ev.min_dcf(-s, y) == pytest.approx(1.0, abs=1e-6)

    # a working system must land strictly between
    rng = np.random.default_rng(5)
    s2 = rng.normal(loc=np.r_[np.ones(500), np.zeros(500)], size=1000)
    y2 = np.r_[np.ones(500, int), np.zeros(500, int)]
    v = ev.min_dcf(s2, y2)
    assert 0.0 < v < 1.0
