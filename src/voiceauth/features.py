"""
Front-end feature extraction for GMM-UBM speaker verification.

  waveform -> 20ms/10ms frames -> pre-emphasis -> Hamming -> FFT ->
  mel filterbank -> log -> DCT -> drop c0 -> deltas over +-2 -> energy VAD

Cepstra: 19 coefficients (c1..c19; c0 carries only frame energy).
Deltas: first-order orthogonal-polynomial regression, window +-2.
Output: 38-D vectors per frame.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.fft import dct

from . import config


# ---------------- mel filterbank ----------------

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int,
                   fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1)."""
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if ctr > lo:
            fb[m - 1, lo:ctr] = (np.arange(lo, ctr) - lo) / (ctr - lo)
        if hi > ctr:
            fb[m - 1, ctr:hi] = (hi - np.arange(ctr, hi)) / (hi - ctr)
    return fb


_FB = mel_filterbank(config.N_MELS, config.N_FFT, config.SR,
                     config.FMIN, config.FMAX)
_WINDOW = np.hamming(config.WIN_LEN).astype(np.float32)


# ---------------- framing ----------------

def frame_signal(y: np.ndarray) -> np.ndarray:
    """Split into overlapping frames, shape (n_frames, WIN_LEN)."""
    if len(y) < config.WIN_LEN:
        return np.empty((0, config.WIN_LEN), dtype=np.float32)
    n = 1 + (len(y) - config.WIN_LEN) // config.HOP_LEN
    idx = (np.arange(config.WIN_LEN)[None, :]
           + config.HOP_LEN * np.arange(n)[:, None])
    return y[idx].astype(np.float32)


def pre_emphasis(y: np.ndarray, coef: float = config.PREEMPH) -> np.ndarray:
    return np.append(y[0], y[1:] - coef * y[:-1]).astype(np.float32)


# ---------------- MFCC + deltas ----------------

def mfcc_from_waveform(y: np.ndarray) -> np.ndarray:
    """Waveform -> (T, 19) cepstra (c0 dropped). No deltas yet."""
    y = pre_emphasis(np.asarray(y, dtype=np.float32))
    frames = frame_signal(y)
    if frames.shape[0] == 0:
        return np.empty((0, config.N_CEPS), dtype=np.float32)
    frames = frames * _WINDOW
    spec = np.abs(np.fft.rfft(frames, n=config.N_FFT, axis=1)) ** 2
    mel = spec @ _FB.T                     # (T, n_mels)
    log_mel = np.log(np.maximum(mel, 1e-10))
    ceps = dct(log_mel, type=2, axis=1, norm="ortho")[:, : config.N_CEPS + 1]
    return ceps[:, 1:].astype(np.float32)  # drop c0, keep c1..c19


def deltas(feats: np.ndarray, win: int = config.DELTA_WIN) -> np.ndarray:
    """First-order deltas, orthogonal-polynomial regression over +-win."""
    T = feats.shape[0]
    if T == 0:
        return np.empty_like(feats)
    padded = np.pad(feats, ((win, win), (0, 0)), mode="edge")
    denom = 2.0 * sum(i * i for i in range(1, win + 1))
    d = np.zeros_like(feats)
    for i in range(1, win + 1):
        d += i * (padded[win + i: win + i + T] - padded[win - i: win - i + T])
    return (d / denom).astype(np.float32)


def energy_vad(feats: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Drop low-energy frames. Returns boolean keep-mask aligned to feats."""
    n = feats.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    y = pre_emphasis(np.asarray(y, dtype=np.float32))
    frames = frame_signal(y)
    # trim to feats length in case of off-by-one from framing on delta-padding
    m = min(frames.shape[0], n)
    energy = np.sum(frames[:m] ** 2, axis=1)
    if m < n:
        energy = np.concatenate([energy, np.full(n - m, energy[-1] if m else 0.0)])
    thr = np.percentile(energy, config.VAD_PERCENTILE)
    keep = energy > thr
    # safety: never drop more than (1 - MIN_RATIO) of frames
    if keep.sum() < max(1, int(config.VAD_MIN_RATIO * n)):
        keep = np.ones(n, dtype=bool)
    return keep


def extract_from_array(y: np.ndarray, sr: int = config.SR) -> np.ndarray:
    """Waveform array -> (T, 38) feature matrix. Same pipeline as extract()."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != config.SR:
        raise ValueError(f"expected {config.SR} Hz, got {sr}")

    ceps = mfcc_from_waveform(y)
    if ceps.shape[0] < config.DELTA_WIN * 2 + 1:
        return np.empty((0, config.FEAT_DIM), dtype=np.float32)

    d = deltas(ceps)
    feats = np.concatenate([ceps, d], axis=1)
    keep = energy_vad(feats, y)
    return feats[keep].astype(np.float32)


def extract(path: str) -> np.ndarray:
    """File path -> (T, 38) float32 feature matrix."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    return extract_from_array(y, sr)
