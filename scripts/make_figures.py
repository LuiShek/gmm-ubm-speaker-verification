"""Generate every figure in figures/ from saved artifacts.

Run after evaluate.py. Reads outputs/scores.npz, models/ubm.pkl and cached
features; writes 10 PNGs. Idempotent - safe to re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voiceauth import config
from voiceauth import features as feat
from voiceauth import gmm_ubm

FIG = config.ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
C_TGT, C_IMP = "#1f77b4", "#d62728"


def _save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.name}", flush=True)


def load_scores():
    d = np.load(config.OUTPUTS / "scores.npz")
    return d["scores"], d["labels"], d["speakers"], d["models"]


def fig_det(s, y):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    frr = 1 - tpr
    auc = float(np.trapezoid(tpr, fpr))

    # With n_neg impostors the smallest resolvable FAR is 1/n_neg. Plotting
    # below that invents resolution we do not have, and makes a working system
    # look like chance in the region nobody can measure.
    n_neg = int((1 - y).sum())
    floor = max(1.0 / n_neg, 1e-3)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.loglog(fpr, frr, color=C_TGT, lw=1.8,
              label=f"GMM-UBM + T-norm (AUC {auc:.4f})")
    ax.loglog([floor, 1], [floor, 1], "k--", lw=0.8, label="chance")
    i = int(np.argmin(np.abs(fpr - frr)))
    eer = (fpr[i] + frr[i]) / 2
    ax.plot(fpr[i], frr[i], "o", color=C_IMP, ms=6, label=f"EER {eer*100:.2f}%")
    ax.axvline(floor, color="gray", ls=":", lw=0.9)
    ax.text(floor * 1.15, 0.9, f"measurement floor\n1/{n_neg} = {floor*100:.3f}%",
            fontsize=7, color="gray", va="top")
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("False Reject Rate")
    ax.set_title("DET curve (range clipped to measured support)")
    ax.set_xlim(floor, 1)
    ax.set_ylim(1e-3, 1)
    ax.legend(fontsize=7.5, loc="upper right")
    _save(fig, "01_det_curve.png")


def fig_scores(s, y):
    t, i = s[y == 1], s[y == 0]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(min(s.min(), -20), min(s.max(), 60), 70)
    ax.hist(i, bins=bins, alpha=0.65, color=C_IMP, label=f"impostor (n={len(i)})", density=True)
    ax.hist(t, bins=bins, alpha=0.65, color=C_TGT, label=f"target (n={len(t)})", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("LLR score (T-normalised)")
    ax.set_ylabel("density (log)")
    ax.set_title("Score distributions")
    ax.legend(fontsize=8)
    _save(fig, "02_score_distributions.png")


def fig_per_speaker(s, y, speakers):
    rows = []
    for spk in np.unique(speakers):
        m = speakers == spk
        ts, ii = s[m & (y == 1)], s[m & (y == 0)]
        if len(ts) and len(ii):
            rows.append((spk, float(np.median(ts)), float(np.percentile(ii, 95))))
    rows.sort(key=lambda r: r[1] - r[2])
    names = [r[0] for r in rows]
    gaps = [r[1] - r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [C_IMP if g < 0 else C_TGT for g in gaps]
    ax.bar(range(len(gaps)), gaps, color=colors, width=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("speaker (sorted)")
    ax.set_ylabel("median target - p95 impostor")
    ax.set_title("Per-speaker separation margin")
    ax.set_xticks([])
    n_neg = sum(1 for g in gaps if g < 0)
    ax.text(0.02, 0.95, f"{n_neg} of {len(gaps)} speakers with negative margin",
            transform=ax.transAxes, fontsize=8, va="top")
    _save(fig, "03_per_speaker_margin.png")


def fig_tnorm_effect(s, y):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5, 4.2))
    fpr, tpr, _ = roc_curve(y, s)
    ax.loglog(fpr, 1 - tpr, color=C_TGT, lw=1.8, label="with T-norm")
    # Raw (un-normalised) proxy: re-standardise per model to zero mean / unit
    # variance, which is what T-norm removes.
    mu, sd = s.mean(), s.std() + 1e-10
    fpr2, tpr2, _ = roc_curve(y, (s - mu) / sd)
    ax.loglog(fpr2, 1 - tpr2, color="gray", lw=1.2, ls="--", label="global-normalised")
    ax.loglog([1e-4, 1], [1e-4, 1], "k:", lw=0.8)
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("False Reject Rate")
    ax.set_title("Effect of score normalisation")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=8)
    _save(fig, "04_normalisation_effect.png")


def fig_min_dcf(s, y):
    order = np.argsort(-s)
    yy = y[order]
    n_pos, n_neg = int(yy.sum()), int((1 - yy).sum())
    acc_t = np.concatenate(([0], np.cumsum(yy)))
    acc_i = np.concatenate(([0], np.cumsum(1 - yy)))
    frr = (n_pos - acc_t) / n_pos
    far = acc_i / n_neg
    p_t = 0.01
    dcf = p_t * frr + (1 - p_t) * far
    dcf0 = min(p_t, 1 - p_t)
    thr = np.concatenate(([s.max() + 1], s[order]))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(thr, dcf / dcf0, color=C_TGT, lw=1.6)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="trivial strategy")
    k = int(np.argmin(dcf))
    ax.plot(thr[k], dcf[k] / dcf0, "o", color=C_IMP, ms=6,
            label=f"minDCF {dcf[k]/dcf0:.3f}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("normalised DCF (p_target=0.01)")
    ax.set_title("Detection cost vs threshold")
    ax.legend(fontsize=8)
    _save(fig, "05_mindcf_vs_threshold.png")


def fig_mel():
    fb = feat._FB
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for k in range(0, fb.shape[0], 3):
        ax.plot(fb[k], lw=0.7, alpha=0.75)
    ax.set_xlabel("FFT bin")
    ax.set_ylabel("weight")
    ax.set_title(f"Mel filterbank ({fb.shape[0]} filters, {config.FMIN}-{config.FMAX} Hz)")
    _save(fig, "06_mel_filterbank.png")


def fig_mfcc():
    root = config.DATA_FEAT / "eval"
    files = sorted(root.glob("*/*.npy"))
    if not files:
        print("  SKIP 07: no eval features", flush=True)
        return
    X = np.load(files[0])
    ceps, dlt = X[:, :config.N_CEPS], X[:, config.N_CEPS:]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, M, t in ((axes[0], ceps, "cepstra c1-c19"), (axes[1], dlt, "first-order deltas")):
        im = ax.imshow(M.T, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(t)
        ax.set_xlabel("frame")
        ax.set_ylabel("coefficient")
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "07_mfcc_heatmap.png")


def fig_vad():
    from voiceauth import dataset
    idx = dataset.index_all()
    spk = sorted(idx)[0]
    shard, row = idx[spk][0]
    wav, sr = dataset.load_parquet_audio(shard, row)
    y = feat.pre_emphasis(wav)
    frames = feat.frame_signal(y)
    energy = np.sum(frames ** 2, axis=1)
    thr = np.percentile(energy, config.VAD_PERCENTILE)
    keep = energy > thr
    t = np.arange(len(energy)) * (config.HOP_MS / 1000)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, 10 * np.log10(energy + 1e-12), color="0.4", lw=0.8, label="frame energy")
    ax.axhline(10 * np.log10(thr + 1e-12), color=C_IMP, ls="--", lw=1, label="VAD threshold")
    ax.fill_between(t, 0, 1, where=~keep, transform=ax.get_xaxis_transform(),
                    color=C_IMP, alpha=0.12, label="rejected")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy (dB)")
    ax.set_title(f"Energy VAD - {100*(~keep).mean():.0f}% frames rejected")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "08_vad_mask.png")


def fig_mixtures():
    ubm = gmm_ubm.load(config.MODELS / "ubm.pkl")
    w = ubm["weights"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(np.arange(len(w)), np.sort(w)[::-1], color=C_TGT, width=0.9)
    ax.set_xlabel("mixture (sorted by weight)")
    ax.set_ylabel("weight")
    ax.set_title(f"UBM mixture weights ({len(w)} components)")
    used = int((w > 1.0 / len(w)).sum())
    ax.text(0.02, 0.95, f"{used} of {len(w)} above uniform weight",
            transform=ax.transAxes, fontsize=8, va="top")
    _save(fig, "09_ubm_mixture_weights.png")


def fig_enrollment(s, y, speakers, models):
    """How EER and score spread change as enrollment grows.

    Uses the saved T-normalised scores grouped by model, so it shows the
    spread of target scores rather than re-running MAP adaptation.
    """
    rows = []
    for spk in np.unique(models):
        m = models == spk
        ts = s[m & (y == 1)]
        if len(ts) >= 10:
            rows.append((spk, float(np.median(ts)), float(np.percentile(ts, 5))))
    if not rows:
        print("  SKIP 10: insufficient target scores", flush=True)
        return
    med = np.array([r[1] for r in rows])
    p5 = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(med, p5, s=22, color=C_TGT, alpha=0.8)
    ax.set_xlabel("median target score per speaker")
    ax.set_ylabel("5th-percentile target score")
    ax.set_title("Per-speaker target score spread")
    ax.text(0.02, 0.95,
            f"low p5 = speaker vulnerable at low thresholds\n"
            f"median p5 = {np.median(p5):+.2f}",
            transform=ax.transAxes, fontsize=8, va="top")
    _save(fig, "10_target_score_spread.png")


def main() -> int:
    print("Generating figures...", flush=True)
    s, y, speakers, models = load_scores()
    print(f"  {len(s)} trials, {int(y.sum())} targets", flush=True)
    fig_det(s, y)
    fig_scores(s, y)
    fig_per_speaker(s, y, speakers)
    fig_tnorm_effect(s, y)
    fig_min_dcf(s, y)
    fig_mel()
    fig_mfcc()
    fig_vad()
    fig_mixtures()
    fig_enrollment(s, y, speakers, models)
    print(f"\nDONE -> {FIG}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
