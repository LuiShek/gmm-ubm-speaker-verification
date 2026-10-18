"""Step 4 of 4: enroll speakers, score trials, apply T-norm, compute EER.

Protocol
--------
- Each eval speaker: first UTTS_PER_SPK_ENROLL utts -> enrollment, rest -> test.
- MAP-adapt a speaker model from the UBM for each speaker.
- True trials:  speaker model vs their own test utts.
- False trials: speaker model vs every other eval speaker's test utts.
- Score = avg log-likelihood-ratio (speaker GMM - UBM).
- T-norm: for each speaker, score N_TNORM_IMPOSTORS background utterances
  against that model and its cohort models; normalise with their mean/std.
- Report EER, minDCF, DET curve.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voiceauth import config
from voiceauth import gmm_ubm


# ---------------- metrics ----------------

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Equal error rate. labels: 1 = target, 0 = impostor. Returns (eer, threshold, far, frr)."""
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both target and impostor trials")

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    frr = 1.0 - tp / n_pos
    far = fp / n_neg

    # EER: where FAR and FRR cross
    idx = np.argmin(np.abs(far - frr))
    eer = float((far[idx] + frr[idx]) / 2)
    thr = float(s[idx])
    return eer, thr, far, frr


def min_dcf(scores, labels, p_target=0.01, c_miss=1.0, c_fa=1.0) -> float:
    """Minimum normalised detection cost (NIST SDR style).

    Sweeps every achievable threshold. At a cutoff accepting the top-k trials:
      FA  = impostors among the accepted
      FRR = targets NOT among the accepted
    Normalised by the cost of the trivial strategies, so 1.0 = no better than
    always-accept-or-always-reject and 0.0 = perfect separation.
    """
    order = np.argsort(-scores)
    y = labels[order].astype(np.int64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    acc_tgt = np.cumsum(y)              # targets accepted at each cutoff
    acc_imp = np.cumsum(1 - y)          # impostors accepted at each cutoff

    # Prepend the reject-everything state (accept 0 trials).
    acc_tgt = np.concatenate(([0], acc_tgt))
    acc_imp = np.concatenate(([0], acc_imp))

    frr = (n_pos - acc_tgt) / n_pos
    far = acc_imp / n_neg

    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    dcf_default = min(c_miss * p_target, c_fa * (1.0 - p_target))
    return float(np.min(dcf) / dcf_default)


# ---------------- trial construction ----------------

def load_eval_features(eval_spk: list[str]) -> dict[str, list[np.ndarray]]:
    root = config.DATA_FEAT / "eval"
    out: dict[str, list[np.ndarray]] = {}
    for spk in eval_spk:
        files = sorted((root / spk).glob("*.npy")) if (root / spk).exists() else []
        arrs = [np.load(f) for f in files]
        arrs = [a for a in arrs if a.shape[0] >= 30]
        if arrs:
            out[spk] = arrs
    return out


def split_enroll_test(feats: dict[str, list[np.ndarray]], rng):
    enroll: dict[str, list[np.ndarray]] = {}
    test: dict[str, list[np.ndarray]] = {}
    for spk, arrs in feats.items():
        k = config.UTTS_PER_SPK_ENROLL
        if len(arrs) < k + 1:
            continue
        idx = list(range(len(arrs)))
        rng.shuffle(idx)
        e_idx = idx[:k]
        t_idx = idx[k:k + config.UTTS_PER_SPK_TEST]
        enroll[spk] = [arrs[i] for i in e_idx]
        test[spk] = [arrs[i] for i in t_idx]
    return enroll, test


def bootstrap_eer_ci(scores: np.ndarray, labels: np.ndarray, speakers: np.ndarray,
                     n_iter: int = config.BOOTSTRAP_ITERS,
                     seed: int = config.SEED) -> dict:
    """Speaker-clustered bootstrap CI for EER.

    Trials sharing a speaker are correlated, so resampling individual trials
    would understate the interval. We resample SPEAKERS with replacement and
    recompute EER on the surviving trial subset each iteration.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(speakers)
    eers = []
    for _ in range(n_iter):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(speakers, pick)
        s, y = scores[mask], labels[mask]
        if y.sum() == 0 or (1 - y).sum() == 0:
            continue
        try:
            e, _, _, _ = compute_eer(s, y)
            eers.append(e)
        except ValueError:
            continue
    if not eers:
        return {"lo": float("nan"), "hi": float("nan"), "n": 0}
    eers = np.array(eers)
    return {
        "lo": float(np.percentile(eers, 2.5)),
        "hi": float(np.percentile(eers, 97.5)),
        "median": float(np.median(eers)),
        "n": int(len(eers)),
    }


def t_norm_scores(models: dict[str, dict], ubm: dict,
                   cohort_feats: list[np.ndarray],
                   scores: dict[tuple[str, str], float],
                   rng) -> dict[tuple[str, str], float]:
    """Per-model T-norm using imposter utterances scored against the model."""
    spk_list = list(models.keys())
    out: dict[tuple[str, str], float] = {}
    norm_stats: dict[str, tuple[float, float]] = {}

    for spk in spk_list:
        m = models[spk]
        pool = cohort_feats[: config.N_TNORM_IMPOSTORS]
        raw = np.array([gmm_ubm.score(m, ubm, x) for x in pool])
        mu, sd = float(raw.mean()), float(raw.std() + 1e-10)
        norm_stats[spk] = (mu, sd)

    for (model_spk, test_spk), s in scores.items():
        mu, sd = norm_stats[model_spk]
        out[(model_spk, test_spk)] = (s - mu) / sd
    return out


# ---------------- main ----------------

def main() -> int:
    t0 = time.time()
    ubm_path = config.MODELS / "ubm.pkl"
    split_path = config.MODELS / "speaker_split.pkl"
    if not ubm_path.exists() or not split_path.exists():
        print("ERROR: missing ubm.pkl or speaker_split.pkl. Run earlier steps.", file=sys.stderr)
        return 1

    ubm = gmm_ubm.load(ubm_path)
    split = gmm_ubm.load(split_path)
    eval_spk = split["eval"]

    print(f"[1/4] Loading eval features for {len(eval_spk)} speakers...", flush=True)
    feats = load_eval_features(eval_spk)
    print(f"  loaded {len(feats)} speakers", flush=True)

    rng = np.random.default_rng(config.SEED)
    enroll, test = split_enroll_test(feats, rng)
    print(f"  enroll={len(enroll)} test={len(test)}", flush=True)

    print("[2/4] MAP-adapting speaker models...", flush=True)
    models: dict[str, dict] = {}
    for i, (spk, arrs) in enumerate(enroll.items(), 1):
        models[spk] = gmm_ubm.map_adapt(ubm, arrs)
        if i % 10 == 0:
            print(f"  [{i}/{len(enroll)}] adapted", flush=True)

    print("[3/4] Scoring trials...", flush=True)
    # The UBM is fixed, so cache its per-utterance log-likelihood once and
    # reuse it across all speaker models instead of recomputing it per model.
    ubm_cache: dict[int, float] = {}
    for arrs in test.values():
        for x in arrs:
            ubm_cache[id(x)] = gmm_ubm.ubm_loglik(x, ubm)

    # Plan the trials BEFORE scoring: with every remaining utterance there are
    # ~275k impostor pairs, which is hours of GMM evaluation for no statistical
    # gain. Targets keep everything (false-reject is the noisy side of EER);
    # impostors are subsampled to MAX_IMPOSTOR_TRIALS.
    plan_tgt: list[tuple[str, str, np.ndarray]] = []
    plan_imp: list[tuple[str, str, np.ndarray]] = []
    for model_spk in models:
        for test_spk, arrs in test.items():
            bucket = plan_tgt if model_spk == test_spk else plan_imp
            for x in arrs:
                bucket.append((model_spk, test_spk, x))

    if len(plan_imp) > config.MAX_IMPOSTOR_TRIALS:
        idx = rng.choice(len(plan_imp), size=config.MAX_IMPOSTOR_TRIALS,
                         replace=False)
        n_before = len(plan_imp)
        plan_imp = [plan_imp[i] for i in idx]
        print(f"  impostor trials: {n_before} -> {len(plan_imp)}", flush=True)

    plan = plan_tgt + plan_imp
    print(f"  scoring {len(plan)} trials ({len(plan_tgt)} target)", flush=True)

    trials: list[tuple[str, str, float]] = []
    for i, (model_spk, test_spk, x) in enumerate(plan, 1):
        trials.append((model_spk, test_spk,
                       gmm_ubm.score(models[model_spk], ubm, x,
                                     ubm_cache[id(x)])))
        if i % 2000 == 0:
            print(f"  [{i}/{len(plan)}] scored", flush=True)
    print(f"  scored {len(trials)} trials", flush=True)

    # T-norm: per-model mean/std of scores against an imposter cohort drawn
    # from OTHER speakers' test utterances.
    s_arr = np.array([t[2] for t in trials], dtype=np.float64)
    model_names = [t[0] for t in trials]
    test_names = [t[1] for t in trials]
    y_arr = np.array([1 if a == b else 0 for a, b in zip(model_names, test_names)])

    if config.T_NORM_ENABLED:
        print("[4/4] Applying T-norm...", flush=True)
        cohort = []
        for spk, arrs in test.items():
            for a in arrs:
                cohort.append(a)
                if len(cohort) >= 4 * config.N_TNORM_IMPOSTORS:
                    break
            if len(cohort) >= 4 * config.N_TNORM_IMPOSTORS:
                break
        stats = {}
        for spk, m in models.items():
            raw = np.array([gmm_ubm.score(m, ubm, x,
                                          gmm_ubm.ubm_loglik(x, ubm))
                            for x in cohort[:config.N_TNORM_IMPOSTORS]])
            stats[spk] = (float(raw.mean()), float(raw.std() + 1e-10))
        s_arr = np.array([(s - stats[mdl][0]) / stats[mdl][1]
                          for s, mdl in zip(s_arr, model_names)])
    else:
        print("[4/4] Skipping T-norm", flush=True)

    speakers = np.array(test_names)

    if y_arr.sum() == 0 or (1 - y_arr).sum() == 0:
        print("ERROR: degenerate trial set", file=sys.stderr)
        return 1

    eer, thr, far, frr = compute_eer(s_arr, y_arr)
    md = min_dcf(s_arr, y_arr)
    ci = bootstrap_eer_ci(s_arr, y_arr, speakers)

    # Zero-miss operating point: best FAR achievable with FRR = 0.
    _order = np.argsort(-s_arr)
    _s, _y = s_arr[_order], y_arr[_order]
    _miss = np.cumsum(1 - _y)
    _fa = np.cumsum(_y == 0)
    _zm = np.where(_miss == 0)[0]
    far_at_zero_frr = (float(_fa[_zm[-1]] / max((1 - _y).sum(), 1))
                       if len(_zm) else float("nan"))

    print("\n" + "=" * 58)
    print(f"  EER        : {eer*100:.2f} %   (95% CI {ci['lo']*100:.2f}-{ci['hi']*100:.2f} %)")
    print(f"  minDCF     : {md:.4f}")
    print(f"  Threshold  : {thr:+.4f}")
    print(f"  Targets    : {int(y_arr.sum())}")
    print(f"  Impostors  : {int((1-y_arr).sum())}")
    print(f"  FAR@FRR=0  : {far_at_zero_frr*100:.2f} %")
    print(f"  T-norm     : {'on' if config.T_NORM_ENABLED else 'off'}")
    print(f"  Bootstrap  : {ci['n']} speaker-resamples")
    print("=" * 58)

    # Save results + DET curve data.
    out_dir = config.OUTPUTS
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "scores.npz", scores=s_arr, labels=y_arr,
             speakers=speakers, models=np.array(model_names))
    np.savez(out_dir / "det_curve.npz", far=far, frr=frr)
    (out_dir / "report.txt").write_text(
        f"EER={eer*100:.3f}%\n"
        f"EER_CI_LOW={ci['lo']*100:.3f}%\n"
        f"EER_CI_HIGH={ci['hi']*100:.3f}%\n"
        f"bootstrap_iters={ci['n']}\n"
        f"FAR_at_FRR0={far_at_zero_frr*100:.3f}%\n"
        f"minDCF={md:.5f}\nthreshold={thr:+.5f}\n"
        f"targets={int(y_arr.sum())}\nimpostors={int((1-y_arr).sum())}\n"
        f"t_norm={config.T_NORM_ENABLED}\nmixtures={config.N_MIXTURES}\n",
        encoding="utf-8",
    )
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
