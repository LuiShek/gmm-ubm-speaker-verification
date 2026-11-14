"""Ablation: what does each part of the GMM-UBM pipeline actually buy?

Every system is scored on the SAME trials (same split, same target trials, same
impostor subsample, same seed as scripts/evaluate.py), and reported under three
scoring rules:

  raw          score as produced (cosine / log-likelihood / LLR)
  tnorm-1spk   z-norm against the cohort evaluate.py actually builds, which is
               a block of utterances from ONE speaker. Kept because it is what
               produced the published 5.42%, so the discrepancy is auditable.
  tnorm-cohort z-norm against one utterance from each of the other 49 eval
               speakers, excluding the target speaker - the standard T-norm.

Systems:
  cosine      cosine between mean 38-D feature vectors. No model at all.
  decoupled   one 32-mixture diagonal GMM per speaker, no UBM, mean
              log-likelihood: the naive approach the UBM is meant to fix.
  gmm-ubm     the project system: MAP adaptation from the UBM + LLR.

The gmm-ubm numbers are recomputed here from scratch and cross-checked against
outputs/report.txt, so this script cannot silently disagree with the headline.

Usage
-----
  py scripts/baselines.py                    # ~13 min
  py scripts/baselines.py --fast             # skip GMM-UBM, ~2 min
  py scripts/baselines.py --json out.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voiceauth import config, gmm_ubm

# evaluate.py lives beside this file; reuse its trial protocol and metrics
# rather than re-implementing them, so the two can never drift apart.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate import (  # noqa: E402
    compute_eer,
    load_eval_features,
    min_dcf,
    split_enroll_test,
)


# ---------------- trial plan (identical to evaluate.py) ----------------

def plan_trials(enroll: dict, test: dict, rng) -> tuple[list, list]:
    """Rebuild evaluate.py's trial plan, including its impostor subsample."""
    plan_tgt, plan_imp = [], []
    for model_spk in enroll:
        for test_spk, arrs in test.items():
            bucket = plan_tgt if model_spk == test_spk else plan_imp
            for x in arrs:
                bucket.append((model_spk, test_spk, x))
    if len(plan_imp) > config.MAX_IMPOSTOR_TRIALS:
        idx = rng.choice(len(plan_imp), size=config.MAX_IMPOSTOR_TRIALS,
                         replace=False)
        plan_imp = [plan_imp[i] for i in idx]
    return plan_tgt, plan_imp


def cohort_as_built(test: dict) -> list[np.ndarray]:
    """The cohort evaluate.py actually builds: a block from one speaker.

    evaluate.py fills a 4x buffer by walking `test` in order and stopping, so
    all N_TNORM_IMPOSTORS entries come from the first speaker in the dict.
    """
    pool: list[np.ndarray] = []
    for _spk, arrs in test.items():
        for a in arrs:
            pool.append(a)
            if len(pool) >= 4 * config.N_TNORM_IMPOSTORS:
                break
        if len(pool) >= 4 * config.N_TNORM_IMPOSTORS:
            break
    return pool[: config.N_TNORM_IMPOSTORS]


def cohort_by_speaker(test: dict, exclude: str,
                      n: int) -> list[np.ndarray]:
    """Standard T-norm cohort: one utterance from each other speaker."""
    out: list[np.ndarray] = []
    for spk, arrs in test.items():
        if spk == exclude or not arrs:
            continue
        out.append(arrs[0])
        if len(out) >= n:
            break
    return out


# ---------------- normalisation ----------------

def stats(scores: np.ndarray) -> tuple[float, float]:
    return float(scores.mean()), float(scores.std() + 1e-10)


def apply_stats(scores: np.ndarray, model_names: list[str],
                table: dict[str, tuple[float, float]]) -> np.ndarray:
    out = np.empty_like(scores)
    for spk, (mu, sd) in table.items():
        mask = np.array([m == spk for m in model_names])
        out[mask] = (scores[mask] - mu) / sd
    return out


# ---------------- the systems ----------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine_scorer(enroll: dict) -> callable:
    models = {spk: _unit(np.mean([a.mean(axis=0) for a in arrs], axis=0))
              for spk, arrs in enroll.items()}
    means: dict[int, np.ndarray] = {}

    def score(spk: str, x: np.ndarray) -> float:
        m = means.get(id(x))
        if m is None:
            m = _unit(x.mean(axis=0))
            means[id(x)] = m
        return float(np.dot(models[spk], m))
    return score


def decoupled_scorer(enroll: dict, n_components: int = 32) -> callable:
    from sklearn.mixture import GaussianMixture

    models = {}
    for spk, arrs in enroll.items():
        g = GaussianMixture(n_components=n_components, covariance_type="diag",
                            max_iter=100, reg_covar=config.COVARIANCE_FLOOR,
                            random_state=config.SEED)
        g.fit(np.concatenate(arrs, axis=0))
        models[spk] = g

    cache: dict[tuple[int, str], float] = {}

    def score(spk: str, x: np.ndarray) -> float:
        key = (id(x), spk)
        if key not in cache:
            cache[key] = float(np.mean(models[spk].score_samples(x)))
        return cache[key]
    return score


def gmm_ubm_scorer(enroll: dict, ubm: dict) -> callable:
    print("    adapting speaker models ...", flush=True)
    models = {}
    for i, (spk, arrs) in enumerate(enroll.items(), 1):
        models[spk] = gmm_ubm.map_adapt(ubm, arrs)
        if i % 10 == 0:
            print(f"      [{i}/{len(enroll)}] adapted", flush=True)

    ubm_ll: dict[int, float] = {}

    def score(spk: str, x: np.ndarray) -> float:
        u = ubm_ll.get(id(x))
        if u is None:
            u = gmm_ubm.ubm_loglik(x, ubm)
            ubm_ll[id(x)] = u
        return float(gmm_ubm.score(models[spk], ubm, x, u))
    return score


# ---------------- reporting ----------------

def evaluate_system(name: str, score, enroll: dict, test: dict,
                    plan: list, y: np.ndarray) -> tuple[list[dict], np.ndarray]:
    model_names = [m for m, _t, _x in plan]
    raw = np.array([score(m, x) for m, _t, x in plan])

    built = cohort_as_built(test)
    stats_built = {spk: stats(np.array([score(spk, x) for x in built]))
                   for spk in enroll}

    stats_coh: dict[str, tuple[float, float]] = {}
    for spk in enroll:
        coh = cohort_by_speaker(test, spk, config.N_TNORM_IMPOSTORS)
        stats_coh[spk] = stats(np.array([score(spk, x) for x in coh]))

    rows = []
    for label, s in (("raw", raw),
                     ("tnorm-1spk", apply_stats(raw, model_names, stats_built)),
                     ("tnorm-cohort", apply_stats(raw, model_names, stats_coh))):
        eer, _thr, _, _ = compute_eer(s, y)
        rows.append({"system": name, "norm": label, "eer": float(eer),
                     "min_dcf": float(min_dcf(s, y))})
    return rows, raw


def published() -> dict | None:
    p = config.OUTPUTS / "report.txt"
    if not p.exists():
        return None
    kv = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    try:
        return {"eer": float(kv["EER"].rstrip("%")) / 100.0,
                "min_dcf": float(kv["minDCF"])}
    except (KeyError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    fast = "--fast" in argv
    out_json = None
    if "--json" in argv:
        i = argv.index("--json")
        if i + 1 >= len(argv):
            print("error: --json needs a path", file=sys.stderr)
            return 1
        out_json = Path(argv[i + 1])

    t0 = time.time()
    split = gmm_ubm.load(config.MODELS / "speaker_split.pkl")
    print(f"loading features for {len(split['eval'])} eval speakers ...",
          end="", flush=True)
    feats = load_eval_features(split["eval"])
    rng = np.random.default_rng(config.SEED)
    enroll, test = split_enroll_test(feats, rng)
    print(f" {len(enroll)} speakers", flush=True)

    plan_tgt, plan_imp = plan_trials(enroll, test, rng)
    plan = plan_tgt + plan_imp
    y = np.array([1 if m == t else 0 for m, t, _x in plan], dtype=np.int64)
    built = cohort_as_built(test)
    owners = {spk for spk, arrs in test.items() if any(a is b for b in built
                                                       for a in arrs)}
    print(f"trials: {len(plan)} ({int(y.sum())} target / "
          f"{int((1 - y).sum())} impostor)", flush=True)
    print(f"published cohort: {len(built)} utterances from "
          f"{len(owners)} speaker(s) -> {sorted(owners)}", flush=True)

    rows: list[dict] = []
    raw_gmm = None

    print("  [1/3] cosine baseline ...", flush=True)
    r, _ = evaluate_system("cosine (mean cepstra)", cosine_scorer(enroll),
                           enroll, test, plan, y)
    rows += r

    print("  [2/3] decoupled GMM, no UBM (32 mixtures) ...", flush=True)
    r, _ = evaluate_system("decoupled GMM (no UBM)", decoupled_scorer(enroll),
                           enroll, test, plan, y)
    rows += r

    if fast:
        print("  [3/3] GMM-UBM skipped (--fast)", flush=True)
    else:
        print("  [3/3] GMM-UBM (MAP + LLR) ...", flush=True)
        ubm = gmm_ubm.load(config.MODELS / "ubm.pkl")
        r, raw_gmm = evaluate_system("GMM-UBM (project)",
                                     gmm_ubm_scorer(enroll, ubm),
                                     enroll, test, plan, y)
        rows += r
        pub = published()
        got = [x for x in r if x["norm"] == "tnorm-1spk"][0]
        if pub:
            d = abs(got["eer"] - pub["eer"])
            print(f"    cross-check vs report.txt (1-speaker cohort): "
                  f"EER {got['eer']*100:.2f}% vs {pub['eer']*100:.2f}%  "
                  f"[{'OK' if d < 2e-3 else 'MISMATCH'}]", flush=True)

    print()
    print(f"{'system':28} {'scoring':14} {'EER':>7} {'minDCF':>8}")
    print("-" * 62)
    for r in rows:
        print(f"{r['system'][:28]:28} {r['norm']:14} "
              f"{r['eer'] * 100:6.2f}% {r['min_dcf']:8.4f}")
    print()
    print("Same speakers, same trials, same seed as scripts/evaluate.py.")
    print("`raw` isolates model quality. The two tnorm rows isolate score")
    print("normalisation: `tnorm-1spk` is what evaluate.py builds today,")
    print("`tnorm-cohort` uses one utterance per other speaker (standard).")

    obj = {"trials": int(len(plan)), "targets": int(y.sum()),
           "impostors": int((1 - y).sum()),
           "cohort_built_speakers": sorted(owners),
           "impostor_cap": config.MAX_IMPOSTOR_TRIALS, "seed": config.SEED,
           "systems": rows, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if out_json:
        out_json.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"\nwritten: {out_json}")

    if raw_gmm is not None:
        np.savez(config.OUTPUTS / "baselines_scores.npz",
                 scores_raw=raw_gmm, labels=y,
                 models=np.array([m for m, _t, _x in plan]))
        print(f"written: {config.OUTPUTS / 'baselines_scores.npz'}")

    print(f"\nDONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
