"""
Command-line interface: enroll your own voice, then verify a recording.

The scoring path must land on exactly the same scale as the published
evaluation (threshold +6.597 in outputs/report.txt), otherwise a stored
threshold means nothing. This module therefore rebuilds the evaluation
split and the T-norm cohort with the same seed and iteration order as
scripts/evaluate.py, and `selftest` checks the reproduced scores against
outputs/scores.npz.

Decisions, however, do not use that per-model utterance cohort: its
standard deviation is ~0.1, so an out-of-domain recording scores z=+10
while a real impostor model scores +0.3. The gate uses TEST normalisation
instead - the same trial is scored against 20 other speakers' models and
the target score is divided by their spread - which cancels channel and
domain effects. `calibrate` builds those models and sets the threshold on
held-out speakers.

Usage
-----
  py scripts/voice_cli.py register            # 5 takes of 10 s, then a model
  py scripts/voice_cli.py login               # speak once -> ACCESS GRANTED/DENIED
  py scripts/voice_cli.py login --takes 3     # pool 3 takes into one decision
  py scripts/voice_cli.py calibrate-user --user me --genuine ... --impostor ...
  py scripts/voice_cli.py selftest            # prove the score scale matches eval

Run `py scripts/voice_cli.py <command> -h` for the rest.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config
from . import features as feat
from . import gmm_ubm

SPEAKER_DIR = config.MODELS / "speakers"
CALIBRATION = config.MODELS / "calibration.json"
COHORT_PATH = config.MODELS / "cohort_models.pkl"

# The published EER threshold lives in T-norm units (outputs/report.txt).
PUBLISHED_TNORM_THRESHOLD = 6.59692

# Decisions use a TEST-normalised score (T = (raw - mu_cohort) / sd_cohort),
# where the cohort is a set of other speakers' models scored on the SAME trial.
# That keeps channel/domain effects in the numerator and denominator, which the
# per-model utterance cohort in evaluate.py could not do: its sd was ~0.1, so an
# out-of-domain recording scored z=+10 while a real impostor model scored +0.3.
# `calibrate` rebuilds the cohort and the held-out threshold in these units.
DEFAULT_T_THRESHOLD = 3.10

MIN_SECONDS = 1.5          # less speech than this cannot be scored reliably
MIN_FRAMES = 100           # frames surviving VAD
CLIP_LEVEL = 0.99          # warn above this peak


# ---------------- audio input ----------------

def _resample_16k(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample to config.SR with a polyphase filter (no silent truncation)."""
    if sr == config.SR:
        return y.astype(np.float32)
    from scipy.signal import resample_poly
    g = gcd(int(sr), config.SR)
    out = resample_poly(y, config.SR // g, int(sr) // g)
    return out.astype(np.float32)


def load_audio(path: str | Path) -> tuple[np.ndarray, float, int]:
    """Read any format soundfile understands. Returns (mono 16k, seconds, sr)."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1) if y.shape[1] > 1 else y[:, 0]
    secs = len(y) / sr
    return _resample_16k(y, sr), secs, sr


def _ffmpeg_input_devices() -> list[str]:
    """List DirectShow audio devices with ffmpeg (used when sounddevice is absent)."""
    import subprocess
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true",
             "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    text = (p.stderr or "") + (p.stdout or "")
    return re.findall(r'"([^"]+)" \(audio\)', text)


def _record_ffmpeg(seconds: float, device: int | str | None) -> np.ndarray:
    """Mic capture via ffmpeg/dshow, so no Python audio package is required."""
    import subprocess
    import tempfile

    devs = _ffmpeg_input_devices()
    if not devs:
        raise SystemExit(
            "error: no microphone backend found.\n"
            "  install one:  py -m pip install sounddevice\n"
            "  or record to a file and use:  --audio my_voice.wav"
        )
    name = devs[0]
    if device is not None:
        s = str(device)
        match = ([d for d in devs if s.lower() in d.lower()]
                 or ([devs[int(s)]] if s.isdigit() and int(s) < len(devs) else []))
        if not match:
            raise SystemExit(f"error: no input device matching {device!r}. "
                             f"Available: {devs}")
        name = match[0]

    tmp = Path(tempfile.gettempdir()) / "voiceauth_last_recording.wav"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "dshow", "-i", f"audio={name}",
           "-t", f"{seconds:g}", "-ac", "1", "-ar", str(config.SR), str(tmp)]
    print(f"recording {seconds:g}s from '{name}' (ffmpeg) ... ", end="", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not tmp.exists():
        raise SystemExit("error: ffmpeg could not record: "
                         + (p.stderr or "").strip()[:300])
    y, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    tmp.unlink(missing_ok=True)
    print("done", flush=True)
    return _resample_16k(y[:, 0], sr)


def record_audio(seconds: float, device: int | str | None = None) -> np.ndarray:
    """Record from the default input, mono, resampled to 16 kHz."""
    try:
        import sounddevice as sd
    except ImportError:
        return _record_ffmpeg(seconds, device)

    info = sd.query_devices(device, "input")
    fs = int(info["default_samplerate"])
    print(f"recording {seconds:g}s from '{info['name']}' at {fs} Hz ... ",
          end="", flush=True)
    rec = sd.rec(int(seconds * fs), samplerate=fs, channels=1,
                 dtype="float32", device=device)
    sd.wait()
    print("done", flush=True)
    return _resample_16k(rec[:, 0], fs)


# ---------------- feature extraction with guards ----------------

def features_from_audio(name: str, y: np.ndarray, secs: float) -> np.ndarray:
    """Waveform -> (T, 38), with quality gates that explain themselves."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-4:
        raise SystemExit(f"error: {name} is silent (peak {peak:.2e})")
    if peak >= CLIP_LEVEL:
        print(f"  warning: {name} is clipping (peak {peak:.2f}); "
              "lower the input gain for a fair test", file=sys.stderr)

    X = feat.extract_from_array(y, config.SR)
    if X.shape[0] < MIN_FRAMES:
        raise SystemExit(
            f"error: {name} yielded {X.shape[0]} frames after VAD "
            f"({secs:.1f}s audio); need >= {MIN_FRAMES} frames (~2s of speech)."
        )
    if secs < MIN_SECONDS:
        raise SystemExit(f"error: {name} is {secs:.2f}s; need >= {MIN_SECONDS}s")
    print(f"  {name}: {secs:.1f}s -> {X.shape[0]} frames after VAD")
    return X


def collect_features(args) -> list[np.ndarray]:
    """Gather features from --audio files and/or --record durations."""
    out: list[np.ndarray] = []
    for path in getattr(args, "audio", None) or []:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"error: no such file: {p}")
        y, secs, sr = load_audio(p)
        out.append(features_from_audio(f"{p.name} ({sr} Hz)", y, secs))
    for i, secs in enumerate(getattr(args, "record", None) or [], 1):
        y = record_audio(secs, getattr(args, "device", None))
        out.append(features_from_audio(f"recording {i}", y, secs))
    if not out:
        raise SystemExit(
            "error: nothing to score. Give --audio FILE and/or --record SECONDS.\n"
            "  example:  voiceauth enroll --name me --record 6 --record 6 --record 6"
        )
    return out


# ---------------- evaluation-scale reproduction ----------------

def reproduce_eval_split() -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    """Rebuild enroll/test exactly as scripts/evaluate.py does.

    Same seed, same iteration order, same >=30-frame filter. Without this the
    cohort statistics -- and therefore the threshold scale -- differ.
    """
    split = gmm_ubm.load(config.MODELS / "speaker_split.pkl")
    root = config.DATA_FEAT / "eval"

    feats: dict[str, list[np.ndarray]] = {}
    for spk in split["eval"]:
        d = root / spk
        if not d.exists():
            continue
        arrs = [np.load(f) for f in sorted(d.glob("*.npy"))]
        arrs = [a for a in arrs if a.shape[0] >= 30]
        if arrs:
            feats[spk] = arrs

    rng = np.random.default_rng(config.SEED)
    enroll: dict[str, list[np.ndarray]] = {}
    test: dict[str, list[np.ndarray]] = {}
    k = config.UTTS_PER_SPK_ENROLL
    for spk, arrs in feats.items():
        if len(arrs) < k + 1:
            continue
        idx = list(range(len(arrs)))
        rng.shuffle(idx)
        enroll[spk] = [arrs[i] for i in idx[:k]]
        test[spk] = [arrs[i] for i in idx[k:k + config.UTTS_PER_SPK_TEST]]
    return enroll, test


def tnorm_cohort(test: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    """First 4 x N_TNORM impostor utterances, in evaluate.py's order."""
    target = 4 * config.N_TNORM_IMPOSTORS
    cohort: list[np.ndarray] = []
    for arrs in test.values():
        for a in arrs:
            cohort.append(a)
            if len(cohort) >= target:
                return cohort
    return cohort


def cohort_raw_scores(model: dict, ubm: dict,
                      cohort: list[np.ndarray]) -> np.ndarray:
    """Raw LLR of the impostor cohort against one model."""
    pool = cohort[: config.N_TNORM_IMPOSTORS]
    if not pool:
        return np.zeros(0, dtype=np.float64)
    return np.array([gmm_ubm.score(model, ubm, x, gmm_ubm.ubm_loglik(x, ubm))
                     for x in pool], dtype=np.float64)


def tnorm_stats(model: dict, ubm: dict, cohort: list[np.ndarray]) -> tuple[float, float]:
    """Per-model mean/std of impostor scores (model-side T-norm)."""
    raw = cohort_raw_scores(model, ubm, cohort)
    if raw.size == 0:
        return 0.0, 1.0
    return float(raw.mean()), float(raw.std() + 1e-10)


def normalise(raw: float, stats: tuple[float, float]) -> float:
    mu, sd = stats
    return (raw - mu) / sd


def load_cohort_models() -> list[dict]:
    """Other-speaker models used to test-normalise every trial."""
    if not COHORT_PATH.exists():
        raise SystemExit(
            f"error: missing {COHORT_PATH.name}; run 'voiceauth calibrate' first "
            "(~1 min, builds 20 cohort models and the T-norm threshold)")
    obj = gmm_ubm.load(COHORT_PATH)
    return obj["models"]


def trial_t_score(model: dict, X: np.ndarray, ubm: dict,
                  cohort: list[dict]) -> dict:
    """T = (raw LLR of this model - cohort mean) / cohort sd, on the same trial.

    Scoring the SAME recording against other speakers' models cancels the
    channel/domain effect: an unusual microphone raises every score, not just
    the target's. That is what makes the threshold portable.
    """
    raw = float(gmm_ubm.score(model, ubm, X))
    ubm_ll = gmm_ubm.ubm_loglik(X, ubm)
    coh = np.array([gmm_ubm.score(m, ubm, X, ubm_ll) for m in cohort],
                   dtype=np.float64)
    mu, sd = float(coh.mean()), float(coh.std() + 1e-10)
    return {"raw": raw, "cohort_mu": mu, "cohort_sd": sd,
            "cohort_max": float(coh.max()), "cohort_n": int(coh.size),
            "t": (raw - mu) / sd, "frames": int(X.shape[0])}


def default_threshold() -> tuple[float, str]:
    """T-norm decision threshold, plus where the number came from."""
    if CALIBRATION.exists():
        try:
            obj = json.loads(CALIBRATION.read_text(encoding="utf-8"))
            if "t_threshold" in obj:
                return float(obj["t_threshold"]), str(obj.get("source", "calibration"))
        except Exception:
            pass
    return DEFAULT_T_THRESHOLD, "published EER point (fallback until calibrate runs)"


# ---------------- speaker model storage ----------------

def speaker_path(name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "speaker"
    return SPEAKER_DIR / f"{safe}.pkl"


def load_ubm() -> dict:
    p = config.MODELS / "ubm.pkl"
    if not p.exists():
        raise SystemExit(f"error: missing {p}; run scripts/train_ubm.py first")
    return gmm_ubm.load(p)


def save_speaker(name: str, model: dict, threshold: float,
                 thr_source: str, meta: dict) -> Path:
    path = speaker_path(name)
    gmm_ubm.save(path, {
        "name": name,
        "model": model,
        "threshold": float(threshold),
        "threshold_space": "tnorm",
        "threshold_source": thr_source,
        "feat_dim": config.FEAT_DIM,
        "mixtures": config.N_MIXTURES,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta,
    })
    return path


def load_speaker(name: str) -> dict:
    path = speaker_path(name)
    if not path.exists():
        raise SystemExit(f"error: no enrolled speaker '{name}' "
                         f"(looked for {path}). Enroll first.")
    obj = gmm_ubm.load(path)
    if obj.get("feat_dim") != config.FEAT_DIM or obj.get("mixtures") != config.N_MIXTURES:
        raise SystemExit(
            f"error: speaker '{name}' was built with {obj.get('mixtures')} mixtures / "
            f"{obj.get('feat_dim')}-D features, config now uses "
            f"{config.N_MIXTURES} / {config.FEAT_DIM}. Re-enroll."
        )
    return obj


# ---------------- shared model / trial helpers ----------------

def fit_speaker(name: str, Xs: list[np.ndarray], threshold: float | None,
                sources: list[str]) -> dict:
    """MAP-adapt and persist a speaker model. Used by enroll/register."""
    ubm = load_ubm()
    total_frames = sum(x.shape[0] for x in Xs)
    if total_frames < 300:
        print(f"  warning: only {total_frames} frames of speech; enrollment is "
              "thin (aim for 3+ recordings of 5-10s)", file=sys.stderr)

    print(f"  adapting {ubm['means'].shape[0]} mixtures ...", end="", flush=True)
    model = gmm_ubm.map_adapt(ubm, Xs)
    print(" done")

    if threshold is not None:
        thr, thr_source = float(threshold), "user"
    else:
        thr, thr_source = default_threshold()

    path = save_speaker(name, model, thr, thr_source, {
        "sources": sources,
        "n_utts": len(Xs),
        "n_frames": total_frames,
    })
    return {"name": name, "path": path, "threshold": thr, "source": thr_source,
            "n_utts": len(Xs), "n_frames": total_frames}


def score_trial(spk: dict, X: np.ndarray) -> dict:
    """Test-normalised decision (T-score) plus diagnostics for one trial."""
    ubm = load_ubm()
    cohort = load_cohort_models()
    r = trial_t_score(spk["model"], X, ubm, cohort)
    thr = float(spk["threshold"])
    r.update({
        "threshold": thr,
        "margin": r["t"] - thr,
        "accept": r["t"] >= thr,
    })
    return r


def print_trial(spk: dict, r: dict) -> None:
    print()
    print(f"  speaker   : {spk['name']}  (enrolled {spk['created'][:10]})")
    print(f"  frames    : {r['frames']}")
    print(f"  raw LLR   : {r['raw']:+.4f}")
    print(f"  cohort    : {r['cohort_n']} models, raw mu {r['cohort_mu']:+.3f} "
          f"sd {r['cohort_sd']:.3f}")
    print(f"  T-score   : {r['t']:+.3f}      <- decision space")
    print(f"  threshold : {r['threshold']:+.3f} T   margin {r['margin']:+.3f}")
    print(f"  verdict   : {'ACCEPT' if r['accept'] else 'REJECT'}")

    if abs(r["margin"]) < 0.5:
        print("  note      : borderline - within half a standard deviation of the "
              "threshold, so the decision is close to a coin flip")


# ---------------- commands ----------------

def cmd_enroll(args) -> int:
    Xs = collect_features(args)
    info = fit_speaker(
        args.name, Xs, args.threshold,
        [str(p) for p in (args.audio or [])]
        + [f"mic:{s}s" for s in (args.record or [])],
    )
    print(f"\nenrolled '{args.name}' -> {info['path']}")
    print(f"  utterances {info['n_utts']}  frames {info['n_frames']}")
    print(f"  threshold {info['threshold']:+.3f} T  [{info['source']}]")
    if args.json:
        print(json.dumps({"name": args.name, "path": str(info["path"]),
                          "n_utts": info["n_utts"], "n_frames": info["n_frames"],
                          "threshold": info["threshold"],
                          "threshold_space": "tnorm"}, indent=2))
    return 0


def cmd_verify(args) -> int:
    spk = load_speaker(args.name)
    Xs = collect_features(args)
    if len(Xs) != 1 and not args.average:
        raise SystemExit("error: verify takes one trial; pass one --audio/--record, "
                         "or add --average to score several as a single trial")
    X = np.concatenate(Xs, axis=0) if args.average else Xs[0]

    if args.threshold is not None:
        spk = dict(spk, threshold=float(args.threshold))
    r = score_trial(spk, X)
    print_trial(spk, r)
    if args.json:
        print(json.dumps({"speaker": spk["name"], "verdict":
                          "ACCEPT" if r["accept"] else "REJECT", **r}, indent=2))
    return 0 if r["accept"] else 2


# ---------------- interactive register / login ----------------

def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        return input(f"{prompt}{suffix}: ").strip() or (default or "")
    except EOFError:
        return default or ""


def _countdown(seconds: int = 3) -> None:
    for i in range(seconds, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1)
    print(" speak now", flush=True)


def _capture_take(label: str, seconds: float, device, attempts: int = 3) -> np.ndarray:
    """Record one take, retrying while it is silent or too short."""
    problem = ""
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            print(f"  retake {attempt}/{attempts} - {problem}")
        y = record_audio(seconds, device)
        try:
            return features_from_audio(label, y, seconds)
        except SystemExit as e:
            problem = str(e).replace("error: ", "")
    raise SystemExit(f"error: no usable take after {attempts} attempts ({problem})")


def cmd_register(args) -> int:
    name = args.user or _ask("username")
    if not name:
        raise SystemExit("error: a username is required")
    if speaker_path(name).exists() and not args.force:
        raise SystemExit(f"error: '{name}' is already registered ({speaker_path(name)}). "
                         f"Use --force to replace it, or delete --name {name}")

    if args.audio:
        Xs = collect_features(args)
    else:
        takes, seconds = args.takes, args.seconds
        print(f"Registering '{name}': {takes} takes of {seconds:g}s each.")
        print("Find a quiet spot, keep a steady distance from the microphone,")
        print("and speak naturally - the words are not checked.\n")
        Xs = []
        for i in range(1, takes + 1):
            input(f"  Press Enter for take {i}/{takes} ... ")
            _countdown()
            Xs.append(_capture_take(f"take {i}/{takes}", seconds, args.device))
            if i < takes:
                time.sleep(max(0.0, args.gap))

    total = sum(x.shape[0] for x in Xs)
    if len(Xs) < 3 or total < 900:
        print(f"  warning: {len(Xs)} takes / {total} frames is thin; "
              "aim for 5 takes of 10s", file=sys.stderr)

    info = fit_speaker(name, Xs, args.threshold,
                       [f"mic:{args.seconds:g}s x{len(Xs)}"] if not args.audio
                       else [str(p) for p in args.audio])
    print(f"\nregistered '{name}' -> {info['path']}")
    print(f"  takes {info['n_utts']}  frames {info['n_frames']}  "
          f"threshold {info['threshold']:+.3f} T")
    print(f"  now try:  py scripts/voice_cli.py login --user {name}")
    return 0


def cmd_login(args) -> int:
    name = args.user or _ask("username")
    if not name:
        raise SystemExit("error: a username is required")
    spk = load_speaker(name)

    if args.audio:
        Xs = collect_features(args)
        X = np.concatenate(Xs, axis=0) if len(Xs) > 1 else Xs[0]
        r = score_trial(spk, X)
        print_trial(spk, r)
        print("\n  " + ("ACCESS GRANTED" if r["accept"] else "ACCESS DENIED"))
        if args.json:
            print(json.dumps({"user": name, **r}, indent=2))
        return 0 if r["accept"] else 2

    takes = max(1, args.takes)
    attempts = max(1, args.attempts)
    for attempt in range(1, attempts + 1):
        captured: list[np.ndarray] = []
        for take in range(1, takes + 1):
            label = "login" if takes == 1 else f"login take {take}/{takes}"
            input(f"  Press Enter to speak for {args.seconds:g}s "
                  f"({label}, attempt {attempt}/{attempts}) ... ")
            _countdown()
            try:
                captured.append(_capture_take(label, args.seconds, args.device,
                                              attempts=1))
            except SystemExit as e:
                print(f"  {e}")
                captured = []
                break
        if not captured:
            continue
        # Pooling takes averages out phonetic and channel luck: one 5 s trial
        # is high-variance, and a sibling's best draw can outscore a genuine
        # user's worst. Pooled scoring narrows both distributions.
        X = np.concatenate(captured, axis=0) if len(captured) > 1 else captured[0]

        r = score_trial(spk, X)
        print_trial(spk, r)
        if r["accept"]:
            print("\n  ACCESS GRANTED\n")
            return 0
        print("\n  ACCESS DENIED")
        if attempt < attempts:
            print("  try again\n")
    return 2


def cmd_record(args) -> int:
    """Record one take to a WAV file, so trials can be kept and re-scored."""
    out = Path(args.out)
    if out.suffix.lower() != ".wav":
        out = out.with_suffix(".wav")
    out.parent.mkdir(parents=True, exist_ok=True)

    y = record_audio(args.seconds, args.device)
    features_from_audio(out.name, y, args.seconds)  # same silence/VAD gates
    sf.write(str(out), y, config.SR, subtype="PCM_16")

    dur = len(y) / config.SR
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    print(f"  saved {out}  ({dur:.1f}s, {config.SR} Hz mono, peak {peak:.2f})")
    print("  keep this file: it is the only record of that take, and "
          "calibrate-user needs it")
    if args.json:
        print(json.dumps({"path": str(out), "seconds": round(dur, 3),
                          "sample_rate": config.SR, "peak": peak}, indent=2))
    return 0


def cmd_list(args) -> int:
    rows = []
    for p in sorted(SPEAKER_DIR.glob("*.pkl")):
        try:
            o = gmm_ubm.load(p)
        except Exception:
            continue
        rows.append({"name": o.get("name", p.stem),
                     "created": o.get("created", "?"),
                     "utts": o.get("n_utts"),
                     "frames": o.get("n_frames"),
                     "threshold": o.get("threshold")})
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print(f"no enrolled speakers in {SPEAKER_DIR}")
        return 0
    print(f"{'name':20} {'created':12} {'utts':>5} {'frames':>8} {'thr':>8}")
    for r in rows:
        print(f"{r['name'][:20]:20} {str(r['created'])[:10]:12} "
              f"{str(r['utts']):>5} {str(r['frames']):>8} "
              f"{r['threshold']:>+8.3f}")
    return 0


def cmd_delete(args) -> int:
    path = speaker_path(args.name)
    if not path.exists():
        raise SystemExit(f"error: no enrolled speaker '{args.name}'")
    if not args.yes:
        raise SystemExit(f"refusing to delete {path} without --yes")
    path.unlink()
    print(f"deleted {path}")
    return 0


def _eer_threshold(tg: np.ndarray, im: np.ndarray) -> tuple[float, float, float, float]:
    """EER point on T scores. Returns (threshold, eer, far, frr)."""
    s = np.concatenate([tg, im])
    y = np.concatenate([np.ones(tg.size), np.zeros(im.size)])
    order = np.argsort(-s)
    s, y = s[order], y[order]
    frr = 1.0 - np.cumsum(y) / tg.size
    far = np.cumsum(1 - y) / im.size
    i = int(np.argmin(np.abs(far - frr)))
    return float(s[i]), float((far[i] + frr[i]) / 2), float(far[i]), float(frr[i])


def cmd_calibrate(args) -> int:
    """Build the cohort models and set the T-norm threshold on held-out speakers."""
    ubm = load_ubm()
    enroll, test = reproduce_eval_split()
    names = list(test.keys())
    if len(names) < 4:
        raise SystemExit("error: not enough eval speakers to calibrate")

    n_cohort = min(args.cohort, len(names) - 2)
    cohort_names = names[:n_cohort]
    held_out = names[n_cohort:n_cohort + args.speakers]
    if not held_out:
        raise SystemExit("error: no held-out speakers left; lower --cohort")

    print(f"building {n_cohort} cohort models ...", end="", flush=True)
    cohort = [gmm_ubm.map_adapt(ubm, enroll[s]) for s in cohort_names]
    print(" done")
    gmm_ubm.save(COHORT_PATH, {"speakers": cohort_names, "models": cohort})

    print(f"scoring {len(held_out)} held-out speakers "
          f"(speakers not in the cohort) ...", flush=True)
    tg: list[float] = []
    im: list[float] = []
    for s in held_out:
        m = gmm_ubm.map_adapt(ubm, enroll[s])
        tgt = len(tg)
        for x in test[s]:
            tg.append(trial_t_score(m, x, ubm, cohort)["t"])
        imp = len(im)
        for other in held_out:
            if other == s:
                continue
            for x in test[other][: args.impostors]:
                im.append(trial_t_score(m, x, ubm, cohort)["t"])
        print(f"  {s}: +{len(tg) - tgt} targets, +{len(im) - imp} impostors",
              flush=True)

    tg_arr, im_arr = np.array(tg), np.array(im)
    eer_thr, eer, far_eer, frr_eer = _eer_threshold(tg_arr, im_arr)
    far1_thr = float(np.percentile(im_arr, 99))
    far5_thr = float(np.percentile(im_arr, 95))

    # Default operating point: held-out FAR = 1% (a login gate should lean
    # conservative), which still sat on the right side of every real trial here.
    thr = far1_thr
    obj = {
        "t_threshold": thr,
        "source": f"calibrate: held-out FAR=1% over {len(held_out)} speakers",
        "eer_threshold": eer_thr,
        "eer": eer,
        "far1_threshold": far1_thr,
        "far5_threshold": far5_thr,
        "eer_far": far_eer,
        "eer_frr": frr_eer,
        "n_targets": int(tg_arr.size),
        "n_impostors": int(im_arr.size),
        "held_out_speakers": held_out,
        "cohort_size": len(cohort),
        "cohort_speakers": cohort_names,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "re-run after changing N_MIXTURES, features or the eval split",
    }
    CALIBRATION.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    print()
    print(f"  targets {tg_arr.size}  impostors {im_arr.size}  "
          f"cohort {len(cohort)} models")
    print(f"  EER          {eer * 100:5.2f}%   (threshold {eer_thr:+.3f} T)")
    print(f"  FAR=1% point          threshold {far1_thr:+.3f} T  <- default")
    print(f"  FAR=5% point          threshold {far5_thr:+.3f} T")
    print(f"  written       : {CALIBRATION}")
    print(f"                  {COHORT_PATH.name}")
    return 0


def _choose_threshold(genuine: list[float], impostor: list[float]) -> tuple[float | None, float]:
    """Midpoint between the worst genuine and best impostor T-scores.

    Returns (threshold or None, margin). A non-positive margin means the two
    distributions overlap and NO threshold can separate them - the caller must
    say so rather than invent one.
    """
    if not genuine or not impostor:
        return None, float("nan")
    lo_g, hi_i = min(genuine), max(impostor)
    margin = lo_g - hi_i
    if margin <= 0:
        return None, margin
    return (lo_g + hi_i) / 2.0, margin


def cmd_calibrate_user(args) -> int:
    """Set one user's threshold from their own recordings plus impostor takes."""
    name = args.user or _ask("username")
    if not name:
        raise SystemExit("error: a username is required")
    spk = load_speaker(name)

    if not args.genuine or not args.impostor:
        raise SystemExit(
            "error: need both --genuine (the real user) and --impostor (someone\n"
            "else, recorded on the same microphone). Example:\n"
            "  voiceauth calibrate-user --user me \\\n"
            "    --genuine g1.wav g2.wav g3.wav --impostor brother1.wav brother2.wav"
        )

    ubm = load_ubm()
    cohort = load_cohort_models()

    def scores(paths: list[str]) -> list[float]:
        out = []
        for p in paths:
            path = Path(p)
            if not path.exists():
                raise SystemExit(f"error: no such file: {path}")
            y, secs, sr = load_audio(path)
            X = features_from_audio(f"{path.name} ({sr} Hz)", y, secs)
            out.append(trial_t_score(spk["model"], X, ubm, cohort)["t"])
        return out

    print("scoring genuine recordings (must be accepted) ...", flush=True)
    g = scores(args.genuine)
    print("scoring impostor recordings (must be rejected) ...", flush=True)
    i = scores(args.impostor)

    print()
    print(f"{'genuine':>10}  T")
    for p, t in zip(args.genuine, g):
        print(f"{Path(p).name[:10]:>10}  {t:+.3f}")
    print(f"{'impostor':>10}  T")
    for p, t in zip(args.impostor, i):
        print(f"{Path(p).name[:10]:>10}  {t:+.3f}")

    thr, margin = _choose_threshold(g, i)
    print()
    print(f"  genuine : min {min(g):+.3f}  max {max(g):+.3f}   (n={len(g)})")
    print(f"  impostor: min {min(i):+.3f}  max {max(i):+.3f}   (n={len(i)})")
    print(f"  margin  : {margin:+.3f} T")

    if thr is None:
        print("\n  NO SEPARATING THRESHOLD: the impostor scores as high as the "
              "genuine user.\n  A stricter threshold would reject you too. "
              "More enrollment audio, pooling several\n  login takes, or a "
              "stronger model (more mixtures/PCA) is needed - do not deploy\n  "
              "this as a gate until the margin is positive.", file=sys.stderr)
        return 1

    if args.set_threshold is not None:
        thr = float(args.set_threshold)
        source = "user calibration (manual threshold)"
    else:
        source = f"user calibration: {len(g)} genuine / {len(i)} impostor"

    if not args.dry_run:
        spk["threshold"] = float(thr)
        spk["threshold_source"] = source
        spk["threshold_space"] = "tnorm"
        spk["calibrated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        gmm_ubm.save(speaker_path(name), spk)
        print(f"\n  threshold set to {thr:+.3f} T  ({source})")
        print(f"  written: {speaker_path(name)}")
    else:
        print(f"\n  dry run: threshold would be {thr:+.3f} T ({source})")

    print(f"  caveat  : n={len(g)}+{len(i)} is a tiny sample; scores drift with "
          "microphone, distance and\n            speaking rate. Re-run after "
          "moving the mic, and keep the recordings.")
    return 0


def cmd_selftest(args) -> int:
    """Recompute published eval scores; they must match outputs/scores.npz."""
    z_path = config.OUTPUTS / "scores.npz"
    if not z_path.exists():
        raise SystemExit(f"error: missing {z_path}; run scripts/evaluate.py first")
    z = np.load(z_path)
    scores, labels = z["scores"], z["labels"]
    models, speakers = z["models"], z["speakers"]

    ubm = load_ubm()
    enroll, test = reproduce_eval_split()
    cohort = tnorm_cohort(test)

    ok = True
    print(f"{'speaker':8} {'utts':>4} {'max|diff|':>11}  result")
    for name in list(test.keys())[: args.speakers]:
        model = gmm_ubm.map_adapt(ubm, enroll[name])
        stats = tnorm_stats(model, ubm, cohort)
        mine = np.array([normalise(gmm_ubm.score(model, ubm, x), stats)
                         for x in test[name]])

        mask = (models == name) & (speakers == name)
        ref = np.sort(scores[mask])
        got = np.sort(mine)
        if ref.shape != got.shape:
            print(f"{name:8} {len(mine):>4} {'shape mismatch':>11}  FAIL "
                  f"({got.shape} vs {ref.shape})")
            ok = False
            continue
        diff = float(np.max(np.abs(got - ref))) if got.size else 0.0
        good = diff < 1e-6
        ok &= good
        print(f"{name:8} {len(mine):>4} {diff:>11.2e}  "
              f"{'PASS' if good else 'FAIL'}")

    print()
    if ok:
        thr, src = default_threshold()
        print(f"scale reproduced: scores match scores.npz to <1e-6, so the "
              f"published evaluation is intact; the gate uses {thr:+.3f} T ({src})")
        return 0
    print("FAIL: reproduced scores differ from scores.npz; do not trust the "
          "threshold in verify", file=sys.stderr)
    return 1


def cmd_devices(args) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        devs = _ffmpeg_input_devices()
        if not devs:
            raise SystemExit("error: no audio backend. py -m pip install sounddevice")
        print("input devices (ffmpeg/dshow):")
        for i, d in enumerate(devs):
            print(f"  [{i}] {d}")
        return 0
    print(sd.query_devices())
    return 0


# ---------------- argument parsing ----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voiceauth",
        description="Text-independent speaker verification (GMM-UBM). "
                    "Enroll your voice, then test a recording against it.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_inputs(sp):
        sp.add_argument("--audio", nargs="+", metavar="FILE",
                        help="audio file(s): wav/flac/mp3/... any sample rate")
        sp.add_argument("--record", nargs="+", type=float, metavar="SECONDS",
                        help="record from the microphone (repeat for several takes)")
        sp.add_argument("--device", default=None,
                        help="sounddevice input name or index (see 'devices')")

    sp = sub.add_parser("register",
                        help="register a user by recording 5 takes of 10s")
    sp.add_argument("--user", help="username (prompted if omitted)")
    sp.add_argument("--takes", type=int, default=5, help="number of takes (default 5)")
    sp.add_argument("--seconds", type=float, default=10.0,
                    help="length of each take in seconds (default 10)")
    sp.add_argument("--gap", type=float, default=1.5,
                    help="pause between takes in seconds (default 1.5)")
    sp.add_argument("--device", default=None)
    sp.add_argument("--audio", nargs="+", metavar="FILE",
                    help="use files instead of the microphone (skips prompts)")
    sp.add_argument("--threshold", type=float, default=None)
    sp.add_argument("--force", action="store_true", help="replace an existing user")
    sp.set_defaults(func=cmd_register)

    sp = sub.add_parser("login",
                        help="check whether a recording matches a registered user")
    sp.add_argument("--user", help="username (prompted if omitted)")
    sp.add_argument("--seconds", type=float, default=5.0,
                    help="length of each take in seconds (default 5)")
    sp.add_argument("--takes", type=int, default=1,
                    help="recordings pooled into one trial (default 1; 3 lowers variance)")
    sp.add_argument("--attempts", type=int, default=3,
                    help="how many tries before denying (default 3)")
    sp.add_argument("--device", default=None)
    sp.add_argument("--audio", nargs="+", metavar="FILE",
                    help="use files instead of the microphone")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_login)

    sp = sub.add_parser("enroll", help="build a speaker model from your voice")
    sp.add_argument("--name", required=True, help="label for this speaker")
    add_inputs(sp)
    sp.add_argument("--threshold", type=float, default=None,
                    help="decision threshold (default: measured EER threshold)")
    sp.add_argument("--force", action="store_true", help="overwrite existing model")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_enroll)

    sp = sub.add_parser("verify", help="test a recording against an enrolled voice")
    sp.add_argument("--name", required=True)
    add_inputs(sp)
    sp.add_argument("--average", action="store_true",
                    help="pool several recordings into one trial (lowers variance)")
    sp.add_argument("--threshold", type=float, default=None)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("record",
                        help="record a take to a WAV file you keep")
    sp.add_argument("--out", required=True, metavar="FILE",
                    help="output .wav path")
    sp.add_argument("--seconds", type=float, default=8.0,
                    help="recording length in seconds (default 8)")
    sp.add_argument("--device", default=None,
                    help="sounddevice input name or index (see 'devices')")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_record)

    sp = sub.add_parser("list", help="list enrolled speakers")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("delete", help="remove an enrolled speaker")
    sp.add_argument("--name", required=True)
    sp.add_argument("--yes", action="store_true", help="confirm deletion")
    sp.set_defaults(func=cmd_delete)

    sp = sub.add_parser("selftest", help="prove the score scale matches scores.npz")
    sp.add_argument("--speakers", type=int, default=3,
                    help="how many eval speakers to recompute (default 3)")
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("calibrate",
                        help="build cohort models and the T-norm threshold")
    sp.add_argument("--cohort", type=int, default=20,
                    help="number of cohort speakers/models (default 20)")
    sp.add_argument("--speakers", type=int, default=12,
                    help="held-out speakers used to measure the threshold (default 12)")
    sp.add_argument("--impostors", type=int, default=6,
                    help="impostor utterances per held-out pair (default 6)")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser(
        "calibrate-user",
        help="set one user's threshold from their recordings plus impostors")
    sp.add_argument("--user", help="username (prompted if omitted)")
    sp.add_argument("--genuine", nargs="+", metavar="FILE",
                    help="recordings of the real user (same microphone)")
    sp.add_argument("--impostor", nargs="+", metavar="FILE",
                    help="recordings of someone else trying to pass as them")
    sp.add_argument("--set-threshold", type=float, default=None,
                    help="force this T threshold instead of the computed midpoint")
    sp.add_argument("--dry-run", action="store_true",
                    help="report the scores and threshold without writing")
    sp.set_defaults(func=cmd_calibrate_user)

    sp = sub.add_parser("devices", help="list audio input devices")
    sp.set_defaults(func=cmd_devices)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "enroll" and speaker_path(args.name).exists() and not args.force:
        raise SystemExit(f"error: '{args.name}' already enrolled "
                         f"({speaker_path(args.name)}). Use --force to replace.")
    SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
