"""Step 1 of 4: train the Universal Background Model.

- Indexes speakers from downloaded parquet (or LibriSpeech flac if present)
- Reserves N_SPEAKERS_EVAL speakers exclusively for enrollment/eval
- Extracts features for UBM speakers (cached to data/features/ubm/)
- Subsamples UBM_MAX_FRAMES, trains GMM, saves to models/ubm.pkl

Idempotent: cached features are skipped on rerun.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voiceauth import config
from voiceauth import dataset
from voiceauth import features as feat
from voiceauth import gmm_ubm


def main() -> int:
    t0 = time.time()
    feat_dir = config.DATA_FEAT / "ubm"
    feat_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Indexing speakers...", flush=True)
    by_spk = dataset.index_all()
    dataset.describe(by_spk, "all")

    ubm_spk, eval_spk = dataset.split_speakers(by_spk)
    dataset.describe({s: by_spk[s] for s in ubm_spk}, "UBM")
    dataset.describe({s: by_spk[s] for s in eval_spk}, "EVAL (reserved)")

    # Persist the speaker split so enroll/eval scripts agree.
    gmm_ubm.save(config.MODELS / "speaker_split.pkl",
                 {"ubm": ubm_spk, "eval": eval_spk})

    print("\n[2/4] Extracting UBM features...", flush=True)
    rng = np.random.default_rng(config.SEED)
    n_done = 0
    n_skip = 0
    for spk in ubm_spk:
        for i, (shard, row_idx) in enumerate(by_spk[spk]):
            cache = feat_dir / f"{spk}__{i:04d}.npy"
            if cache.exists():
                n_skip += 1
                continue
            try:
                wav, sr = dataset.load_parquet_audio(shard, row_idx)
                if sr != config.SR:
                    raise ValueError(f"sr={sr}")
                X = feat.extract_from_array(wav)
            except Exception as e:
                print(f"  skip {spk}[{i}]: {type(e).__name__}: {e}", flush=True)
                continue
            if X.shape[0] < 30:
                continue
            np.save(cache, X)
            n_done += 1
        if n_done and n_done % 100 < len(by_spk[spk]):
            print(f"  extracted {n_done} new, {n_skip} cached", flush=True)
            np.save(cache, X)
            n_done += 1
        if n_done and n_done % 100 == 0:
            print(f"  extracted {n_done} new, {n_skip} cached", flush=True)

    print(f"  extraction done: {n_done} new, {n_skip} cached", flush=True)

    print("\n[3/4] Loading + subsampling frames...", flush=True)
    all_files = sorted(feat_dir.glob("*.npy"))
    if not all_files:
        print("ERROR: no features extracted", file=sys.stderr)
        return 1

    # Sample FILES first, then take a frame slice from each. Reading the whole
    # corpus to subsample it is pure I/O waste (ponytail: file-level sampling
    # is unbiased enough for UBM training; frame-level would cost ~40 GB of reads).
    rng.shuffle(all_files)
    per_file = max(1, config.UBM_MAX_FRAMES // min(len(all_files), 2000))

    chunks = []
    total = 0
    for f in all_files:
        if total >= config.UBM_MAX_FRAMES:
            break
        try:
            arr = np.load(f)
        except Exception:
            continue
        if arr.shape[0] == 0:
            continue
        if arr.shape[0] > per_file:
            start = int(rng.integers(0, arr.shape[0] - per_file + 1))
            arr = arr[start:start + per_file]
        chunks.append(arr)
        total += arr.shape[0]
        del arr

    X = np.concatenate(chunks, axis=0)
    del chunks
    if X.shape[0] > config.UBM_MAX_FRAMES:
        X = X[:config.UBM_MAX_FRAMES]
    print(f"  sampled {X.shape[0]:,} frames from {len(all_files)} files, dim={X.shape[1]}", flush=True)

    print(f"\n[4/4] Training UBM ({config.N_MIXTURES} mixtures)...", flush=True)
    ubm = gmm_ubm.train_ubm(X, n_mixtures=config.N_MIXTURES)
    out = config.MODELS / "ubm.pkl"
    gmm_ubm.save(out, ubm)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
