"""Step 2 of 4: extract features for the reserved EVAL speakers.

Reads the speaker split saved by train_ubm.py, extracts (T, 38) feature
matrices for every utterance of every eval speaker, caches them to
 data/features/eval/<speaker>/<utt>.npy. Idempotent.
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
    split_path = config.MODELS / "speaker_split.pkl"
    if not split_path.exists():
        print(f"ERROR: {split_path} not found. Run train_ubm.py first.", file=sys.stderr)
        return 1
    split = gmm_ubm.load(split_path)
    eval_spk = split["eval"]

    print(f"[1/2] Extracting features for {len(eval_spk)} eval speakers...", flush=True)
    by_spk = dataset.index_all()
    out_root = config.DATA_FEAT / "eval"
    out_root.mkdir(parents=True, exist_ok=True)

    n_new = 0
    n_cached = 0
    n_fail = 0
    for i, spk in enumerate(eval_spk, 1):
        spk_dir = out_root / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        for j, (shard, row_idx) in enumerate(by_spk.get(spk, [])):
            cache = spk_dir / f"{spk}__{j:03d}.npy"
            if cache.exists():
                n_cached += 1
                continue
            try:
                wav, sr = dataset.load_parquet_audio(shard, row_idx)
                if sr != config.SR:
                    raise ValueError(f"sr={sr}")
                X = feat.extract_from_array(wav)
                if X.shape[0] < 30:
                    continue
                np.save(cache, X)
                n_new += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f"  skip {spk}[{j}]: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(eval_spk)}] new={n_new} cached={n_cached} fail={n_fail}", flush=True)

    print(f"\n[2/2] Done. new={n_new} cached={n_cached} fail={n_fail}", flush=True)
    print(f"DONE in {time.time()-t0:.0f}s -> {out_root}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())
