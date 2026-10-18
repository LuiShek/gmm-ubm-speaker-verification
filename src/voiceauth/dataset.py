"""
Dataset indexing for LibriSpeech and VoxCeleb.

LibriSpeech train-clean-100 layout:
  train-clean-100/<spk>/<chapter>/<spk>-<chapter>-<utt>.flac
  Each utterance is a SEPARATE speaker recording with transcript in
  <spk>-<chapter>.trans.txt; we treat each .flac as one utterance, all
  sharing the same speaker id (first hyphen-delimited token).
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict, OrderedDict
import random

import numpy as np
import soundfile as sf

from . import config

# Set by index_all() when loading from parquet: maps speaker -> parquet row keys.
PARQUET_ROOT = config.DATA_RAW / "librispeech_parquet"


def _speaker_from_row(row) -> str:
    """Pull the speaker id out of a parquet row, tolerating schema variants."""
    for key in ("speaker_id", "speaker", "speakerId"):
        v = row.get(key) if isinstance(row, dict) else None
        if v is not None:
            return str(v)
    # Fallback: parse from path/file name like 103-1240-0000.flac
    for key in ("file", "path", "audio_path"):
        v = row.get(key) if isinstance(row, dict) else None
        if v is None:
            continue
        if isinstance(v, dict):
            v = v.get("path") or v.get("filename") or ""
        name = str(v).split("/")[-1]
        if "-" in name:
            return name.split("-")[0]
    return "unknown"


def index_parquet(root: Path = PARQUET_ROOT) -> dict[str, list[tuple[Path, int]]]:
    """speaker_id -> [(shard_path, row_index), ...]. Reads only metadata + speaker column."""
    import pyarrow.parquet as pq
    shards = sorted(Path(root).glob("*.parquet"))
    if not shards:
        return {}
    by_spk: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    skipped = []
    for shard in shards:
        # An in-flight download has no parquet footer yet; skip it rather
        # than crashing on the magic-bytes check.
        try:
            pf = pq.ParquetFile(str(shard))
        except Exception:
            skipped.append(shard.name)
            continue

        names = pf.schema_arrow.names
        if "speaker_id" in names:
            spk_col = pq.read_table(str(shard), columns=["speaker_id"]).column("speaker_id").to_pylist()
        else:
            rows = pq.read_table(str(shard)).to_pylist()
            spk_col = [_speaker_from_row(r) for r in rows]

        for i, spk in enumerate(spk_col):
            by_spk[str(spk)].append((shard, i))

    if skipped:
        print(f"[index_parquet] skipped {len(skipped)} incomplete shard(s): {', '.join(skipped[:3])}", flush=True)
    return dict(by_spk)


def index_all() -> dict[str, list[tuple[Path, int]]]:
    """Unified index: prefers parquet if downloaded, else flac tree."""
    pq_idx = index_parquet()
    if pq_idx:
        return pq_idx
    flac = index_librispeech()
    return {s: [(p, -1) for p in files] for s, files in flac.items()}


def index_librispeech(root: Path = config.LIBRISPEECH_DIR) -> dict[str, list[Path]]:
    """speaker_id -> list of .flac paths, sorted."""
    if not root.exists():
        raise FileNotFoundError(f"LibriSpeech not extracted at {root}")
    by_spk: dict[str, list[Path]] = defaultdict(list)
    for flac in root.rglob("*.flac"):
        spk = flac.stem.split("-")[0]
        by_spk[spk].append(flac)
    for spk in by_spk:
        by_spk[spk].sort()
    return dict(by_spk)


# Decoded row groups, LRU. One row group is ~100 rows (~24 MB decoded), so a
# cache of 4 stays well under the 400 MB RAM floor while removing the ~40x
# cost of re-reading a whole parquet column per row.
_RG_CACHE: "OrderedDict[tuple[str, int], object]" = OrderedDict()
_RG_CACHE_MAX = 4


def _row_group_for(pf, row_idx: int) -> tuple[int, int]:
    """Map a global row index to (row_group_index, row_within_group)."""
    seen = 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if row_idx < seen + n:
            return g, row_idx - seen
        seen += n
    raise IndexError(f"row {row_idx} beyond {seen} rows")


def _read_row_group(shard: Path, rg: int):
    """Return a cached (or freshly read) audio-only table for one row group."""
    key = (str(shard), rg)
    hit = _RG_CACHE.get(key)
    if hit is not None:
        _RG_CACHE.move_to_end(key)
        return hit
    import pyarrow.parquet as pq
    tbl = pq.ParquetFile(str(shard)).read_row_group(rg, columns=["audio"])
    _RG_CACHE[key] = tbl
    if len(_RG_CACHE) > _RG_CACHE_MAX:
        _RG_CACHE.popitem(last=False)
    return tbl


def load_parquet_audio(shard: Path, row_idx: int) -> tuple[np.ndarray, int]:
    """Read one row's audio from a parquet shard.

    Returns (float32 waveform, sample_rate). HF audio columns come back as
    {'array': np.ndarray, 'sampling_rate': int} or {'bytes': ..., 'path': ...}.
    """
    import io
    import numpy as np
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(shard))
    rg, local = _row_group_for(pf, row_idx)
    tbl = _read_row_group(shard, rg)
    cell = tbl.column("audio")[local].as_py()
    if isinstance(cell, dict):
        if cell.get("array") is not None:
            return np.asarray(cell["array"], dtype=np.float32), int(cell.get("sampling_rate", config.SR))
        if cell.get("bytes") is not None:
            arr, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            return arr, int(sr)
    raise ValueError(f"Unrecognised audio cell type at {shard.name}:{row_idx}: {type(cell)}")


def split_speakers(by_spk: dict[str, list],
                   n_ubm: int = config.N_SPEAKERS_UBM,
                   n_eval: int = config.N_SPEAKERS_EVAL,
                   seed: int = config.SEED):
    """Return (ubm_speakers, eval_speakers), disjoint.

    Filters to speakers with >= (enroll + test) usable utterances.
    """
    need = config.UTTS_PER_SPK_ENROLL + config.UTTS_PER_SPK_TEST
    usable = {s: fs for s, fs in by_spk.items() if len(fs) >= need}
    speakers = sorted(usable.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)
    if len(speakers) < n_ubm + n_eval:
        raise ValueError(
            f"Need {n_ubm + n_eval} usable speakers, have {len(speakers)}"
        )
    ubm = speakers[:n_ubm]
    ev = speakers[n_ubm:n_ubm + n_eval]
    return {s: usable[s] for s in ubm}, {s: usable[s] for s in ev}


def pick_utterances(files: list[Path], k: int, rng: random.Random) -> list[Path]:
    """Deterministic selection of k non-overlapping utterances."""
    if len(files) <= k:
        return list(files)
    return rng.sample(files, k)


def describe(by_spk: dict[str, list[Path]], label: str) -> None:
    n_utt = sum(len(v) for v in by_spk.values())
    print(f"[{label}] speakers={len(by_spk)}  utterances={n_utt}", flush=True)
    if by_spk:
        lens = [len(v) for v in by_spk.values()]
        print(f"  per-speaker utts: min={min(lens)} max={max(lens)} "
              f"mean={sum(lens)/len(lens):.1f}", flush=True)
