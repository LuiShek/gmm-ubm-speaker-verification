"""Download LibriSpeech train-clean-100 as parquet shards from HuggingFace.

Source: openslr/librispeech_asr (mirror of openslr.org/resources/12).
Faster than openslr.org by ~60x in our measurement.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

REPO = "openslr/librispeech_asr"
SUBSET = "train.clean.100"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
DL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def list_shards() -> list[dict]:
    r = requests.get(API, timeout=60)
    r.raise_for_status()
    return [x for x in r.json()
            if x.get("type") == "file" and x["path"].startswith(f"all/{SUBSET}/")]


def download(url: str, out: Path, timeout: int = 120) -> int:
    """Resumable single-file download; returns bytes written."""
    pos = out.stat().st_size if out.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    r = requests.get(url, stream=True, timeout=timeout, headers=headers)
    if r.status_code == 416:
        return pos
    r.raise_for_status()
    mode = "ab" if pos else "wb"
    t0 = time.time()
    start = pos
    with open(out, mode) as f:
        for chunk in r.iter_content(1024 * 512):
            f.write(chunk)
            pos += len(chunk)
    dt = max(time.time() - t0, 0.1)
    rate = (pos - start) / 1e6 / dt
    print(f"  {out.name}: {pos/1e6:.0f} MB  ({rate:.1f} MB/s)", flush=True)
    return pos


def main() -> int:
    dest = config.DATA_RAW / "librispeech_parquet"
    dest.mkdir(parents=True, exist_ok=True)
    shards = list_shards()
    if not shards:
        print("ERROR: no shards found", file=sys.stderr)
        return 1
    shards.sort(key=lambda x: x["path"])
    total_bytes = sum(x.get("size", 0) for x in shards)
    print(f"Found {len(shards)} shards, ~{total_bytes/1e9:.2f} GB total", flush=True)

    t0 = time.time()
    for i, s in enumerate(shards, 1):
        name = Path(s["path"]).name
        out = dest / name
        if out.exists() and out.stat().st_size == s.get("size", 0):
            print(f"[{i}/{len(shards)}] {name} already complete", flush=True)
            continue
        print(f"[{i}/{len(shards)}] {name}  ({s.get('size',0)/1e9:.2f} GB)", flush=True)
        try:
            download(DL + s["path"], out)
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return 1

    print(f"\nDONE in {time.time()-t0:.0f}s -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
