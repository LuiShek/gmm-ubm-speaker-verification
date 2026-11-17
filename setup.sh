#!/usr/bin/env bash
set -euo pipefail

echo "=== voiceauth setup ==="

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Error: Python 3.10+ is required." >&2
  exit 1
fi

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'

echo "Installing dependencies..."
"$PY" -m pip install -r requirements.txt

echo "Running test suite..."
"$PY" -m pytest tests/ -q

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo " 1. py scripts/download_data.py     # 6.4 GB dataset"
echo " 2. py scripts/train_ubm.py         # ~30 min"
echo " 3. py scripts/extract_features.py  # ~4 min"
echo " 4. py scripts/evaluate.py          # ~12 min"
echo " 5. py scripts/make_figures.py      # ~1 min"
