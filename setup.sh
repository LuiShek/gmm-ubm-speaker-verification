#!/usr/bin/env bash
set -euo pipefail

echo "=== voiceauth setup ==="

command -v py >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || {
 echo "Error: Python 3.10+ is required."; exit 1; }

PY=(command -v py || command -v python3)
"PY" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'

echo "Installing dependencies..."
"PY" -m pip install -r requirements.txt

echo "Running test suite..."
"PY" -m pytest tests/ -q

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo " 1. py scripts/download_data.py # 6.4 GB dataset"
echo " 2. py scripts/train_ubm.py # ~30 min"
echo " 3. py scripts/extract_features.py"
echo " 4. py scripts/evaluate.py"
echo " 5. py scripts/make_figures.py"
