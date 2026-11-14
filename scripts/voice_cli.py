"""Run the voiceauth CLI from a source checkout without installing.

    py scripts/voice_cli.py --help

If the package is installed (`py -m pip install -e .`), the equivalent
console script is `voiceauth`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from voiceauth.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
