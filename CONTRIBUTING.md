# Contributing

## Setup

bash
py -m pip install -r requirements.txt
py -m pytest tests/ -q # must pass before you push


The data pipeline downloads 6.4 GB. If you only want to touch code, the tests
run without it — they use synthetic arrays.

## Before opening a PR

1. py -m pytest tests/ -q — 12 checks, must be green.
2. py -m py_compile src/voiceauth/*.py scripts/*.py — must exit 0.
3. If you changed the model or the protocol, re-run evaluate.py and paste
 the new outputs/report.txt in the PR description. A metric change without
 evidence is not reviewable.

## What is worth changing

- Bug fixes in feature extraction, EM, MAP adaptation, or scoring.
- New normalisation schemes (Z-norm, adaptive T-norm, score calibration).
- Additional datasets behind the existing dataset.py interface.
- Performance with a before/after measurement.

## What will be rejected

- Changes to evaluate.py metrics that make numbers look better without
 stating what got easier. This project has already had one inflated result
 (score averaging across utterances); do not add another.
- Reporting an EER without a confidence interval.
- Removing the limitations section from the README.

## Reporting a result

Always state: EER, the 95% CI, trial counts (targets and impostors), speaker
count, and the dataset. A number without those five things is not a result.

## Code style

Match what is there. Type hints on public functions, from __future__ import
annotations at the top, no bare except:, no commented-out code.
