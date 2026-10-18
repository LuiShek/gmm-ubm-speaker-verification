# Methodology

Why the system is built this way, and what the numbers actually mean.

## Why a UBM at all

The detector is a likelihood ratio:

```
score(x) = log p(x | speaker) - log p(x | UBM)
```

The UBM is the denominator. Without it you are comparing raw likelihoods,
which mostly measure how well a model fits the audio — not who is speaking.
The UBM is also the MAP adaptation prior, so a speaker model built from 5
utterances inherits sensible parameter values instead of overfitting.

## Why MAP adaptation rather than training from scratch

With ~5 enrollment utterances a per-speaker GMM trained independently would
overfit badly. MAP adaptation shrinks the speaker model toward the UBM in
proportion to how much data each component actually sees:

```
alpha_i = n_i / (n_i + r)          # r = 16
mu_i    = alpha_i * E_i[x] + (1 - alpha_i) * mu_UBM_i
```

Components the speaker never exercises keep their UBM values, so unseen
acoustic classes contribute approximately zero evidence either way instead
of dragging the score down. This is the main reason GMM-UBM beat decoupled
GMMs.

## Why per-utterance trials, not averaged trials

An earlier version of this project scored a test speaker as the **mean** of
three utterance scores. That is an ensemble: averaging cancels per-utterance
noise and produces a much lower EER (0.43% vs 5.42%).

It is also not what a deployment does. A real system verifies you on the one
recording it just received. Averaging three reduces variance by roughly a
factor of √3 in the ideal case — and the empirical gap is much larger than
that, which suggests the averaged score was exploiting something specific to
the LibriSpeech recording structure.

**The reported number is per single utterance.**

## Why the confidence interval is speaker-clustered

Trials are not independent. Two impostor trials sharing a test speaker share
recording session, microphone, and channel. Treating 8,000 impostor trials as
8,000 independent samples would understate the interval badly.

So the bootstrap resamples **speakers** with replacement and recomputes EER on
the surviving trial subset, 1,000 times. The 2.5th and 97.5th percentiles form
the reported interval.

## Why DET is plotted only above the measurement floor

With 8,000 impostor trials, the smallest measurable false-accept rate is
1/8000 ≈ 0.0125%. Plotting the DET curve down to 1e-4 shows a region where the
curve is determined by a handful of trials and looks like chance. The figure
clips to the measured support and marks the floor.

## What minDCF measures

The normalised detection cost at `p_target = 0.01`:

```
DCF = C_miss * p_target * P_miss + C_fa * (1 - p_target) * P_fa
```

Normalised by the cost of the trivial strategy (always accept or always
reject), so `1.0` means no better than trivial and `0.0` means perfect. This
project scores **0.720**, meaning it cuts the cost of the best trivial
strategy by about 28%.

An earlier implementation reported 0.000 — that was a bug. The cumulative
sums for false accepts and false rejects were indexed such that a single
lucky top-ranked target produced zero cost at both. Fixed and covered by
`tests/test_pipeline.py::test_min_dcf_bounds`.

## Known weaknesses

1. **Channel mismatch.** The dominant real-world failure mode for GMM-UBM.
   Clean-read training does not prepare the model for telephone or far-field
   audio. Score normalisation mitigates but does not solve this.
2. **50 speakers.** EER resolution is limited by speaker count. More speakers
   is the only real fix; more utterances per speaker does not help the
   speaker-level uncertainty.
3. **Cohort leakage.** T-norm cohort is drawn from eval utterances. A
   disjoint cohort would raise the EER and is the stricter protocol.
4. **No spoofing defence.** A replay of the target's voice is accepted.

## Reproducing

Seeds are fixed (`config.SEED = 1234`) for speaker selection, feature
sampling, trial planning, and the bootstrap. Re-running the full pipeline on
the same data should reproduce the reported numbers exactly.
