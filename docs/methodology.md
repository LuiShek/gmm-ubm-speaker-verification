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

## Why the T-norm cohort must span speakers

T-norm subtracts the score a model gives *other people's* audio. That only
works if the cohort is actually other people.

The first implementation did not do that. It filled a buffer by walking the
test dictionary in order and stopping once it had `4 × N_TNORM_IMPOSTORS`
utterances, so all 50 cohort utterances came from the single first speaker.
Every model in the evaluation was normalised against one voice, and that
voice's idiosyncrasies became the reference point for everyone.

Measured on the same 13,315 trials:

| Cohort | EER | minDCF |
|---|---:|---:|
| none (raw LLR) | 1.45% | 0.380 |
| 50 utterances from **one** speaker | 5.42% | 0.720 |
| one utterance from each **other** speaker | **0.98%** | **0.219** |

The middle row was the published result. Being wrong in the conservative
direction is still being wrong: it understated the system by a factor of five
and hid the fact that normalisation works at all.

The fix lives in `evaluate.py`: `cohort_for(speaker)` returns one utterance per
other speaker and never the target's own audio. `scripts/baselines.py`
reproduces all three rows, so the old number stays auditable rather than being
quietly replaced.

## Why per-utterance trials, not averaged trials

An earlier version of this project scored a test speaker as the **mean** of
three utterance scores. That is an ensemble: averaging cancels per-utterance
noise and produces a much lower EER. Under the buggy one-speaker cohort it
reported 0.43%; that figure is withdrawn, because it was measured on the same
broken normalisation as the old 5.42% headline and the two cannot be compared
across protocols.

The argument for per-utterance trials does not depend on that number. A real
system verifies you on the one recording it just received. Averaging three
reduces variance by roughly a factor of √3 in the ideal case, which flatters
the metric while describing a deployment that does not exist.

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
project scores **0.219**, meaning it cuts the cost of the best trivial
strategy by about 78%.

An earlier implementation reported 0.000 — that was a bug. The cumulative
sums for false accepts and false rejects were indexed such that a single
lucky top-ranked target produced zero cost at both. Fixed and covered by
`tests/test_pipeline.py::test_min_dcf_bounds`.

## Why not a neural speaker embedding (x-vector / ECAPA-TDNN)

This is a deliberate scope decision, and the number that decision costs is
known. SpeechBrain's ECAPA-TDNN recipe reports **0.80% EER** on VoxCeleb1-O
with score normalisation, against **0.98%** here.

That is not an apples-to-apples comparison, and saying so is part of the
result:

| | This project | ECAPA-TDNN reference |
|---|---|---|
| Training data | 180 speakers, clean read audiobooks | 5,994 VoxCeleb2 speakers, in the wild |
| Eval speakers | 50 | 40 (VoxCeleb1-O) |
| Channel | One corpus, similar equipment | Interview/web video, heavy mismatch |
| Enrolment | 5 utterances, MAP adaptation at test time | Network trained on millions of speaker labels |
| Decision score | LLR + cohort T-norm | Cosine on embeddings + s-norm |
| Training cost | CPU, minutes | GPU, hours to days |

What a neural embedding buys: discriminative training. A TDNN trained on
speaker labels learns the *contrast* between speakers directly, instead of
modelling each speaker's acoustic density and subtracting a background model.
It also produces a fixed-length vector, so scoring becomes a cosine or PLDA,
and it is far more robust to channel mismatch because mismatch is in the
training distribution.

What it costs: GPU time, a large labelled corpus, and interpretability. No
component of an embedding corresponds to "this mixture covers this vowel".

GMM-UBM is kept here because (a) it trains on CPU in minutes from 180
speakers, (b) every parameter is inspectable, and (c) it is the correct
baseline to measure a neural system against. That is the natural follow-up
experiment: swap the front end for a pretrained ECAPA embedding, keep this
trial protocol, and report the gap *on this data* rather than across papers.

## Known weaknesses

1. **Channel mismatch.** The dominant real-world failure mode for GMM-UBM.
   Clean-read training does not prepare the model for telephone or far-field
   audio. Score normalisation mitigates but does not solve this.
2. **50 speakers.** EER resolution is limited by speaker count. More speakers
   is the only real fix; more utterances per speaker does not help the
   speaker-level uncertainty.
3. **Cohort leakage.** The per-speaker T-norm cohort uses one test utterance
   from each other speaker, and those utterances also appear in the trial set.
   A fully disjoint cohort would be stricter and is the next experiment to run.
4. **No spoofing defence.** A replay of the target's voice is accepted.

## Reproducing

Seeds are fixed (`config.SEED = 1234`) for speaker selection, feature
sampling, trial planning, and the bootstrap. Re-running the full pipeline on
the same data should reproduce the reported numbers exactly.
