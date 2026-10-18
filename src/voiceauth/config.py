"""
Voice Authentication - GMM-UBM speaker verification.

Implements the GMM-UBM approach established in the NIST Speaker Recognition
Evaluation era (2000-2004): a universal background model trained on pooled
speaker data, per-speaker MAP adaptation from that UBM, and a log-likelihood
ratio detector with score normalisation.
"""
from pathlib import Path

# config.py lives at <repo>/src/voiceauth/config.py, so the project root is
# three levels up. Resolved once here and reused everywhere.
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_FEAT = ROOT / "data" / "features"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

for d in (DATA_RAW, DATA_FEAT, MODELS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

# ---------- Audio / features ----------
SR = 16000                    # Hz, matches LibriSpeech
WIN_MS = 20                   # 20 ms window
HOP_MS = 10                   # 10 ms frame rate
WIN_LEN = int(SR * WIN_MS / 1000)   # 320 samples
HOP_LEN = int(SR * HOP_MS / 1000)   # 160 samples
N_FFT = 512
PREEMPH = 0.97

N_MELS = 40
FMIN = 300                    # Hz, per paper telephone-band emphasis
FMAX = 8000                   # Nyquist for 16 kHz
N_CEPS = 19                   # cepstra kept (c0 dropped)
N_DELTAS = 19                 # first-order deltas over +-2
FEAT_DIM = N_CEPS + N_DELTAS  # 38
DELTA_WIN = 2                 # paper: two to the left and two to the right

# VAD: energy-based, paper calls it a self-normalizing energy detector
VAD_PERCENTILE = 30.0         # drop bottom 30% of frames as silence/non-speech
VAD_MIN_RATIO = 0.10          # never drop more than 90% of frames

# ---------- UBM ----------
N_MIXTURES = 256              # CPU-friendly; bump to 512/1024 on GPU
UBM_MAX_ITER = 30
UBM_MAX_FRAMES = 400_000      # sampled subsample to train on
UBM_FRAME_CHUNK = 2000        # posterior chunking to keep RAM headroom
COVARIANCE_FLOOR = 1e-3       # prevents var collapse
VAR_INIT = 1.0

# ---------- MAP adaptation ----------
RELEVANCE_R = 16.0            # paper: r=16, adapt means only
ADAPT_MEANS = True
ADAPT_VARS = False
ADAPT_WEIGHTS = False

# ---------- Trials / scoring ----------
SEED = 1234
UTTS_PER_SPK_ENROLL = 5       # enrollment utterances per speaker
UTTS_PER_SPK_TEST = 999       # 999 = use every remaining utterance
N_TNORM_IMPOSTORS = 50        # cohort size for T-norm
T_NORM_ENABLED = True
# Impostor trials are sampled to this budget. False-accept is already measured
# well at this size; it is false-reject (the noisy side of EER) that needs
# every available target trial.
MAX_IMPOSTOR_TRIALS = 8000
BOOTSTRAP_ITERS = 1000        # speaker-clustered bootstrap for the EER CI

# ---------- Dataset ----------
LIBRISPEECH_TAR = DATA_RAW / "train-clean-100.tar.gz"
LIBRISPEECH_DIR = DATA_RAW / "LibriSpeech" / "train-clean-100"
VOXCELEB_DIR = DATA_RAW / "voxceleb1"
N_SPEAKERS_UBM = 180          # background speakers for UBM
N_SPEAKERS_EVAL = 50          # eval speakers (disjoint from UBM)
# train-clean-100 has ~251 speakers; 180+50=230 leaves ~21 spare for
# speakers that fail the min-utterance filter.
