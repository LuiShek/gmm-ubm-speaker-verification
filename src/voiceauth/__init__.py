"""voiceauth - GMM-UBM speaker verification.

Feature extraction, UBM training, MAP adaptation and likelihood-ratio scoring
for text-independent speaker verification.
"""

__version__ = "1.0.0"

from . import config, dataset, features, gmm_ubm

__all__ = ["config", "dataset", "features", "gmm_ubm", "__version__"]
