"""
utils/config.py — Project-wide constants and paths.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the ROLE repository (two levels up from this file: utils/ -> ROLE/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Default embedding cache directory
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# Dataset root directories (override with env-vars or edit here)
ESC50_ROOT = os.environ.get(
    "ESC50_ROOT",
    os.path.join(PROJECT_ROOT, "data", "ESC-50-master"),
)
FSD_ROOT = os.environ.get(
    "FSD_ROOT",
    os.path.join(PROJECT_ROOT, "data", "FSDKaggle2018"),
)
URBANSOUND_ROOT = os.environ.get(
    "URBANSOUND_ROOT",
    os.path.join(PROJECT_ROOT, "data", "UrbanSound8K"),
)

# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000   # Hz — matches AST HuggingFace processor expectation
N_MELS      = 128      # mel-filterbank bins
N_FFT       = 1024     # FFT window size
HOP_LENGTH  = 512      # STFT hop
WIN_LENGTH  = 1024     # STFT window length

# Number of time frames for a 5-second clip at 16 kHz with the above settings
# floor((16000*5 - WIN_LENGTH) / HOP_LENGTH) + 1 = 498, padded to 512
ESC50_TDIM  = 512

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Default random seeds used by FewShotEvaluator
SEEDS = [0]
