"""
Dataset registry for AOTE few-shot experiments.

Usage:
    from data.dataset_registry import get_dataset, DATASET_ALIASES

    dataset = get_dataset("esc50")
    dataset = get_dataset("fsd")
    dataset = get_dataset("esc50")
    dataset = get_dataset("urbansound")
    dataset = get_dataset("voxceleb")
    dataset = get_dataset("birdclef")

Each dataset provides a `dataset_config` dict with recommended few-shot
protocol parameters (n_way, shots, queries, fold_range, inlier/outlier pools).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CACHE_DIR, PROJECT_ROOT

# ------------------------------------------------------------------ #
#  Dataset aliases and descriptions                                    #
# ------------------------------------------------------------------ #
DATASET_ALIASES = {
    "esc50":      "ESC-50 (50 classes, 5 folds, 40 samples/class)",
    "fsd":        "FSDKaggle2018 (41 classes, 5 artificial folds)",
    "urbansound": "UrbanSound8k (10 classes, 5 OOF folds from 10 built-in)",
    "voxceleb":   "VoxCeleb1 (1251 speakers, 5 artificial folds)",
    "birdclef":   "BirdCLEF2020 via HuggingFace BirdSet (715 species)",
}

# Cache directory for each dataset
DATASET_CACHE_DIRS = {
    "esc50":      os.path.join(PROJECT_ROOT, "cache", "esc50_embeddings"),
    "fsd":        os.path.join(PROJECT_ROOT, "cache", "fsd_embeddings"),
    "urbansound": os.path.join(PROJECT_ROOT, "cache", "urbansound_embeddings"),
    "voxceleb":   os.path.join(PROJECT_ROOT, "cache", "voxceleb_embeddings"),
    "birdclef":   os.path.join(PROJECT_ROOT, "cache", "birdclef_embeddings"),
}


def get_dataset(name: str, **kwargs):
    """
    Instantiate a dataset by its alias.

    Args:
        name:    Dataset alias (see DATASET_ALIASES).
        **kwargs: Forwarded to the dataset constructor.

    Returns:
        An instance of the appropriate BaseDataset subclass.
    """
    name = name.lower().strip()

    if name == "esc50":
        from data.esc50_dataloader import ESC50Dataset
        return ESC50Dataset(**kwargs)
    elif name == "fsd":
        from data.fsdkaggle_dataloader import FSDKaggle2018Dataset
        return FSDKaggle2018Dataset(**kwargs)
    elif name == "urbansound":
        from data.urbansound_dataloader import UrbanSound8kDataset
        return UrbanSound8kDataset(**kwargs)
    elif name == "voxceleb":
        from data.voxceleb_dataloader import VoxCeleb1Dataset
        return VoxCeleb1Dataset(**kwargs)
    elif name == "birdclef":
        from data.birdclef_dataloader import BirdCLEFDataset
        return BirdCLEFDataset(**kwargs)
    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Available: {list(DATASET_ALIASES.keys())}"
        )


def get_cache_dir(name: str) -> str:
    """Return the cache directory path for the given dataset alias."""
    name = name.lower().strip()
    if name not in DATASET_CACHE_DIRS:
        raise ValueError(f"Unknown dataset: '{name}'")
    return DATASET_CACHE_DIRS[name]


def list_datasets() -> None:
    """Print a summary of all available datasets."""
    print("Available datasets:")
    for alias, desc in DATASET_ALIASES.items():
        cache = DATASET_CACHE_DIRS[alias]
        status = "cached" if os.path.isdir(cache) and os.listdir(cache) else "no cache"
        print(f"  {alias:<12} {desc}  [{status}]")
