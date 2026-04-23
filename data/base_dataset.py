"""
Base dataset interface for AOTE few-shot audio classification.
All dataset loaders must implement this interface to be compatible with
EpisodeSampler, FewShotEvaluator, and the unified experiment runner.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np


class BaseDataset(ABC):
    """
    Abstract base class for few-shot audio datasets.

    Every subclass must populate:
        class_names      : List[str]  - sorted list of class names
        num_classes      : int
        class_to_idx     : Dict[str, int]
        idx_to_class     : Dict[int, str]

    And implement the three abstract methods below.
    """

    # ------------------------------------------------------------------ #
    #  Required abstract methods                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_fold_data(self,
                      fold: int,
                      exclude_fold: bool = True) -> List[Tuple[str, int, str]]:
        """
        Return samples for a fold.

        Args:
            fold:         fold index (dataset-defined range)
            exclude_fold: True  -> return samples NOT in this fold (train pool)
                          False -> return samples in this fold only (test pool)

        Returns:
            List of (audio_path, class_idx, class_name)
        """

    @abstractmethod
    def get_class_samples(self,
                          class_idx: int,
                          fold: int,
                          exclude_fold: bool = True,
                          max_samples: Optional[int] = None) -> List[str]:
        """
        Return audio paths for a specific class and fold selection.

        Args:
            class_idx:    dataset class index
            fold:         fold index
            exclude_fold: True -> exclude this fold (training pool)
            max_samples:  cap on returned samples (None = all)

        Returns:
            List of audio file paths
        """

    @abstractmethod
    def get_all_classes_in_fold(self,
                                fold: int,
                                exclude_fold: bool = True) -> List[int]:
        """
        Return class indices that have at least one sample in the selection.

        Args:
            fold:         fold index
            exclude_fold: True -> classes outside this fold

        Returns:
            Sorted list of class indices
        """

    # ------------------------------------------------------------------ #
    #  Optional: override for custom statistics                            #
    # ------------------------------------------------------------------ #

    def get_statistics(self) -> Dict:
        """Return a statistics dictionary (override in subclasses)."""
        return {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }

    # ------------------------------------------------------------------ #
    #  Dataset configuration (few-shot protocol defaults)                  #
    # ------------------------------------------------------------------ #

    @property
    def dataset_config(self) -> Dict:
        """
        Default few-shot protocol parameters for this dataset.
        Subclasses should override to return dataset-specific defaults.

        Keys:
            n_way          : int   - number of classes per episode
            shots          : list  - list of N-shot values to evaluate
            queries        : int   - query samples per class
            num_folds      : int   - number of OOF folds
            fold_range     : list  - list of valid fold indices (1-based)
            inlier_pool    : list  - class indices for open-set inlier pool
            outlier_pool   : list  - class indices for open-set outlier pool
            openset_mode   : str   - "fixed" or "dynamic"
            cache_subdir   : str   - subdirectory name under cache/
        """
        half = self.num_classes // 2
        return {
            "n_way":        5,
            "shots":        [1, 5],
            "queries":      10,
            "num_folds":    5,
            "fold_range":   list(range(1, 6)),
            "inlier_pool":  list(range(0, half)),
            "outlier_pool": list(range(half, self.num_classes)),
            "openset_mode": "dynamic",
            "cache_subdir": "unknown_embeddings",
        }

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"num_classes={self.num_classes}, "
                f"folds={self.dataset_config['fold_range']})")
