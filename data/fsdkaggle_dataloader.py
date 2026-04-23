"""
FSDKaggle2018 dataset loader for few-shot audio classification.

Dataset facts:
  - 41 sound event classes (general audio events)
  - 9,473 training clips, min 94 / max 300 per class
  - No built-in folds -> 5 stratified folds created deterministically (seed=42)
  - Few-shot defaults: 5-way, 1-shot / 5-shot, Q=10
  - Open-set: inlier pool = classes 0..19, outlier pool = classes 20..40
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.base_dataset import BaseDataset

FSD_ROOT = "/root/AOTE/data/FSDKaggle2018"
FSD_TRAIN_META = os.path.join(FSD_ROOT, "FSDKaggle2018.meta", "train_post_competition.csv")
FSD_AUDIO_TRAIN = os.path.join(FSD_ROOT, "FSDKaggle2018.audio_train")
FSD_NUM_FOLDS = 5
FSD_FOLD_SEED = 42


class FSDKaggle2018Dataset(BaseDataset):
    """FSDKaggle2018 dataset with artificial 5-fold cross-validation."""

    def __init__(self, root_dir: str = FSD_ROOT, num_folds: int = FSD_NUM_FOLDS,
                 fold_seed: int = FSD_FOLD_SEED, verified_only: bool = False):
        """
        Args:
            root_dir:      Dataset root directory.
            num_folds:     Number of OOF folds to create (default 5).
            fold_seed:     Seed for reproducible fold assignment.
            verified_only: If True, use only manually_verified=1 samples.
        """
        self.root_dir = root_dir
        self.num_folds_cfg = num_folds
        self.fold_seed = fold_seed

        meta = pd.read_csv(FSD_TRAIN_META)
        if verified_only:
            meta = meta[meta["manually_verified"] == 1].reset_index(drop=True)

        # Build class mappings
        self.class_names = sorted(meta["label"].unique().tolist())
        self.num_classes = len(self.class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Assign folds (stratified by class)
        rng = np.random.default_rng(fold_seed)
        fold_col = np.zeros(len(meta), dtype=int)
        for cls_name in self.class_names:
            idx = meta.index[meta["label"] == cls_name].tolist()
            perm = rng.permutation(len(idx))
            for rank, pos in enumerate(perm):
                fold_col[idx[pos]] = (rank % num_folds) + 1  # folds 1..num_folds
        meta = meta.copy()
        meta["fold"] = fold_col

        # Populate per-sample info
        self._paths: List[str] = []
        self._class_idxs: List[int] = []
        self._folds: List[int] = []
        for _, row in meta.iterrows():
            path = os.path.join(FSD_AUDIO_TRAIN, row["fname"])
            self._paths.append(path)
            self._class_idxs.append(self.class_to_idx[row["label"]])
            self._folds.append(int(row["fold"]))

        self._paths = np.array(self._paths)
        self._class_idxs = np.array(self._class_idxs)
        self._folds = np.array(self._folds)

        print(f"Loaded FSDKaggle2018: {self.num_classes} classes, "
              f"{len(self._paths)} samples, {num_folds} folds")

    # ------------------------------------------------------------------ #
    def get_fold_data(self, fold: int, exclude_fold: bool = True):
        mask = (self._folds != fold) if exclude_fold else (self._folds == fold)
        return [(self._paths[i], int(self._class_idxs[i]),
                 self.idx_to_class[int(self._class_idxs[i])])
                for i in np.where(mask)[0]]

    def get_class_samples(self, class_idx: int, fold: int,
                          exclude_fold: bool = True,
                          max_samples: Optional[int] = None) -> List[str]:
        fold_mask = (self._folds != fold) if exclude_fold else (self._folds == fold)
        class_mask = self._class_idxs == class_idx
        idxs = np.where(fold_mask & class_mask)[0]
        paths = self._paths[idxs].tolist()
        if max_samples is not None:
            paths = paths[:max_samples]
        return paths

    def get_all_classes_in_fold(self, fold: int, exclude_fold: bool = True) -> List[int]:
        mask = (self._folds != fold) if exclude_fold else (self._folds == fold)
        return sorted(set(self._class_idxs[mask].tolist()))

    def get_statistics(self) -> Dict:
        counts = {self.idx_to_class[c]: int((self._class_idxs == c).sum())
                  for c in range(self.num_classes)}
        return {"num_classes": self.num_classes, "samples_per_class": counts,
                "total_samples": len(self._paths), "num_folds": self.num_folds_cfg}

    @property
    def dataset_config(self) -> Dict:
        return {
            "n_way":        5,
            "shots":        [1, 5],
            "queries":      10,
            "num_folds":    self.num_folds_cfg,
            "fold_range":   list(range(1, self.num_folds_cfg + 1)),
            "inlier_pool":  list(range(0, 20)),
            "outlier_pool": list(range(20, self.num_classes)),
            "openset_mode": "dynamic",
            "cache_subdir": "fsd_embeddings",
        }
