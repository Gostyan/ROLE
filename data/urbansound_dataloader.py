"""
UrbanSound8k dataset loader for few-shot audio classification.

Dataset facts:
  - 10 urban sound classes
  - 8,732 clips (varied length, 16kHz target)
  - 10 built-in folds -> remapped to 5 OOF folds (fold 1+2->1, 3+4->2, ...)
  - Few-shot defaults: 5-way, 1-shot / 5-shot, Q=10
  - Open-set: 5 inlier + 5 outlier = all 10 classes (dynamic mode)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.base_dataset import BaseDataset

US8K_ROOT  = "/root/AOTE/data/UrbanSound8k/UrbanSound8K"
US8K_META  = os.path.join(US8K_ROOT, "metadata", "UrbanSound8K.csv")
US8K_AUDIO = os.path.join(US8K_ROOT, "audio")


class UrbanSound8kDataset(BaseDataset):
    """UrbanSound8k with its 10 built-in folds remapped to 5 OOF folds."""

    def __init__(self, root_dir: str = US8K_ROOT, num_oof_folds: int = 5):
        """
        Args:
            root_dir:      Dataset root (contains metadata/ and audio/).
            num_oof_folds: Number of OOF folds. Must divide 10 evenly (1,2,5,10).
        """
        assert 10 % num_oof_folds == 0, "num_oof_folds must divide 10 evenly"
        self.root_dir = root_dir
        self.num_folds_cfg = num_oof_folds
        group_size = 10 // num_oof_folds   # original folds per OOF fold

        meta = pd.read_csv(US8K_META)

        # Build class mappings
        self.class_names = sorted(meta["class"].unique().tolist())
        self.num_classes = len(self.class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Remap folds: original fold 1..10 -> OOF fold 1..num_oof_folds
        # e.g. for 5 folds: orig 1,2->1   3,4->2   5,6->3   7,8->4   9,10->5
        def _remap(orig_fold):
            return ((orig_fold - 1) // group_size) + 1

        self._paths = []
        self._class_idxs = []
        self._folds = []
        for _, row in meta.iterrows():
            path = os.path.join(US8K_AUDIO,
                                f"fold{int(row['fold'])}",
                                row["slice_file_name"])
            self._paths.append(path)
            self._class_idxs.append(self.class_to_idx[row["class"]])
            self._folds.append(_remap(int(row["fold"])))

        self._paths = np.array(self._paths)
        self._class_idxs = np.array(self._class_idxs)
        self._folds = np.array(self._folds)

        print(f"Loaded UrbanSound8k: {self.num_classes} classes, "
              f"{len(self._paths)} samples, {num_oof_folds} OOF folds")

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
        # 10 classes: perfect 5+5 split for open-set (dynamic mode)
        return {
            "n_way":        5,
            "shots":        [1, 5],
            "queries":      10,
            "num_folds":    self.num_folds_cfg,
            "fold_range":   list(range(1, self.num_folds_cfg + 1)),
            "inlier_pool":  list(range(0, 5)),
            "outlier_pool": list(range(5, 10)),
            "openset_mode": "dynamic",   # dynamic b/c tight pool (10 classes)
            "external_outlier_source": "fsd",  # only 10 classes total; use FSD for outlier queries
            "cache_subdir": "urbansound_embeddings",
        }
