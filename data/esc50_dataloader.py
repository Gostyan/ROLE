"""
ESC-50 dataset loader.
Handles loading ESC-50 metadata and creating data structures for few-shot learning.
"""

import pandas as pd
import os
import sys
from typing import List, Dict, Tuple
import numpy as np
from functools import lru_cache

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from data.base_dataset import BaseDataset


class ESC50Dataset(BaseDataset):
    """
    ESC-50 dataset loader for few-shot audio classification.
    
    ESC-50 contains:
    - 2000 audio clips (5 seconds each)
    - 50 classes (40 samples per class)
    - 5 folds for cross-validation
    """
    
    def __init__(self, root_dir: str = ESC50_ROOT):
        """
        Initialize ESC-50 dataset loader.
        
        Args:
            root_dir: Root directory of ESC-50 dataset
        """
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, "audio")
        self.meta_path = os.path.join(root_dir, "meta/esc50.csv")
        
        # Check if paths exist
        if not os.path.exists(self.meta_path):
            # Try alternate path
            self.meta_path = os.path.join(root_dir, "meta.csv")
        
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"ESC-50 metadata not found at {self.meta_path}")
        
        if not os.path.exists(self.audio_dir):
            raise FileNotFoundError(f"ESC-50 audio directory not found at {self.audio_dir}")
        
        # Load metadata
        self.metadata = pd.read_csv(self.meta_path)
        
        # Create mappings
        self._create_mappings()
        
        print(f"✓ Loaded ESC-50 dataset:")
        print(f"  Total samples: {len(self.metadata)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Samples per class: {len(self.metadata) // self.num_classes}")
        print(f"  Folds: {sorted(self.metadata['fold'].unique())}")
    
    def _create_mappings(self):
        """Create useful mappings for the dataset."""
        # Class mappings
        self.class_names = sorted(self.metadata['category'].unique())
        self.num_classes = len(self.class_names)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Create file to class mapping
        self.file_to_class = {}
        self.file_to_fold = {}
        self.file_to_target = {}
        
        for _, row in self.metadata.iterrows():
            filename = row['filename']
            category = row['category']
            fold = row['fold']
            target = row['target']  # Original ESC-50 class index (0-49)
            
            self.file_to_class[filename] = category
            self.file_to_fold[filename] = fold
            self.file_to_target[filename] = target
    
    @lru_cache(maxsize=None)
    def get_fold_data(self, fold: int, exclude_fold: bool = True) -> List[Tuple[str, int, str]]:
        """
        Get data for a specific fold.
        
        Args:
            fold: Fold number (1-5)
            exclude_fold: If True, return all data EXCEPT this fold (for training/support)
                         If False, return only this fold (for testing/query)
        
        Returns:
            List of (audio_path, class_idx, class_name) tuples
        """
        if exclude_fold:
            fold_data = self.metadata[self.metadata['fold'] != fold]
        else:
            fold_data = self.metadata[self.metadata['fold'] == fold]
        
        data = []
        for _, row in fold_data.iterrows():
            audio_path = os.path.join(self.audio_dir, row['filename'])
            class_name = row['category']
            class_idx = self.class_to_idx[class_name]
            data.append((audio_path, class_idx, class_name))
        
        return data
    
    @lru_cache(maxsize=None)
    def get_class_samples(self, 
                         class_idx: int, 
                         fold: int, 
                         exclude_fold: bool = True,
                         max_samples: int = None) -> List[str]:
        """
        Get all audio file paths for a specific class and fold.
        
        Args:
            class_idx: Class index (0-49)
            fold: Fold number (1-5)
            exclude_fold: If True, exclude samples from this fold
            max_samples: Maximum number of samples to return (None = all)
        
        Returns:
            List of audio file paths
        """
        class_name = self.idx_to_class[class_idx]
        
        # Filter by class and fold
        if exclude_fold:
            class_data = self.metadata[
                (self.metadata['category'] == class_name) & 
                (self.metadata['fold'] != fold)
            ]
        else:
            class_data = self.metadata[
                (self.metadata['category'] == class_name) & 
                (self.metadata['fold'] == fold)
            ]
        
        # Get file paths
        audio_paths = [
            os.path.join(self.audio_dir, row['filename']) 
            for _, row in class_data.iterrows()
        ]
        
        # Limit if requested
        if max_samples is not None:
            audio_paths = audio_paths[:max_samples]
        
        return audio_paths
    
    @lru_cache(maxsize=None)
    def get_all_classes_in_fold(self, fold: int, exclude_fold: bool = True) -> List[int]:
        """
        Get all class indices that have samples in/outside a fold.
        
        Args:
            fold: Fold number (1-5)
            exclude_fold: If True, get classes outside this fold
        
        Returns:
            List of class indices
        """
        fold_data = self.get_fold_data(fold, exclude_fold=exclude_fold)
        class_indices = sorted(set([class_idx for _, class_idx, _ in fold_data]))
        return class_indices
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.metadata),
            'num_classes': self.num_classes,
            'num_folds': len(self.metadata['fold'].unique()),
            'samples_per_class': {},
            'samples_per_fold': {}
        }
        
        # Samples per class
        for class_name in self.class_names:
            count = len(self.metadata[self.metadata['category'] == class_name])
            stats['samples_per_class'][class_name] = count
        
        # Samples per fold
        for fold in sorted(self.metadata['fold'].unique()):
            count = len(self.metadata[self.metadata['fold'] == fold])
            stats['samples_per_fold'][fold] = count
        
        return stats


    @property
    def dataset_config(self) -> dict:
        """Default few-shot protocol for ESC-50."""
        return {
            "n_way":        5,
            "shots":        [1, 5],
            "queries":      10,
            "num_folds":    5,
            "fold_range":   [1, 2, 3, 4, 5],
            "inlier_pool":  list(range(0, 25)),   # classes 0-24
            "outlier_pool": list(range(25, 50)),  # classes 25-49
            "openset_mode": "fixed",
            "cache_subdir": "esc50_embeddings",
        }


def test_esc50_loader():
    """Test ESC-50 dataset loader."""
    print("=" * 60)
    print("Testing ESC-50 Dataset Loader")
    print("=" * 60)
    
    # Initialize dataset
    dataset = ESC50Dataset()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    stats = dataset.get_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Number of folds: {stats['num_folds']}")
    
    print(f"\nSamples per fold:")
    for fold, count in stats['samples_per_fold'].items():
        print(f"  Fold {fold}: {count} samples")
    
    print(f"\nFirst 5 class names:")
    for i, class_name in enumerate(dataset.class_names[:5]):
        count = stats['samples_per_class'][class_name]
        print(f"  {i}: {class_name} ({count} samples)")
    
    # Test fold data retrieval
    print("\n" + "=" * 60)
    print("Testing Fold Data Retrieval")
    print("=" * 60)
    
    fold = 1
    train_data = dataset.get_fold_data(fold, exclude_fold=True)
    test_data = dataset.get_fold_data(fold, exclude_fold=False)
    
    print(f"Fold {fold}:")
    print(f"  Training samples (excluding fold {fold}): {len(train_data)}")
    print(f"  Test samples (fold {fold} only): {len(test_data)}")
    
    # Test class samples retrieval
    print("\n" + "=" * 60)
    print("Testing Class Samples Retrieval")
    print("=" * 60)
    
    class_idx = 0
    class_name = dataset.idx_to_class[class_idx]
    samples = dataset.get_class_samples(class_idx, fold=1, exclude_fold=True)
    
    print(f"Class {class_idx} ({class_name}):")
    print(f"  Samples outside fold 1: {len(samples)}")
    print(f"  Sample paths:")
    for path in samples[:3]:
        print(f"    {os.path.basename(path)}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_esc50_loader()
