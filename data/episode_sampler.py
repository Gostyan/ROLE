"""
Episode sampler for few-shot learning.
Generates N-way K-shot episodes for meta-learning evaluation.
Supports both closed-set and open-set (OSFSL) episode protocols.
"""

import numpy as np
import random
from typing import List, Tuple, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from data.esc50_dataloader import ESC50Dataset
from data.base_dataset import BaseDataset

# ESC-50 Open-Set class pool partition (fixed for reproducibility)
# Class 0-24 → inlier pool, class 25-49 → outlier pool
INLIER_POOL = list(range(0, 25))
OUTLIER_POOL = list(range(25, 50))


class EpisodeSampler:
    """
    Sampler for few-shot learning episodes.
    
    An episode consists of:
    - Support set: N classes × K shots
    - Query set: N classes × Q queries
    """
    
    def __init__(self, dataset, seed: int = None, valid_paths: set = None,
                 outlier_dataset=None, outlier_valid_paths: set = None):
        """
        Initialize episode sampler.

        Args:
            dataset: Primary dataset (inlier source).
            seed: Random seed for reproducibility.
            valid_paths: Set of allowed paths within the primary cache
                (for cache-intersection constraints).
            outlier_dataset: Optional external dataset to sample outlier classes
                from. Required when the primary dataset has too few classes to
                satisfy n_way_in + n_way_out (e.g. UrbanSound8k with
                only 10 classes). When set, outlier class selection and query
                path lookup are routed to this dataset instead.
            outlier_valid_paths: Set of valid paths within the external outlier
                dataset's cache (analogous to valid_paths for the primary cache).
        """
        self.dataset = dataset
        self.seed = seed
        self.valid_paths = valid_paths
        self.outlier_dataset = outlier_dataset
        self.outlier_valid_paths = outlier_valid_paths

        if seed is not None:
            self.set_seed(seed)
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def sample_episode(self,
                      fold: int,
                      n_way: int = N_WAY,
                      n_shot: int = 1,
                      n_query: int = QUERIES_PER_CLASS,
                      exclude_fold: bool = True,
                      sampling_mode: str = "cross_fold") -> Dict:
        """
        Sample a single N-way K-shot episode.
        
        Args:
            fold: Fold number (1-5) - this is the TEST fold
            n_way: Number of classes in the episode
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
            exclude_fold: If True, sample support and query from other folds (training)
                         If False, sample both from the test fold
            sampling_mode: Sampling strategy:
                - "cross_fold": Default mode, sample from training folds (exclude_fold=True)
                - "oof_test_fold": Out-of-fold mode, sample from test fold with auto Q adjustment
        
        Returns:
            Dictionary containing:
                - 'support_paths': List of (path, class_idx) tuples
                - 'query_paths': List of (path, class_idx) tuples
                - 'class_mapping': Dict mapping episode class indices (0 to n_way-1) to dataset class indices
                - 'class_names': List of class names in the episode
        """
        # Handle OOF test fold mode
        if sampling_mode == "oof_test_fold":
            # Force sampling from test fold
            exclude_fold = False
            # Auto-adjust Q based on K.
            # For ESC-50: 8 samples/class/fold (K+Q <= 8)
            # For other datasets: use conservative floor of samples_per_fold
            # Approximate: assume at least 8 samples/class/fold as a safe default;
            # real constraint is checked inside get_class_samples.
            n_query = max(1, 8 - n_shot)  # conservative default
        # Get available classes
        # For ESC-50 protocol: sample both support and query from training folds (exclude test fold)
        # The test fold is used only for final evaluation
        if exclude_fold:
            # Both support and query from training folds (other 4 folds)
            available_classes = self.dataset.get_all_classes_in_fold(fold, exclude_fold=True)
        else:
            # Both support and query from test fold (for final validation)
            available_classes = self.dataset.get_all_classes_in_fold(fold, exclude_fold=False)
        
        if len(available_classes) < n_way:
            raise ValueError(
                f"Not enough classes available (need {n_way}, have {len(available_classes)})"
            )
        
        # Randomly sample n_way classes
        episode_class_indices = random.sample(available_classes, n_way)
        
        # Create class mapping: episode index (0 to n_way-1) -> dataset index
        class_mapping = {i: class_idx for i, class_idx in enumerate(episode_class_indices)}
        class_names = [self.dataset.idx_to_class[class_idx] for class_idx in episode_class_indices]
        
        # Sample support and query sets
        support_paths = []
        query_paths = []
        
        for episode_idx, dataset_class_idx in class_mapping.items():
            # Get all samples for this class from the appropriate fold(s)
            all_samples = self.dataset.get_class_samples(
                dataset_class_idx, fold=fold, exclude_fold=exclude_fold
            )
            
            # We need n_shot + n_query samples total
            total_needed = n_shot + n_query
            
            if len(all_samples) < total_needed:
                raise ValueError(
                    f"Not enough samples for class {dataset_class_idx} "
                    f"(need {total_needed}, have {len(all_samples)})"
                )
            
            # Randomly sample without replacement
            random.shuffle(all_samples)
            support_samples = all_samples[:n_shot]
            query_samples = all_samples[n_shot:n_shot + n_query]
            
            # Add to episode (with episode class index)
            for path in support_samples:
                support_paths.append((path, episode_idx))
            
            for path in query_samples:
                query_paths.append((path, episode_idx))
        
        # Shuffle the order (optional, but good practice)
        random.shuffle(support_paths)
        random.shuffle(query_paths)
        
        return {
            'support_paths': support_paths,
            'query_paths': query_paths,
            'class_mapping': class_mapping,
            'class_names': class_names,
            'fold': fold,
            'n_way': n_way,
            'n_shot': n_shot,
            'n_query': n_query,
        }
    
    def sample_episodes(self,
                       fold: int,
                       num_episodes: int,
                       n_way: int = N_WAY,
                       n_shot: int = 1,
                       n_query: int = QUERIES_PER_CLASS,
                       exclude_fold: bool = True,
                       sampling_mode: str = "cross_fold") -> List[Dict]:
        """
        Sample multiple episodes.
        
        Args:
            fold: Fold number (1-5)
            num_episodes: Number of episodes to sample
            n_way: Number of classes per episode
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
            exclude_fold: If True, cross-fold sampling
            sampling_mode: "cross_fold" or "oof_test_fold"
        
        Returns:
            List of episode dictionaries
        """
        episodes = []
        for _ in range(num_episodes):
            episode = self.sample_episode(
                fold=fold,
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                exclude_fold=exclude_fold,
                sampling_mode=sampling_mode
            )
            episodes.append(episode)
        
        return episodes

    def sample_openset_episode(self,
                               fold: int,
                               n_way_in: int = 5,
                               n_way_out: int = 5,
                               n_shot: int = 1,
                               k_query_in: int = 3,
                               k_query_out: int = 3,
                               exclude_fold: bool = True,
                               mode: str = 'fixed') -> Dict:
        """
        Sample an open-set episode with both inlier and outlier query samples.

        The support set contains ONLY inlier classes.
        The query set contains a mix of inlier and outlier class samples.

        Args:
            fold: Fold number (1-5), the test fold.
            n_way_in: Number of inlier classes (support + query).
            n_way_out: Number of outlier classes (query only).
            n_shot: Support samples per inlier class.
            k_query_in: Query samples per inlier class.
            k_query_out: Query samples per outlier class.
            exclude_fold: If True, sample from training folds.
            mode: Sampling mode. 'fixed' uses INLIER_POOL/OUTLIER_POOL (default),
                  'dynamic' samples inlier/outlier from all available classes randomly.

        Returns:
            Dictionary containing:
              - 'support_paths': List of (path, episode_class_idx) for inlier support
              - 'query_paths': List of (path, episode_class_idx_or_-1) for mixed query
              - 'outlier_labels': List of int (0=inlier, 1=outlier) aligned with query_paths
              - 'class_mapping': Dict mapping episode idx -> dataset class idx (inlier only)
              - 'class_names': List of inlier class names
              - 'outlier_class_mapping': Dict mapping outlier idx -> dataset class idx
              - 'outlier_class_names': List of outlier class names
              - fold, n_way_in, n_way_out, n_shot, k_query_in, k_query_out
        """
        # Get available classes for this fold
        if exclude_fold:
            available_classes = set(
                self.dataset.get_all_classes_in_fold(fold, exclude_fold=True)
            )
        else:
            available_classes = set(
                self.dataset.get_all_classes_in_fold(fold, exclude_fold=False)
            )

            # Dynamic or fixed class pool sampling.
        # When outlier_dataset is configured, always route outlier class selection
        # to it. Required for tight-pool datasets (UrbanSound8k: 10 classes
        # total) that cannot supply both n_way_in inlier AND n_way_out outlier classes.
        if self.outlier_dataset is not None:
            # ---- External outlier source ----------------------------------------
            # Inlier classes: from primary dataset (dynamic or fixed pool)
            if mode == 'dynamic':
                if len(available_classes) < n_way_in:
                    raise ValueError(
                        f"Not enough inlier classes (need {n_way_in}, "
                        f"have {len(available_classes)})"
                    )
                inlier_classes = random.sample(sorted(available_classes), n_way_in)
            else:
                cfg = getattr(self.dataset, 'dataset_config', {})
                _inlier_pool = cfg.get('inlier_pool', INLIER_POOL)
                avail_inlier = [c for c in _inlier_pool if c in available_classes]
                if len(avail_inlier) < n_way_in:
                    raise ValueError(
                        f"Not enough inlier classes (need {n_way_in}, "
                        f"have {len(avail_inlier)})"
                    )
                inlier_classes = random.sample(avail_inlier, n_way_in)
            # Outlier classes: from external dataset (all classes available in fold)
            ext_classes = self.outlier_dataset.get_all_classes_in_fold(
                fold, exclude_fold=exclude_fold
            )
            if len(ext_classes) < n_way_out:
                raise ValueError(
                    f"Not enough external outlier classes (need {n_way_out}, "
                    f"have {len(ext_classes)} in "
                    f"{type(self.outlier_dataset).__name__})"
                )
            outlier_classes = random.sample(sorted(ext_classes), n_way_out)
            outlier_class_names = [
                self.outlier_dataset.idx_to_class[c] for c in outlier_classes
            ]
        elif mode == 'dynamic':
            # ---- Internal dynamic mode ------------------------------------------
            if len(available_classes) < n_way_in + n_way_out:
                raise ValueError(
                    f"Not enough classes (need {n_way_in + n_way_out}, "
                    f"have {len(available_classes)})"
                )
            all_classes = list(available_classes)
            inlier_classes = random.sample(all_classes, n_way_in)
            remaining = [c for c in all_classes if c not in inlier_classes]
            outlier_classes = random.sample(remaining, n_way_out)
            outlier_class_names = [self.dataset.idx_to_class[c] for c in outlier_classes]
        else:
            # ---- Internal fixed mode --------------------------------------------
            # Fixed: use dataset-specific inlier/outlier pools (fallback to global INLIER_POOL)
            cfg = getattr(self.dataset, 'dataset_config', {})
            _inlier_pool  = cfg.get('inlier_pool',  INLIER_POOL)
            _outlier_pool = cfg.get('outlier_pool', OUTLIER_POOL)
            avail_inlier  = [c for c in _inlier_pool  if c in available_classes]
            avail_outlier = [c for c in _outlier_pool if c in available_classes]

            if len(avail_inlier) < n_way_in:
                raise ValueError(
                    f"Not enough inlier classes (need {n_way_in}, have {len(avail_inlier)})"
                )
            if len(avail_outlier) < n_way_out:
                raise ValueError(
                    f"Not enough outlier classes (need {n_way_out}, have {len(avail_outlier)})"
                )

            inlier_classes = random.sample(avail_inlier, n_way_in)
            outlier_classes = random.sample(avail_outlier, n_way_out)
            outlier_class_names = [self.dataset.idx_to_class[c] for c in outlier_classes]

        # Class mapping (inlier only, 0 to n_way_in-1)
        class_mapping = {i: c for i, c in enumerate(inlier_classes)}
        class_names = [self.dataset.idx_to_class[c] for c in inlier_classes]
        outlier_class_mapping = {i: c for i, c in enumerate(outlier_classes)}

        support_paths = []
        query_paths = []
        outlier_labels = []

        # --- Inlier classes: support + query ---
        for ep_idx, ds_class in class_mapping.items():
            samples = self.dataset.get_class_samples(
                ds_class, fold=fold, exclude_fold=exclude_fold
            )
            if getattr(self, 'valid_paths', None) is not None:
                samples = [s for s in samples if s in self.valid_paths]
                
            total_needed = n_shot + k_query_in
            if len(samples) < total_needed:
                raise ValueError(
                    f"Not enough samples for inlier class {ds_class} "
                    f"(need {total_needed}, have {len(samples)})"
                )
            random.shuffle(samples)
            sup = samples[:n_shot]
            qry = samples[n_shot:n_shot + k_query_in]

            for path in sup:
                support_paths.append((path, ep_idx))
            for path in qry:
                query_paths.append((path, ep_idx))
                outlier_labels.append(0)  # inlier

        # --- Outlier classes: query only ---
        # Route to external outlier_dataset when configured (UrbanSound8k).
        _outlier_src   = self.outlier_dataset if self.outlier_dataset is not None else self.dataset
        _outlier_valid = self.outlier_valid_paths if self.outlier_dataset is not None else self.valid_paths
        for out_idx, ds_class in outlier_class_mapping.items():
            samples = _outlier_src.get_class_samples(
                ds_class, fold=fold, exclude_fold=exclude_fold
            )
            if _outlier_valid is not None:
                samples = [s for s in samples if s in _outlier_valid]
                
            if len(samples) < k_query_out:
                raise ValueError(
                    f"Not enough samples for outlier class {ds_class} "
                    f"(need {k_query_out}, have {len(samples)})"
                )
            random.shuffle(samples)
            qry = samples[:k_query_out]
            for path in qry:
                query_paths.append((path, -1))  # -1 marks outlier
                outlier_labels.append(1)  # outlier

        # Shuffle query set (preserving alignment with outlier_labels)
        combined = list(zip(query_paths, outlier_labels))
        random.shuffle(combined)
        query_paths, outlier_labels = zip(*combined)
        query_paths = list(query_paths)
        outlier_labels = list(outlier_labels)

        random.shuffle(support_paths)

        return {
            'support_paths': support_paths,
            'query_paths': query_paths,
            'outlier_labels': outlier_labels,
            'class_mapping': class_mapping,
            'class_names': class_names,
            'outlier_class_mapping': outlier_class_mapping,
            'outlier_class_names': outlier_class_names,
            'fold': fold,
            'n_way_in': n_way_in,
            'n_way_out': n_way_out,
            'n_shot': n_shot,
            'k_query_in': k_query_in,
            'k_query_out': k_query_out,
            'outlier_external': self.outlier_dataset is not None,
        }

    def sample_openset_episodes(self,
                                fold: int,
                                num_episodes: int,
                                n_way_in: int = 5,
                                n_way_out: int = 5,
                                n_shot: int = 1,
                                k_query_in: int = 3,
                                k_query_out: int = 3,
                                exclude_fold: bool = True) -> List[Dict]:
        """Sample multiple open-set episodes. See sample_openset_episode for args."""
        return [
            self.sample_openset_episode(
                fold=fold, n_way_in=n_way_in, n_way_out=n_way_out,
                n_shot=n_shot, k_query_in=k_query_in, k_query_out=k_query_out,
                exclude_fold=exclude_fold
            )
            for _ in range(num_episodes)
        ]


def test_episode_sampler():
    """Test episode sampler."""
    print("=" * 80)
    print("Testing Episode Sampler")
    print("=" * 80)
    
    # Load dataset
    dataset = ESC50Dataset()
    
    # Create sampler
    sampler = EpisodeSampler(dataset, seed=42)
    
    # Test single episode sampling
    print("\n" + "=" * 80)
    print("Test 1: Sample a single 5-way 1-shot episode")
    print("=" * 80)
    
    episode = sampler.sample_episode(
        fold=1,
        n_way=5,
        n_shot=1,
        n_query=10,
        exclude_fold=True
    )
    
    print(f"\nEpisode info:")
    print(f"  Fold: {episode['fold']}")
    print(f"  N-way: {episode['n_way']}")
    print(f"  N-shot: {episode['n_shot']}")
    print(f"  N-query per class: {episode['n_query']}")
    print(f"  Total support samples: {len(episode['support_paths'])}")
    print(f"  Total query samples: {len(episode['query_paths'])}")
    
    print(f"\nClass mapping:")
    for episode_idx, dataset_idx in episode['class_mapping'].items():
        class_name = episode['class_names'][episode_idx]
        print(f"  Episode class {episode_idx} -> Dataset class {dataset_idx} ({class_name})")
    
    print(f"\nFirst 3 support samples:")
    for path, class_idx in episode['support_paths'][:3]:
        class_name = episode['class_names'][class_idx]
        print(f"  Class {class_idx} ({class_name}): {os.path.basename(path)}")
    
    print(f"\nFirst 3 query samples:")
    for path, class_idx in episode['query_paths'][:3]:
        class_name = episode['class_names'][class_idx]
        print(f"  Class {class_idx} ({class_name}): {os.path.basename(path)}")
    
    # Test multiple episode sampling
    print("\n" + "=" * 80)
    print("Test 2: Sample 10 episodes with 5-way 5-shot")
    print("=" * 80)
    
    episodes = sampler.sample_episodes(
        fold=1,
        num_episodes=10,
        n_way=5,
        n_shot=5,
        n_query=10,
        exclude_fold=True
    )
    
    print(f"\nSampled {len(episodes)} episodes")
    print(f"Each episode has:")
    print(f"  Support: {episodes[0]['n_way']} classes × {episodes[0]['n_shot']} shots = {len(episodes[0]['support_paths'])} samples")
    print(f"  Query: {episodes[0]['n_way']} classes × {episodes[0]['n_query']} queries = {len(episodes[0]['query_paths'])} samples")
    
    # Test reproducibility
    print("\n" + "=" * 80)
    print("Test 3: Test reproducibility with same seed")
    print("=" * 80)
    
    sampler1 = EpisodeSampler(dataset, seed=123)
    sampler2 = EpisodeSampler(dataset, seed=123)
    
    ep1 = sampler1.sample_episode(fold=1, n_way=5, n_shot=1, n_query=10)
    ep2 = sampler2.sample_episode(fold=1, n_way=5, n_shot=1, n_query=10)
    
    same_classes = ep1['class_mapping'] == ep2['class_mapping']
    same_support = ep1['support_paths'] == ep2['support_paths']
    same_query = ep1['query_paths'] == ep2['query_paths']
    
    print(f"\nSame classes sampled: {same_classes}")
    print(f"Same support samples: {same_support}")
    print(f"Same query samples: {same_query}")
    
    if same_classes and same_support and same_query:
        print("✓ Reproducibility test passed!")
    else:
        print("✗ Reproducibility test failed!")
    
    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_episode_sampler()
