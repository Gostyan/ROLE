"""
Episode evaluator for few-shot learning experiments.
Handles episode sampling, evaluation, and results aggregation.
"""

import numpy as np
import json
import os
import sys
from typing import Dict, List, Callable
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from data.esc50_dataloader import ESC50Dataset
from data.dataset_registry import get_dataset, get_cache_dir
from data.episode_sampler import EpisodeSampler
from data.cache_embeddings import load_cached_embeddings, get_episode_embeddings, get_episode_patch_embeddings
from evaluation.metrics import (
    compute_episode_metrics,
    aggregate_episode_metrics,
    format_results
)
from data.episode_sampler import INLIER_POOL, OUTLIER_POOL


class FewShotEvaluator:
    """
    Evaluator for few-shot learning methods.
    Handles episode sampling, method execution, and results aggregation.
    """
    
    def __init__(self,
                 cache_dir: str = None,
                 num_episodes: int = NUM_EPISODES,
                 seeds: List[int] = SEEDS,
                 verbose: bool = True,
                 dataset=None):
        """
        Initialize evaluator.

        Args:
            cache_dir:     Directory with cached embeddings.
                           Defaults to the ESC-50 cache (backward compatible).
                           Pass None to auto-resolve from the dataset name.
            num_episodes:  Number of episodes per fold/seed.
            seeds:         List of random seeds.
            verbose:       Print progress information.
            dataset:       Dataset to use. Can be:
                             - None        -> ESC-50 (default, backward compat)
                             - str         -> dataset alias (e.g. "fsd", "urbansound")
                             - BaseDataset -> pre-constructed dataset instance
        """
        self.num_episodes = num_episodes
        self.seeds = seeds
        self.verbose = verbose

        # Resolve dataset
        if dataset is None:
            self.dataset = ESC50Dataset()
            self.dataset_name = "esc50"
        elif isinstance(dataset, str):
            self.dataset_name = dataset.lower()
            self.dataset = get_dataset(self.dataset_name)
        else:
            # Pre-constructed BaseDataset instance
            self.dataset = dataset
            self.dataset_name = type(dataset).__name__.lower()

        # Resolve cache directory
        if cache_dir is None:
            try:
                resolved_cache = get_cache_dir(self.dataset_name)
            except ValueError:
                resolved_cache = CACHE_DIR  # fallback
        else:
            resolved_cache = cache_dir
        self.cache_dir = resolved_cache

        # Load cached embeddings
        if self.verbose:
            print(f"Dataset: {self.dataset_name}")
            print("Loading cached embeddings...")
        self.cache_data = load_cached_embeddings(resolved_cache)

        # Dataset config (for fold defaults)
        self._ds_cfg = getattr(self.dataset, 'dataset_config', {})
        num_folds = self._ds_cfg.get('num_folds', NUM_FOLDS)

        if self.verbose:
            print(f"Evaluator initialized:")
            print(f"  Episodes per fold/seed: {num_episodes}")
            print(f"  Seeds: {seeds}")
            print(f"  Total evaluations: {num_episodes * len(seeds) * num_folds}")

    def evaluate_episode(self,
                        method_fn: Callable,
                        episode: Dict,
                        measure_time: bool = True) -> Dict:
        """
        Evaluate a single episode with a given method.
        
        Args:
            method_fn: Function that takes (support_emb, support_lbl, query_emb) 
                      and returns (predictions, probabilities)
            episode: Episode dictionary from EpisodeSampler
            measure_time: If True, measure execution time
        
        Returns:
            Dictionary of metrics for this episode
        """
        # Get embeddings for this episode
        support_emb, support_lbl, query_emb, query_lbl = get_episode_embeddings(
            episode, self.cache_data
        )
        
        # Run method and measure time
        if measure_time:
            start_time = time.time()
        
        result = method_fn(support_emb, support_lbl, query_emb)
        # Support both 2-tuple and 3-tuple returns
        if len(result) == 3:
            predictions, probabilities, _ = result
        else:
            predictions, probabilities = result
        
        if measure_time:
            inference_time = time.time() - start_time
            timing_info = {'inference_time': inference_time}
        else:
            timing_info = None
        
        # Compute metrics
        metrics = compute_episode_metrics(
            predictions, query_lbl, probabilities, timing_info
        )
        
        return metrics
    
    def evaluate_fold(self,
                     method_fn: Callable,
                     fold: int,
                     n_way: int = N_WAY,
                     n_shot: int = 1,
                     n_query: int = QUERIES_PER_CLASS,
                     seed: int = 0,
                     exclude_fold: bool = True) -> List[Dict]:
        """
        Evaluate a method on a single fold.
        
        Args:
            method_fn: Method function
            fold: Fold number (1-5)
            n_way: Number of classes
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
            seed: Random seed
            exclude_fold: If True, use cross-fold sampling
        
        Returns:
            List of episode metrics
        """
        # Create episode sampler
        sampler = EpisodeSampler(self.dataset, seed=seed)
        
        # Sample episodes
        episodes = sampler.sample_episodes(
            fold=fold,
            num_episodes=self.num_episodes,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            exclude_fold=exclude_fold
        )
        
        # Evaluate each episode
        episode_metrics = []
        
        iterator = tqdm(episodes, desc=f"Fold {fold}, Seed {seed}") if self.verbose else episodes
        
        for episode in iterator:
            metrics = self.evaluate_episode(method_fn, episode)
            episode_metrics.append(metrics)
        
        return episode_metrics

    def evaluate_openset_episode(self,
                                 method_fn,
                                 episode: Dict,
                                 measure_time: bool = True) -> Dict:
        """
        Evaluate a single open-set episode.

        Args:
            method_fn: Function that takes (support_emb, support_lbl, query_emb)
                      and returns (predictions, probabilities_or_logits, outlier_scores).
                      outlier_scores may be None for non-open-set methods.
            episode: Open-set episode dictionary from EpisodeSampler.sample_openset_episode
            measure_time: If True, measure execution time

        Returns:
            Dictionary of metrics including open-set metrics
        """
        support_emb, support_lbl, query_emb, query_lbl = get_episode_embeddings(
            episode, self.cache_data
        )

        outlier_labels = np.array(episode['outlier_labels'])

        if measure_time:
            start_time = time.time()

        # Pass frequency-band patch embeddings when available (used by Glocal)
        _extra = {}
        if 'patch_embeddings' in self.cache_data:
            sup_p, qry_p = get_episode_patch_embeddings(episode, self.cache_data)
            if sup_p is not None:
                _extra = {'support_patches': sup_p, 'query_patches': qry_p}

        result = method_fn(support_emb, support_lbl, query_emb, **_extra)

        if measure_time:
            inference_time = time.time() - start_time
            timing_info = {'inference_time': inference_time}
        else:
            timing_info = None

        # Unpack result: method_fn can return 2-tuple or 3-tuple
        if len(result) == 3:
            predictions, probabilities, outlier_scores = result
        else:
            predictions, probabilities = result
            outlier_scores = None

        # For inlier queries, query_lbl has correct class labels
        # For outlier queries, query_lbl is -1 (set by sampler)
        metrics = compute_episode_metrics(
            predictions, query_lbl, probabilities, timing_info,
            outlier_scores=outlier_scores,
            outlier_labels=outlier_labels
        )

        return metrics

    def evaluate_openset_method(self,
                                method_fn,
                                n_way_in: int = 5,
                                n_way_out: int = 5,
                                n_shot: int = 1,
                                k_query_in: int = 3,
                                k_query_out: int = 3,
                                folds: List[int] = None,
                                exclude_fold: bool = True,
                                method_name: str = "Method",
                                outlier_dataset=None) -> Dict:
        """
        Full open-set evaluation across folds and seeds.

        Args:
            method_fn: Function returning (preds, probs, outlier_scores)
            n_way_in: Number of inlier classes
            n_way_out: Number of outlier classes
            n_shot: Support shots per class
            k_query_in: Query samples per inlier class
            k_query_out: Query samples per outlier class
            folds: List of folds to evaluate
            exclude_fold: If True, cross-fold sampling
            method_name: Name for logging
            outlier_dataset: Optional external dataset for outlier class sampling.
                Pass when the primary dataset has too few classes
                (e.g. UrbanSound8k with only 10 classes).

        Returns:
            Aggregated results dictionary
        """
        if folds is None:
            folds = self._ds_cfg.get('fold_range', list(range(1, NUM_FOLDS + 1)))

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"Evaluating {method_name} (Open-Set)")
            print(f"{'=' * 80}")
            print(f"Config: {n_way_in}-in {n_way_out}-out {n_shot}-shot")
            print(f"Query: {k_query_in} inlier + {k_query_out} outlier per class")
            print(f"Folds: {folds}, Seeds: {self.seeds}")

        all_episode_metrics = []

        for fold in folds:
            for seed in self.seeds:
                sampler = EpisodeSampler(self.dataset, seed=seed,
                                         outlier_dataset=outlier_dataset)
                episodes = sampler.sample_openset_episodes(
                    fold=fold, num_episodes=self.num_episodes,
                    n_way_in=n_way_in, n_way_out=n_way_out,
                    n_shot=n_shot, k_query_in=k_query_in,
                    k_query_out=k_query_out, exclude_fold=exclude_fold
                )

                desc = f"OS Fold {fold}, Seed {seed}"
                iterator = tqdm(episodes, desc=desc) if self.verbose else episodes

                for episode in iterator:
                    metrics = self.evaluate_openset_episode(method_fn, episode)
                    all_episode_metrics.append(metrics)

        aggregated = aggregate_episode_metrics(all_episode_metrics)

        aggregated['config'] = {
            'method_name': method_name,
            'n_way_in': n_way_in,
            'n_way_out': n_way_out,
            'n_shot': n_shot,
            'k_query_in': k_query_in,
            'k_query_out': k_query_out,
            'num_folds': len(folds),
            'seeds': self.seeds,
            'episodes_per_config': self.num_episodes,
            'total_episodes': len(all_episode_metrics),
            'exclude_fold': exclude_fold,
            'openset': True,
        }

        if self.verbose:
            print(format_results(aggregated, method_name=method_name))

        return aggregated
    
    def evaluate_method(self,
                       method_fn: Callable,
                       n_way: int = N_WAY,
                       n_shot: int = 1,
                       n_query: int = QUERIES_PER_CLASS,
                       folds: List[int] = None,
                       exclude_fold: bool = True,
                       method_name: str = "Method") -> Dict:
        """
        Full evaluation of a method across all folds and seeds.
        
        Args:
            method_fn: Method function
            n_way: Number of classes
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
            folds: List of folds to evaluate (default: all 5)
            exclude_fold: If True, use cross-fold sampling
            method_name: Name for logging
        
        Returns:
            Dictionary with aggregated results
        """
        if folds is None:
            folds = self._ds_cfg.get('fold_range', list(range(1, NUM_FOLDS + 1)))
        
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"Evaluating {method_name}")
            print(f"{'=' * 80}")
            print(f"Configuration: {n_way}-way {n_shot}-shot")
            print(f"Folds: {folds}")
            print(f"Seeds: {self.seeds}")
            print(f"Episodes: {self.num_episodes} per fold/seed")
        
        all_episode_metrics = []
        
        # Evaluate each fold and seed combination
        for fold in folds:
            for seed in self.seeds:
                fold_metrics = self.evaluate_fold(
                    method_fn=method_fn,
                    fold=fold,
                    n_way=n_way,
                    n_shot=n_shot,
                    n_query=n_query,
                    seed=seed,
                    exclude_fold=exclude_fold
                )
                all_episode_metrics.extend(fold_metrics)
        
        # Aggregate results
        aggregated = aggregate_episode_metrics(all_episode_metrics)
        
        # Add configuration info
        aggregated['config'] = {
            'method_name': method_name,
            'n_way': n_way,
            'n_shot': n_shot,
            'n_query': n_query,
            'num_folds': len(folds),
            'seeds': self.seeds,
            'episodes_per_config': self.num_episodes,
            'total_episodes': len(all_episode_metrics),
            'exclude_fold': exclude_fold
        }
        
        if self.verbose:
            print(format_results(aggregated, method_name=method_name))
        
        return aggregated
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to save JSON
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        
        results_json = convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        if self.verbose:
            print(f"\n✓ Results saved to {output_path}")



    def evaluate_cross_cache_method(self,
                                    method_fn,
                                    support_cache,
                                    query_cache,
                                    n_way=5,
                                    n_shot=1,
                                    n_query=3,
                                    folds=None,
                                    exclude_fold=True,
                                    method_name="CrossCache Method"):
        """Evaluate with different caches for support and query (domain shift)."""
        import numpy as np
        from tqdm import tqdm
        from data.episode_sampler import EpisodeSampler
        from evaluation.metrics import compute_episode_metrics, aggregate_episode_metrics, format_results
        
        if folds is None:
            folds = list(range(1, 6))
        
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"Evaluating {method_name} (Cross-Cache)")
            print(f"Configuration: {n_way}-way {n_shot}-shot")
        
        all_episode_metrics = []
        
        for fold in folds:
            for seed in self.seeds:
                sampler = EpisodeSampler(self.dataset, seed=seed)
                episodes = sampler.sample_episodes(
                    fold=fold, n_way=n_way, n_shot=n_shot, n_query=n_query,
                    num_episodes=self.num_episodes, exclude_fold=exclude_fold
                )
                
                for episode in tqdm(episodes, desc=f"Fold {fold}, Seed {seed}", disable=not self.verbose):
                    # Support from clean cache
                    support_emb, support_lbl = [], []
                    for path, class_idx in episode['support_paths']:
                        if 'file_paths' in support_cache:
                            idx = support_cache['file_paths'].index(path)
                        else:
                            idx = support_cache['audio_paths'].index(path)
                        support_emb.append(support_cache['embeddings'][idx])
                        support_lbl.append(class_idx)
                    support_emb = np.array(support_emb)
                    support_lbl = np.array(support_lbl)
                    
                    # Query from degraded cache
                    query_emb, query_lbl = [], []
                    for path, class_idx in episode['query_paths']:
                        if 'file_paths' in query_cache:
                            idx = query_cache['file_paths'].index(path)
                        else:
                            idx = query_cache['audio_paths'].index(path)
                        query_emb.append(query_cache['embeddings'][idx])
                        query_lbl.append(class_idx)
                    query_emb = np.array(query_emb)
                    query_lbl = np.array(query_lbl)
                    
                    result = method_fn(support_emb, support_lbl, query_emb)
                    if len(result) == 3:
                        predictions, probabilities, _ = result
                    else:
                        predictions, probabilities = result
                    metrics = compute_episode_metrics(predictions, query_lbl, probabilities)
                    all_episode_metrics.append(metrics)
        
        aggregated = aggregate_episode_metrics(all_episode_metrics)
        aggregated['config'] = {
            'method_name': method_name,
            'n_way': n_way,
            'n_shot': n_shot,
            'n_query': n_query,
            'cross_cache': True,
            'query_scale': query_cache.get('scale', None)
        }
        
        if self.verbose:
            print(format_results(aggregated, method_name=method_name))
        
        return aggregated

    def evaluate_openset_cross_cache_method(self,
                                           method_fn,
                                           support_cache,
                                           query_cache,
                                           n_way_in=5,
                                           n_way_out=5,
                                           n_shot=1,
                                           k_query_in=3,
                                           k_query_out=3,
                                           folds=None,
                                           exclude_fold=True,
                                           mode='dynamic',
                                           adaptive_b=True,
                                           method_name="OpenSet CrossCache",
                                           outlier_cache=None,
                                           outlier_dataset=None):
        """Evaluate open-set with cross-cache and adaptive b.

        Args:
            outlier_cache: Pre-loaded cache dict for an external outlier dataset.
                Required when the primary dataset has too few classes (e.g. UrbanSound8k,
                UrbanSound8k). Outlier query embeddings are looked up here instead
                of query_cache.
            outlier_dataset: External dataset instance to sample outlier classes
                from. Use together with outlier_cache.
        """
        import numpy as np
        from tqdm import tqdm
        from data.episode_sampler import EpisodeSampler
        from evaluation.metrics import compute_episode_metrics, aggregate_episode_metrics, format_results
        
        if folds is None:
            folds = list(range(1, 6))
        
        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"Evaluating {method_name} (Open-Set, Cross-Cache)")
            print(f"Mode: {mode}, Adaptive b: {adaptive_b}")
        
        all_episode_metrics = []
        
        path_key_supp = 'file_paths' if 'file_paths' in support_cache else 'audio_paths'
        path_key_query = 'file_paths' if 'file_paths' in query_cache else 'audio_paths'
        
        valid_supp = set(support_cache[path_key_supp])
        valid_query = set(query_cache[path_key_query])
        valid_paths = valid_supp.intersection(valid_query)
        
        supp_path_to_idx = {path: idx for idx, path in enumerate(support_cache[path_key_supp])}
        query_path_to_idx = {path: idx for idx, path in enumerate(query_cache[path_key_query])}

        # Build external outlier cache index (for tight-pool datasets like UrbanSound8k)
        if outlier_cache is not None:
            path_key_out = 'file_paths' if 'file_paths' in outlier_cache else 'audio_paths'
            outlier_path_to_idx = {path: idx for idx, path in enumerate(outlier_cache[path_key_out])}
            outlier_valid_paths = set(outlier_cache[path_key_out])
        else:
            outlier_path_to_idx = None
            outlier_valid_paths = None

        for fold in folds:
            for seed in self.seeds:
                sampler = EpisodeSampler(self.dataset, seed=seed, valid_paths=valid_paths,
                                         outlier_dataset=outlier_dataset,
                                         outlier_valid_paths=outlier_valid_paths)
                episodes = sampler.sample_openset_episodes(
                    fold=fold, n_way_in=n_way_in, n_way_out=n_way_out,
                    n_shot=n_shot, k_query_in=k_query_in, k_query_out=k_query_out,
                    num_episodes=self.num_episodes, exclude_fold=exclude_fold
                )

                for episode in tqdm(episodes, desc=f"Fold {fold}, Seed {seed}", disable=not self.verbose):
                    support_emb, support_lbl = [], []
                    for path, class_idx in episode['support_paths']:
                        idx = supp_path_to_idx[path]
                        support_emb.append(support_cache['embeddings'][idx])
                        support_lbl.append(class_idx)
                    support_emb = np.array(support_emb)
                    support_lbl = np.array(support_lbl)
                    
                    query_emb, query_lbl, outlier_labels = [], [], []
                    _ep_ext = episode.get('outlier_external', False)
                    for (path, class_idx), is_outlier in zip(episode['query_paths'], episode['outlier_labels']):
                        if is_outlier and _ep_ext and outlier_path_to_idx is not None:
                            # External outlier: look up in the dedicated outlier cache
                            idx = outlier_path_to_idx[path]
                            query_emb.append(outlier_cache['embeddings'][idx])
                        else:
                            idx = query_path_to_idx[path]
                            query_emb.append(query_cache['embeddings'][idx])
                        query_lbl.append(class_idx)
                        outlier_labels.append(is_outlier)
                    query_emb = np.array(query_emb)
                    query_lbl = np.array(query_lbl)
                    outlier_labels = np.array(outlier_labels)
                    
                    if adaptive_b:
                        n_in = np.sum(outlier_labels == 0)
                        n_out = np.sum(outlier_labels == 1)
                        b_value = n_out / (n_in + n_out)
                    else:
                        b_value = 0.5
                    
                    import time
                    t0 = time.perf_counter()
                    try:
                        predictions, probabilities, outlier_scores = method_fn(support_emb, support_lbl, query_emb, b=b_value)
                    except TypeError:
                        predictions, probabilities, outlier_scores = method_fn(support_emb, support_lbl, query_emb)
                    t1 = time.perf_counter()
                    inference_time_ms = (t1 - t0) * 1000.0
                    
                    metrics = compute_episode_metrics(
                        predictions, query_lbl, probabilities,
                        timing_info={'inference_time_ms': inference_time_ms},
                        outlier_scores=outlier_scores, outlier_labels=outlier_labels
                    )
                    all_episode_metrics.append(metrics)
        
        aggregated = aggregate_episode_metrics(all_episode_metrics)
        aggregated['config'] = {
            'method_name': method_name,
            'mode': mode,
            'adaptive_b': adaptive_b,
            'query_scale': query_cache.get('scale', None)
        }
        
        if self.verbose:
            print(format_results(aggregated, method_name=method_name))
        
        return aggregated

def test_evaluator():
    """Test the evaluator with ProtoNet."""
    print("=" * 80)
    print("Testing Few-Shot Evaluator")
    print("=" * 80)
    
    # Import ProtoNet
    from models.protonet import ProtoNet
    
    # Create method function
    protonet = ProtoNet(distance_metric='cosine', temperature=10.0)
    
    def protonet_method(support_emb, support_lbl, query_emb):
        predictions, logits = protonet.predict(
            support_emb, support_lbl, query_emb, return_logits=True
        )
        probabilities = protonet.predict_proba(
            support_emb, support_lbl, query_emb
        )
        return predictions, probabilities
    
    # Create evaluator with reduced episodes for testing
    evaluator = FewShotEvaluator(
        num_episodes=10,  # Small number for testing
        seeds=[0],  # Single seed
        verbose=True
    )
    
    # Test on 1-shot
    print("\n" + "=" * 80)
    print("Test 1: 5-way 1-shot evaluation")
    print("=" * 80)
    
    results_1shot = evaluator.evaluate_method(
        method_fn=protonet_method,
        n_way=5,
        n_shot=1,
        n_query=10,
        folds=[1],  # Single fold for testing
        method_name="ProtoNet 1-shot (Test)"
    )
    
    # Test on 5-shot
    print("\n" + "=" * 80)
    print("Test 2: 5-way 5-shot evaluation")
    print("=" * 80)
    
    results_5shot = evaluator.evaluate_method(
        method_fn=protonet_method,
        n_way=5,
        n_shot=5,
        n_query=10,
        folds=[1],  # Single fold for testing
        method_name="ProtoNet 5-shot (Test)"
    )
    
    # Save results
    print("\n" + "=" * 80)
    print("Test 3: Save results")
    print("=" * 80)
    
    os.makedirs("/root/AOTE/results/test", exist_ok=True)
    evaluator.save_results(results_1shot, "/root/AOTE/results/test/protonet_1shot.json")
    evaluator.save_results(results_5shot, "/root/AOTE/results/test/protonet_5shot.json")
    
    print("\n" + "=" * 80)
    print("✓ Evaluator tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_evaluator()
