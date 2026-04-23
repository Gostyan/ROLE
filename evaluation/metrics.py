"""
Evaluation metrics for few-shot audio classification.
Implements all metrics required by the development plan.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score
import time


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute top-1 accuracy.
    
    Args:
        predictions: Predicted labels (N,)
        labels: True labels (N,)
    
    Returns:
        Accuracy (0-1)
    """
    return np.mean(predictions == labels)


def compute_macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score.
    
    Args:
        predictions: Predicted labels (N,)
        labels: True labels (N,)
    
    Returns:
        Macro F1 score (0-1)
    """
    return f1_score(labels, predictions, average='macro', zero_division=0)


def compute_ece(probabilities: np.ndarray, 
                labels: np.ndarray,
                n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probabilities: Predicted probabilities (N, n_way)
        labels: True labels (N,)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score (0-1, lower is better)
    """
    # Get predicted class and confidence
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Negative Log-Likelihood.
    
    Args:
        probabilities: Predicted probabilities (N, n_way)
        labels: True labels (N,)
    
    Returns:
        Average NLL (lower is better)
    """
    # Get probability of true class
    true_class_probs = probabilities[np.arange(len(labels)), labels]
    
    # Compute NLL (clip to avoid log(0))
    nll = -np.log(np.clip(true_class_probs, 1e-10, 1.0))
    
    return nll.mean()


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for a list of values.
    
    Args:
        values: Array of values
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        (margin of error, confidence interval half-width)
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    
    # Use t-distribution for small samples
    from scipy import stats
    confidence_interval = stats.t.interval(confidence, n - 1, loc=mean, scale=std / np.sqrt(n))
    
    margin = (confidence_interval[1] - confidence_interval[0]) / 2
    
    return mean, margin


def compute_auroc(outlier_scores: np.ndarray, outlier_labels: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve for outlier detection.

    Args:
        outlier_scores: (N,) predicted outlier scores (higher = more likely outlier)
        outlier_labels: (N,) binary labels (0=inlier, 1=outlier)

    Returns:
        AUROC score (0-1, higher is better). Returns 0.5 if only one class present.
    """
    if len(np.unique(outlier_labels)) < 2:
        return 0.5
    return roc_auc_score(outlier_labels, outlier_scores)


def compute_aupr(outlier_scores: np.ndarray, outlier_labels: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve for outlier detection.

    Args:
        outlier_scores: (N,) predicted outlier scores
        outlier_labels: (N,) binary labels (0=inlier, 1=outlier)

    Returns:
        AUPR score (0-1, higher is better).
    """
    if len(np.unique(outlier_labels)) < 2:
        return 0.5
    return average_precision_score(outlier_labels, outlier_scores)


def compute_precision_at_recall(outlier_scores: np.ndarray,
                                outlier_labels: np.ndarray,
                                target_recall: float = 0.9) -> float:
    """
    Compute Precision at a target Recall level.

    Args:
        outlier_scores: (N,) predicted outlier scores
        outlier_labels: (N,) binary labels (0=inlier, 1=outlier)
        target_recall: Target recall level. Default 0.9.

    Returns:
        Precision at target recall. Returns 0.0 if no outliers present.
    """
    if len(np.unique(outlier_labels)) < 2:
        return 0.0

    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(outlier_labels, outlier_scores)
    # recall is sorted descending; find first point where recall >= target
    valid = recall >= target_recall
    if valid.any():
        return float(precision[valid].max())
    return 0.0


def compute_episode_metrics(predictions: np.ndarray,
                           labels: np.ndarray,
                           probabilities: np.ndarray = None,
                           timing_info: Dict = None,
                           outlier_scores: np.ndarray = None,
                           outlier_labels: np.ndarray = None) -> Dict:
    """
    Compute all metrics for a single episode.

    Args:
        predictions: Predicted labels (N_query,)
        labels: True labels (N_query,)
        probabilities: Predicted probabilities (N_query, n_way), optional
        timing_info: Dictionary with timing information, optional
        outlier_scores: (N_query,) outlier scores, optional (for open-set)
        outlier_labels: (N_query,) binary outlier labels, optional (for open-set)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': compute_accuracy(predictions, labels),
        'macro_f1': compute_macro_f1(predictions, labels),
    }

    if probabilities is not None:
        metrics['ece'] = compute_ece(probabilities, labels)
        metrics['nll'] = compute_nll(probabilities, labels)

    # Open-set detection metrics
    if outlier_scores is not None and outlier_labels is not None:
        metrics['auroc'] = compute_auroc(outlier_scores, outlier_labels)
        metrics['aupr'] = compute_aupr(outlier_scores, outlier_labels)
        metrics['prec_at_90'] = compute_precision_at_recall(outlier_scores, outlier_labels, 0.9)
        # Inlier-only accuracy (accuracy on inlier queries only)
        inlier_mask = outlier_labels == 0
        if inlier_mask.any():
            metrics['inlier_accuracy'] = compute_accuracy(
                predictions[inlier_mask], labels[inlier_mask]
            )
            # Replace main accuracy with inlier_accuracy for open-set
            # (since overall accuracy including outliers is meaningless)
            metrics['accuracy_all'] = metrics['accuracy']  # Save for reference
            metrics['accuracy'] = metrics['inlier_accuracy']  # Main metric

    if timing_info is not None:
        metrics.update(timing_info)

    return metrics

    return metrics


def aggregate_episode_metrics(episode_metrics_list: List[Dict],
                             metrics_to_aggregate: List[str] = None) -> Dict:
    """
    Aggregate metrics across multiple episodes.
    
    Args:
        episode_metrics_list: List of metric dictionaries from episodes
        metrics_to_aggregate: List of metric names to aggregate (default: all)
    
    Returns:
        Dictionary with mean, std, and 95% CI for each metric
    """
    if metrics_to_aggregate is None:
        # Auto-detect metric names from first episode
        metrics_to_aggregate = [k for k in episode_metrics_list[0].keys() 
                               if isinstance(episode_metrics_list[0][k], (int, float))]
    
    aggregated = {}
    
    for metric_name in metrics_to_aggregate:
        values = np.array([ep[metric_name] for ep in episode_metrics_list])
        
        mean, ci_margin = compute_confidence_interval(values)
        
        aggregated[f'{metric_name}_mean'] = mean
        aggregated[f'{metric_name}_std'] = np.std(values, ddof=1)
        aggregated[f'{metric_name}_ci95'] = ci_margin
        aggregated[f'{metric_name}_min'] = np.min(values)
        aggregated[f'{metric_name}_max'] = np.max(values)
        aggregated[f'{metric_name}_all'] = values.tolist()  # raw per-episode values for significance testing
    
    aggregated['num_episodes'] = len(episode_metrics_list)
    
    return aggregated


def format_results(aggregated_metrics: Dict, 
                  method_name: str = "",
                  verbose: bool = True) -> str:
    """
    Format aggregated results for display.
    
    Args:
        aggregated_metrics: Dictionary from aggregate_episode_metrics
        method_name: Name of the method
        verbose: If True, show all metrics; if False, show only key metrics
    
    Returns:
        Formatted string
    """
    lines = []
    
    if method_name:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"{method_name} Results")
        lines.append(f"{'=' * 80}")
    
    # Core metrics
    core_metrics = ['accuracy', 'macro_f1', 'ece', 'nll',
                    'auroc', 'aupr', 'prec_at_90', 'inlier_accuracy']
    
    for metric in core_metrics:
        mean_key = f'{metric}_mean'
        ci_key = f'{metric}_ci95'
        
        if mean_key in aggregated_metrics:
            mean = aggregated_metrics[mean_key]
            ci = aggregated_metrics.get(ci_key, 0)
            
            metric_display = metric.replace('_', ' ').title()
            lines.append(f"{metric_display:20s}: {mean:.4f} ± {ci:.4f}")
    
    # Timing metrics
    if verbose:
        timing_metrics = [k.replace('_mean', '') for k in aggregated_metrics.keys() 
                         if '_time' in k and k.endswith('_mean')]
        
        if timing_metrics:
            lines.append(f"\nTiming:")
            for metric in timing_metrics:
                mean_key = f'{metric}_mean'
                if mean_key in aggregated_metrics:
                    mean = aggregated_metrics[mean_key]
                    metric_display = metric.replace('_', ' ').title()
                    lines.append(f"  {metric_display:25s}: {mean*1000:.2f} ms")
    
    # Episode count
    if 'num_episodes' in aggregated_metrics:
        lines.append(f"\nEvaluated on {aggregated_metrics['num_episodes']} episodes")
    
    return '\n'.join(lines)


def compare_methods(results_dict: Dict[str, Dict],
                   metric: str = 'accuracy',
                   sort_by_mean: bool = True) -> str:
    """
    Compare multiple methods on a specific metric.
    
    Args:
        results_dict: Dictionary mapping method names to aggregated metrics
        metric: Metric to compare
        sort_by_mean: If True, sort by mean value
    
    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"Method Comparison: {metric.title()}")
    lines.append(f"{'=' * 80}")
    
    # Collect data
    data = []
    for method_name, metrics in results_dict.items():
        mean_key = f'{metric}_mean'
        ci_key = f'{metric}_ci95'
        
        if mean_key in metrics:
            mean = metrics[mean_key]
            ci = metrics.get(ci_key, 0)
            data.append((method_name, mean, ci))
    
    # Sort if requested
    if sort_by_mean:
        data.sort(key=lambda x: x[1], reverse=True)
    
    # Format table
    lines.append(f"{'Method':<30s} {'Mean':>10s} {'95% CI':>10s}")
    lines.append("-" * 52)
    
    for method_name, mean, ci in data:
        lines.append(f"{method_name:<30s} {mean:10.4f} ± {ci:8.4f}")
    
    return '\n'.join(lines)


# Test functions
def test_metrics():
    """Test metric computation."""
    print("=" * 80)
    print("Testing Evaluation Metrics")
    print("=" * 80)
    
    # Create synthetic data
    np.random.seed(42)
    
    n_way = 5
    n_query = 50
    
    # Perfect predictions
    print("\n" + "=" * 80)
    print("Test 1: Perfect predictions")
    print("=" * 80)
    
    labels = np.random.randint(0, n_way, n_query)
    predictions = labels.copy()
    
    # Create one-hot probabilities
    probabilities = np.zeros((n_query, n_way))
    probabilities[np.arange(n_query), predictions] = 1.0
    
    metrics = compute_episode_metrics(predictions, labels, probabilities)
    
    print(f"Accuracy: {metrics['accuracy']:.4f} (expected: 1.0)")
    print(f"Macro F1: {metrics['macro_f1']:.4f} (expected: 1.0)")
    print(f"ECE: {metrics['ece']:.4f} (expected: ~0.0)")
    print(f"NLL: {metrics['nll']:.4f} (expected: ~0.0)")
    
    # Random predictions
    print("\n" + "=" * 80)
    print("Test 2: Random predictions")
    print("=" * 80)
    
    predictions_random = np.random.randint(0, n_way, n_query)
    probabilities_random = np.random.dirichlet(np.ones(n_way), n_query)
    
    metrics_random = compute_episode_metrics(predictions_random, labels, probabilities_random)
    
    print(f"Accuracy: {metrics_random['accuracy']:.4f} (expected: ~0.2)")
    print(f"Macro F1: {metrics_random['macro_f1']:.4f}")
    print(f"ECE: {metrics_random['ece']:.4f}")
    print(f"NLL: {metrics_random['nll']:.4f}")
    
    # Test aggregation
    print("\n" + "=" * 80)
    print("Test 3: Aggregation across episodes")
    print("=" * 80)
    
    episode_metrics = []
    for _ in range(100):
        preds = np.random.randint(0, n_way, n_query)
        probs = np.random.dirichlet(np.ones(n_way), n_query)
        metrics_ep = compute_episode_metrics(preds, labels, probs)
        episode_metrics.append(metrics_ep)
    
    aggregated = aggregate_episode_metrics(episode_metrics)
    
    print(format_results(aggregated, method_name="Random Baseline"))
    
    print("\n" + "=" * 80)
    print("✓ All metric tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_metrics()
