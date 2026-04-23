"""
Generic embedding cache builder for any AOTE dataset.

Usage:
    python data/build_dataset_cache.py --dataset fsd --device cuda
    python data/build_dataset_cache.py --dataset urbansound
    python data/build_dataset_cache.py --dataset voxceleb --batch_size 16
    python data/build_dataset_cache.py --dataset birdclef

This script replaces the dataset-specific cache scripts for all non-ESC50
datasets. For ESC-50, the original data/cache_embeddings.py still works.

Cache file format (identical to ESC-50):
  {dataset_name}_embeddings.pkl  containing:
    embeddings    : np.ndarray [N, D]
    file_paths    : List[str]
    labels        : np.ndarray [N]
    folds         : np.ndarray [N]
    embedding_dim : int
    num_samples   : int
    class_names   : List[str]
    class_to_idx  : Dict[str, int]
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import PROJECT_ROOT
from data.dataset_registry import get_dataset, get_cache_dir
from data.audio_preprocessing import preprocess_audio
from models.ast_feature_extractor import ASTFeatureExtractor

# Fallback TD-dim (will be auto-detected from first sample if possible)
DEFAULT_TDIM = 512


def build_cache(dataset_name: str,
                batch_size: int = 32,
                device: str = "cuda",
                force_recompute: bool = False,
                max_samples: int = None):
    """
    Extract and cache AST embeddings for all audio files in a dataset.

    Args:
        dataset_name:    Dataset alias (see dataset_registry.DATASET_ALIASES).
        batch_size:      Batch size for AST feature extraction.
        device:          'cuda' or 'cpu'.
        force_recompute: Re-extract even if cache already exists.
        max_samples:     Cap total samples (for quick smoke tests).
    """
    # ------------------------------------------------------------------ #
    #  Resolve paths                                                       #
    # ------------------------------------------------------------------ #
    cache_dir = get_cache_dir(dataset_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset_name}_embeddings.pkl")

    if os.path.exists(cache_file) and not force_recompute:
        print(f"Cache already exists at {cache_file}")
        answer = input("Recompute? (y/n): ").strip().lower()
        if answer != 'y':
            print("Skipping.")
            return

    # ------------------------------------------------------------------ #
    #  Load dataset                                                        #
    # ------------------------------------------------------------------ #
    print(f"Loading dataset: {dataset_name}")
    dataset = get_dataset(dataset_name)

    # ------------------------------------------------------------------ #
    #  Collect all file paths                                              #
    # ------------------------------------------------------------------ #
    print("Collecting audio files ...")
    all_paths, all_labels, all_folds = [], [], []

    ds_cfg = getattr(dataset, "dataset_config", {})
    fold_range = ds_cfg.get("fold_range", list(range(1, 6)))

    seen_paths = set()
    for fold in fold_range:
        fold_items = dataset.get_fold_data(fold, exclude_fold=False)
        for path, cls_idx, _ in fold_items:
            if path not in seen_paths:
                seen_paths.add(path)
                all_paths.append(path)
                all_labels.append(cls_idx)
                all_folds.append(fold)

    if max_samples is not None:
        all_paths  = all_paths[:max_samples]
        all_labels = all_labels[:max_samples]
        all_folds  = all_folds[:max_samples]

    print(f"  Total files: {len(all_paths)}")

    # ------------------------------------------------------------------ #
    #  Initialize AST extractor                                            #
    # ------------------------------------------------------------------ #
    print("Initializing AST feature extractor ...")
    extractor = ASTFeatureExtractor(device=device, verbose=True)

    # ------------------------------------------------------------------ #
    #  Extract embeddings                                                  #
    # ------------------------------------------------------------------ #
    print("Extracting embeddings ...")
    all_embeddings = []
    failed = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(all_paths), batch_size),
                      desc=f"Embedding {dataset_name}"):
            batch_paths = all_paths[i:i + batch_size]
            waveforms = []
            import librosa
            for p in batch_paths:
                try:
                    wav, _ = librosa.load(p, sr=16000)
                    waveforms.append(wav)
                except Exception as e:
                    print(f"\nWarning: failed to process {p}: {e}")
                    failed += 1
                    # Minimal 1-second silence pad for fallback
                    waveforms.append(np.zeros(16000, dtype=np.float32))

            try:
                emb = extractor.extract_features(waveforms, batch_size=len(waveforms))
                all_embeddings.append(emb)
            except Exception as e:
                print(f"\nError on batch {i}: {e}")
                raise

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    if failed:
        print(f"  WARNING: {failed} files failed to process (zero-padded).")

    # ------------------------------------------------------------------ #
    #  Extract patch (frequency-band) embeddings                         #
    # ------------------------------------------------------------------ #
    print("Extracting frequency-band patch embeddings ...")
    all_patch_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_paths), batch_size),
                      desc=f"Patches {dataset_name}"):
            batch_paths = all_paths[i:i + batch_size]
            waveforms_p = []
            for p in batch_paths:
                try:
                    wav, _ = librosa.load(p, sr=16000)
                    waveforms_p.append(wav)
                except Exception:
                    waveforms_p.append(np.zeros(16000, dtype=np.float32))
            try:
                patches = extractor.extract_patch_features(
                    waveforms_p, batch_size=len(waveforms_p))
                all_patch_embeddings.append(patches)
            except Exception as e:
                print(f"\nError on patch batch {i}: {e}")
                raise
    all_patch_embeddings = torch.cat(all_patch_embeddings, dim=0).numpy()

    # ------------------------------------------------------------------ #
    #  Save cache                                                          #
    # ------------------------------------------------------------------ #
    cache_data = {
        "embeddings":       all_embeddings,
        "patch_embeddings": all_patch_embeddings,
        "file_paths":       all_paths,
        "labels":           np.array(all_labels),
        "folds":            np.array(all_folds),
        "embedding_dim":    int(all_embeddings.shape[1]),
        "num_samples":      len(all_paths),
        "class_names":      dataset.class_names,
        "class_to_idx":     dataset.class_to_idx,
        "dataset_name":     dataset_name,
    }

    print(f"Saving cache to {cache_file} ...")
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)

    size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    print(f"Done. Cache size: {size_mb:.2f} MB")
    print(f"Embedding shape: {all_embeddings.shape}")
    return cache_data


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build AST embedding cache for a dataset."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["esc50", "fsd", "urbansound", "voxceleb", "birdclef"],
        help="Dataset to cache"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache exists")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit total samples (for smoke tests)")
    args = parser.parse_args()

    build_cache(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        device=args.device,
        force_recompute=args.force,
        max_samples=args.max_samples,
    )
