"""
Cache ESC-50 embeddings using AST feature extractor.
This script extracts and saves embeddings for all ESC-50 audio files to avoid recomputing them.
"""

import torch
import numpy as np
import os
import sys
from tqdm import tqdm
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from data.esc50_dataloader import ESC50Dataset
from data.audio_preprocessing import preprocess_audio
from models.ast_feature_extractor import ASTFeatureExtractor


def cache_esc50_embeddings(output_dir: str = CACHE_DIR,
                          batch_size: int = 32,
                          device: str = 'cuda',
                          force_recompute: bool = False):
    """
    Extract and cache AST embeddings for all ESC-50 audio files.
    
    Args:
        output_dir: Directory to save cached embeddings
        batch_size: Batch size for feature extraction
        device: Device to use ('cuda' or 'cpu')
        force_recompute: If True, recompute even if cache exists
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, 'esc50_embeddings.pkl')
    
    # Check if cache already exists
    if os.path.exists(cache_file) and not force_recompute:
        print(f"Cache file already exists at {cache_file}")
        response = input("Recompute embeddings? (y/n): ")
        if response.lower() != 'y':
            print("Loading existing cache...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"Loaded {len(cache_data['embeddings'])} cached embeddings")
            return cache_data
    
    print("=" * 80)
    print("Caching ESC-50 Embeddings")
    print("=" * 80)
    
    # Load dataset
    print("\n1. Loading ESC-50 dataset...")
    dataset = ESC50Dataset()
    
    # Initialize feature extractor
    print("\n2. Initializing AST feature extractor...")
    extractor = ASTFeatureExtractor(device=device, verbose=True)
    
    # Collect all audio files
    print("\n3. Collecting all audio files...")
    all_files = []
    all_labels = []
    all_folds = []
    
    for fold in range(1, 6):
        fold_files = dataset.get_fold_data(fold, exclude_fold=False)
        for audio_path, class_idx, class_name in fold_files:
            all_files.append(audio_path)
            all_labels.append(class_idx)
            all_folds.append(fold)
    
    print(f"Total files to process: {len(all_files)}")
    
    # Extract features in batches
    print("\n4. Extracting features...")
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_files), batch_size), desc="Extracting embeddings"):
            batch_files = all_files[i:i+batch_size]
            
            # Preprocess batch
            spectrograms = []
            for audio_path in batch_files:
                try:
                    _, spec = preprocess_audio(audio_path, return_tensor=True)
                    spectrograms.append(spec)
                except Exception as e:
                    print(f"\nWarning: Failed to process {audio_path}: {e}")
                    # Use zero spectrogram as fallback
                    spectrograms.append(torch.zeros(ESC50_TDIM, N_MELS))
            
            spectrograms = torch.stack(spectrograms)
            
            # Extract features
            try:
                embeddings = extractor.extract_features(spectrograms, batch_size=len(batch_files))
                all_embeddings.append(embeddings)
            except Exception as e:
                print(f"\nError extracting features for batch {i}: {e}")
                raise
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    # Extract frequency-band patch embeddings (12 bands x 768)
    print("\n4b. Extracting patch (frequency-band) features...")
    all_patch_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_files), batch_size), desc="Extracting patches"):
            batch_files = all_files[i:i+batch_size]
            spectrograms_b = []
            for audio_path in batch_files:
                try:
                    _, spec = preprocess_audio(audio_path, return_tensor=True)
                    spectrograms_b.append(spec)
                except Exception:
                    spectrograms_b.append(torch.zeros(ESC50_TDIM, N_MELS))
            spectrograms_b = torch.stack(spectrograms_b)
            patches = extractor.extract_patch_features(
                spectrograms_b, batch_size=len(batch_files))
            all_patch_embeddings.append(patches)
    all_patch_embeddings = torch.cat(all_patch_embeddings, dim=0).numpy()
    
    print(f"\n5. Embedding statistics:")
    print(f"  Shape: {all_embeddings.shape}")
    print(f"  Mean: {all_embeddings.mean():.4f}")
    print(f"  Std: {all_embeddings.std():.4f}")
    print(f"  Min: {all_embeddings.min():.4f}")
    print(f"  Max: {all_embeddings.max():.4f}")
    
    # Create cache dictionary
    cache_data = {
        'embeddings': all_embeddings,
        'patch_embeddings': all_patch_embeddings,
        'file_paths': all_files,
        'labels': np.array(all_labels),
        'folds': np.array(all_folds),
        'embedding_dim': all_embeddings.shape[1],
        'num_samples': len(all_files),
        'class_names': dataset.class_names,
        'class_to_idx': dataset.class_to_idx,
    }
    
    # Save cache
    print(f"\n6. Saving cache to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f, protocol=4)
    
    # Verify cache size
    cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    print(f"  Cache file size: {cache_size_mb:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✓ Embedding caching completed successfully!")
    print("=" * 80)
    
    return cache_data


def load_cached_embeddings(cache_dir: str = CACHE_DIR):
    """
    Load cached ESC-50 embeddings.
    
    Args:
        cache_dir: Directory containing cached embeddings
    
    Returns:
        Dictionary containing embeddings and metadata
    """
    dir_name = os.path.basename(os.path.normpath(cache_dir))
    cache_file = os.path.join(cache_dir, f'{dir_name}.pkl')
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found at {cache_file}. Run cache_esc50_embeddings() first.")
    
    print(f"Loading cached embeddings from {cache_file}...")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    print(f"Loaded cache:")
    print(f"  Number of samples: {cache_data['num_samples']}")
    print(f"  Embedding dimension: {cache_data['embedding_dim']}")
    print(f"  Number of classes: {len(cache_data['class_names'])}")
    
    return cache_data


def get_episode_embeddings(episode: dict, cache_data: dict):
    """
    Get embeddings for a sampled episode from cache.
    
    Args:
        episode: Episode dictionary from EpisodeSampler
        cache_data: Cached embeddings dictionary
    
    Returns:
        Tuple of (support_embeddings, support_labels, query_embeddings, query_labels)
    """
    # Create path to index mapping
    path_to_idx = {path: idx for idx, path in enumerate(cache_data['file_paths'])}
    
    # Get support embeddings
    support_embeddings = []
    support_labels = []
    for path, label in episode['support_paths']:
        if path in path_to_idx:
            idx = path_to_idx[path]
            support_embeddings.append(cache_data['embeddings'][idx])
            support_labels.append(label)
        else:
            raise ValueError(f"Path {path} not found in cache")
    
    # Get query embeddings
    query_embeddings = []
    query_labels = []
    for path, label in episode['query_paths']:
        if path in path_to_idx:
            idx = path_to_idx[path]
            query_embeddings.append(cache_data['embeddings'][idx])
            query_labels.append(label)
        else:
            raise ValueError(f"Path {path} not found in cache")
    
    support_embeddings = np.array(support_embeddings)
    support_labels = np.array(support_labels)
    query_embeddings = np.array(query_embeddings)
    query_labels = np.array(query_labels)
    
    return support_embeddings, support_labels, query_embeddings, query_labels


def get_episode_patch_embeddings(episode: dict, cache_data: dict):
    """
    Get frequency-band patch embeddings for a sampled episode.

    Returns (support_patches, query_patches) each of shape (N, 12, 768),
    or (None, None) if patch_embeddings is not in cache_data.
    """
    if 'patch_embeddings' not in cache_data:
        return None, None

    path_to_idx = {p: i for i, p in enumerate(cache_data['file_paths'])}

    sup_patches = np.array(
        [cache_data['patch_embeddings'][path_to_idx[p]] for p, _ in episode['support_paths']])
    qry_patches = np.array(
        [cache_data['patch_embeddings'][path_to_idx[p]] for p, _ in episode['query_paths']])

    return sup_patches, qry_patches


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache ESC-50 embeddings")
    parser.add_argument('--output_dir', type=str, default=CACHE_DIR, help='Output directory for cache')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for extraction')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--force', action='store_true', help='Force recompute even if cache exists')
    
    args = parser.parse_args()
    
    # Run caching
    cache_data = cache_esc50_embeddings(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        force_recompute=args.force
    )
    
    # Test loading
    print("\n" + "=" * 80)
    print("Testing cache loading...")
    print("=" * 80)
    
    loaded_data = load_cached_embeddings(args.output_dir)
    
    # Verify they match
    if np.allclose(cache_data['embeddings'], loaded_data['embeddings']):
        print("✓ Cache save/load verification passed!")
    else:
        print("✗ Cache save/load verification failed!")
