"""
Audio preprocessing utilities for ESC-50 dataset.
Handles audio loading, resampling, fixed-length processing, and log-mel spectrogram computation.
"""

import numpy as np
import librosa
import torch
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *


def load_audio(audio_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate (default: 16kHz)
    
    Returns:
        Audio waveform as numpy array
    """
    try:
        # Load audio with librosa (automatically resamples if needed)
        waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
        return waveform
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {audio_path}: {str(e)}")


def fix_length(waveform: np.ndarray, 
               target_length: float = AUDIO_LENGTH,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Fix audio length to target duration by padding or trimming.
    
    Args:
        waveform: Input audio waveform
        target_length: Target length in seconds
        sr: Sample rate
    
    Returns:
        Fixed-length waveform
    """
    target_samples = int(target_length * sr)
    current_samples = len(waveform)
    
    if current_samples < target_samples:
        # Pad with zeros
        padding = target_samples - current_samples
        waveform = np.pad(waveform, (0, padding), mode='constant')
    elif current_samples > target_samples:
        # Trim from center
        start = (current_samples - target_samples) // 2
        waveform = waveform[start:start + target_samples]
    
    return waveform


def compute_log_mel_spectrogram(waveform: np.ndarray,
                                sr: int = SAMPLE_RATE,
                                n_mels: int = N_MELS,
                                n_fft: int = N_FFT,
                                hop_length: int = HOP_LENGTH,
                                fmin: int = FMIN,
                                fmax: int = FMAX) -> np.ndarray:
    """
    Compute log-mel spectrogram from audio waveform.
    
    Args:
        waveform: Input audio waveform
        sr: Sample rate
        n_mels: Number of mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency
    
    Returns:
        Log-mel spectrogram of shape (n_mels, time_frames)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0  # Power spectrogram
    )
    
    # Convert to log scale (add small epsilon to avoid log(0))
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def normalize_spectrogram(log_mel_spec: np.ndarray,
                         mean: float = AUDIOSET_MEAN,
                         std: float = AUDIOSET_STD) -> np.ndarray:
    """
    Normalize log-mel spectrogram using AudioSet statistics.
    
    Args:
        log_mel_spec: Log-mel spectrogram
        mean: Target mean (AudioSet mean)
        std: Target std (AudioSet std)
    
    Returns:
        Normalized spectrogram
    """
    # Apply normalization: (x + mean_offset) / (std * 2)
    # This follows AST's normalization: input_spec = (input_spec + 4.26) / (4.57 * 2)
    normalized = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
    normalized = normalized * std + mean
    
    return normalized


def preprocess_audio(audio_path: str,
                    sr: int = SAMPLE_RATE,
                    target_length: float = AUDIO_LENGTH,
                    return_tensor: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a single audio file.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        target_length: Target audio length in seconds
        return_tensor: If True, return PyTorch tensor; otherwise numpy array
    
    Returns:
        Tuple of (waveform, log_mel_spectrogram)
        - waveform: shape (num_samples,)
        - log_mel_spec: shape (n_mels, time_frames) or (time_frames, n_mels) for AST
    """
    # Load and resample
    waveform = load_audio(audio_path, sr=sr)
    
    # Fix length
    waveform = fix_length(waveform, target_length=target_length, sr=sr)
    
    # Compute log-mel spectrogram
    log_mel_spec = compute_log_mel_spectrogram(waveform, sr=sr)
    
    # Normalize
    log_mel_spec = normalize_spectrogram(log_mel_spec)
    
    # AST expects input shape: (batch, time_frames, n_mels)
    # Transpose from (n_mels, time_frames) to (time_frames, n_mels)
    log_mel_spec = log_mel_spec.T
    
    if return_tensor:
        waveform = torch.FloatTensor(waveform)
        log_mel_spec = torch.FloatTensor(log_mel_spec)
    
    return waveform, log_mel_spec


def batch_preprocess_audio(audio_paths: list,
                          sr: int = SAMPLE_RATE,
                          target_length: float = AUDIO_LENGTH) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess a batch of audio files.
    
    Args:
        audio_paths: List of audio file paths
        sr: Sample rate
        target_length: Target audio length in seconds
    
    Returns:
        Tuple of (waveforms, log_mel_spectrograms)
        - waveforms: shape (batch, num_samples)
        - log_mel_specs: shape (batch, time_frames, n_mels)
    """
    waveforms = []
    log_mel_specs = []
    
    for audio_path in audio_paths:
        waveform, log_mel_spec = preprocess_audio(
            audio_path, sr=sr, target_length=target_length, return_tensor=True
        )
        waveforms.append(waveform)
        log_mel_specs.append(log_mel_spec)
    
    # Stack into batches
    waveforms = torch.stack(waveforms, dim=0)
    log_mel_specs = torch.stack(log_mel_specs, dim=0)
    
    return waveforms, log_mel_specs


if __name__ == "__main__":
    # Test preprocessing on a sample ESC-50 file
    import glob
    
    sample_files = glob.glob(os.path.join(ESC50_AUDIO_DIR, "*.wav"))[:3]
    
    if not sample_files:
        print(f"No audio files found in {ESC50_AUDIO_DIR}")
    else:
        print(f"Testing preprocessing on {len(sample_files)} files...")
        
        for audio_path in sample_files:
            print(f"\nProcessing: {os.path.basename(audio_path)}")
            waveform, log_mel_spec = preprocess_audio(audio_path)
            
            print(f"  Waveform shape: {waveform.shape}")
            print(f"  Log-mel spec shape: {log_mel_spec.shape}")
            print(f"  Waveform range: [{waveform.min():.3f}, {waveform.max():.3f}]")
            print(f"  Spec range: [{log_mel_spec.min():.3f}, {log_mel_spec.max():.3f}]")
        
        # Test batch processing
        print(f"\nTesting batch processing...")
        waveforms, log_mel_specs = batch_preprocess_audio(sample_files)
        print(f"  Batch waveforms shape: {waveforms.shape}")
        print(f"  Batch log-mel specs shape: {log_mel_specs.shape}")
        
        print("\n✓ Preprocessing test completed successfully!")
