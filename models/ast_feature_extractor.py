"""
AST (Audio Spectrogram Transformer) Feature Extractor.
Frozen backbone for few-shot learning - extracts embeddings without fine-tuning.
"""

import torch
import torch.nn as nn
import sys
import os
from typing import Union, List
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *

# Import AST model from the existing AST repository
AST_SRC_PATH = os.path.join(PROJECT_ROOT, 'ast', 'src')
if AST_SRC_PATH not in sys.path:
    sys.path.insert(0, AST_SRC_PATH)

try:
    # Try importing the models module directly
    import models.ast_models as ast_models
except ImportError as e1:
    try:
        # Alternative: import from MODELS directory
        models_path = os.path.join(PROJECT_ROOT, 'ast', 'src', 'models')
        sys.path.insert(0, models_path)
        import ast_models
    except ImportError as e2:
        print(f"Warning: Could not import original AST models: {e1}, {e2}")
        print("Will rely on HuggingFace transformers library only")
        ast_models = None


class ASTFeatureExtractor(nn.Module):
    """
    AST feature extractor wrapper.
    Loads pretrained AST model and extracts embeddings from the pen ultimate layer.
    """
    
    def __init__(self, 
                 pretrained_path: str = AST_PRETRAINED_PATH,
                 device: str = None,
                 verbose: bool = True):
        """
        Initialize AST feature extractor.
        
        Args:
            pretrained_path: Path to pretrained model checkpoint
            device: Device to use ('cuda' or 'cpu')
            verbose: Print model information
        """
        super().__init__()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if verbose:
            print(f"Initializing AST Feature Extractor...")
            print(f"  Device: {self.device}")
            print(f"  Pretrained model: {pretrained_path}")
        
        # Check if pretrained model exists
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
        
        # Initialize AST model
        # AudioSet pretrained model uses: fstride=10, tstride=10, input_tdim=1024
        # For ESC-50 (5s audio): we have ~500 frames
        try:
            # Try loading using HuggingFace format (since we downloaded from HF)
            from transformers import ASTForAudioClassification
            
            if verbose:
                print("  Loading model from HuggingFace format...")
            
            # Load the model
            state_dict = torch.load(pretrained_path, map_location=self.device)
            
            # Create AST model using transformers
            self.model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593",
                cache_dir=os.path.dirname(pretrained_path)
            )
            
            # Load our saved weights
            self.model.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            if verbose:
                print(f"  HuggingFace loading failed: {e}")
                print("  Trying original AST model format...")
            
            # Fallback: try original AST model format
            self.model = ast_models.ASTModel(
                label_dim=527,  # AudioSet has 527 classes
                fstride=10,
                tstride=10,
                input_fdim=128,
                input_tdim=ESC50_TDIM,  # ~500 frames for 5s audio
                imagenet_pretrain=False,
                audioset_pretrain=False,
                verbose=verbose
            )
            
            # Load pretrained weights
            state_dict = torch.load(pretrained_path, map_location=self.device)
            
            # Handle DataParallel wrapper
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict, strict=False)
            else:
                # Create DataParallel wrapper first
                self.model = nn.DataParallel(self.model)
                self.model.load_state_dict(state_dict, strict=False)
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Set to evaluation mode and freeze parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        if verbose:
            print(f"  ✓ Model loaded successfully")
            print(f"  Embedding dimension: {AST_EMBEDDING_DIM}")
            print(f"  Parameters frozen: True")
    
    @torch.no_grad()
    def extract_features(self, 
                        waveforms: Union[torch.Tensor, np.ndarray, list],
                        batch_size: int = BATCH_SIZE) -> torch.Tensor:
        """
        Extract features securely from raw audio waveforms via the native HuggingFace pipeline.
        
        Args:
            waveforms: Raw audio arrays of shape (N, samples) or list of arrays
            batch_size: Batch size for processing
        
        Returns:
            Features of shape (N, embedding_dim)
        """
        if not hasattr(self, 'processor'):
            from transformers import AutoFeatureExtractor
            # We enforce exactly the same pre-processor the backbone understands
            self.processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        if isinstance(waveforms, torch.Tensor):
            waveforms = waveforms.cpu().numpy()
            
        if isinstance(waveforms, np.ndarray) and waveforms.ndim == 1:
            waveforms = [waveforms]
            
        all_features = []
        num_samples = len(waveforms)
        
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        for i in range(0, num_samples, batch_size):
            batch_waveforms = waveforms[i:i+batch_size]
            
            # The HF Processor natively constructs the exact filterbanks and handles padding properly
            inputs = self.processor(batch_waveforms, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            try:
                if hasattr(base_model, 'audio_spectrogram_transformer'):
                    outputs = base_model.audio_spectrogram_transformer(**inputs)
                    features = outputs.last_hidden_state[:, 0]  # CLS token exactly matches
                else:
                    raise ValueError("AST Structure corrupted or unavailable via AutoModel")
                    
                all_features.append(features.cpu())
            except Exception as e:
                raise RuntimeError(f"HF extraction failed: {str(e)}")
                
        return torch.cat(all_features, dim=0)
    

    @torch.no_grad()
    def extract_patch_features(self,
                               waveforms,
                               batch_size: int = BATCH_SIZE) -> torch.Tensor:
        """
        Extract frequency-band aggregated patch tokens from AST.

        AST produces 1214 tokens: [CLS, distillation, patch_0, ..., patch_1211].
        The 1212 patch tokens are arranged as 101 time x 12 freq x 768 dims.
        We mean-pool over the time axis to get (12, 768) per sample.

        Returns:
            Tensor of shape (N, 12, 768)
        """
        N_TIME = 101  # AST time patches (for 10s audio with 128-bin mel)
        N_FREQ = 12   # AST frequency patches

        if not hasattr(self, 'processor'):
            from transformers import AutoFeatureExtractor
            self.processor = AutoFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593")

        if isinstance(waveforms, torch.Tensor):
            waveforms = waveforms.cpu().numpy()
        if isinstance(waveforms, np.ndarray) and waveforms.ndim == 1:
            waveforms = [waveforms]

        all_patches = []
        num_samples = len(waveforms)
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        for i in range(0, num_samples, batch_size):
            batch_waveforms = waveforms[i:i + batch_size]
            inputs = self.processor(batch_waveforms, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                if hasattr(base_model, 'audio_spectrogram_transformer'):
                    outputs = base_model.audio_spectrogram_transformer(**inputs)
                    # Skip CLS (0) and distillation (1) tokens -> (B, 1212, D)
                    patch_tokens = outputs.last_hidden_state[:, 2:]
                    B, P, D = patch_tokens.shape
                    # Reshape to (B, 101, 12, D) and pool over time -> (B, 12, D)
                    freq_bands = patch_tokens.view(B, N_TIME, N_FREQ, D).mean(dim=1)
                    all_patches.append(freq_bands.cpu())
                else:
                    raise ValueError("AST Structure corrupted or unavailable via AutoModel")
            except Exception as e:
                raise RuntimeError(f"HF patch extraction failed: {str(e)}")

        return torch.cat(all_patches, dim=0)  # (N, 12, D)

    def forward(self, x):
        """Forward pass (for nn.Module compatibility)"""
        return self.extract_features(x)


def test_ast_feature_extractor():
    """Test AST feature extractor."""
    print("=" * 80)
    print("Testing AST Feature Extractor")
    print("=" * 80)
    
    # Initialize extractor
    try:
        extractor = ASTFeatureExtractor(verbose=True)
    except Exception as e:
        print(f"\n✗ Failed to initialize feature extractor: {e}")
        return
    
    print("\n" + "=" * 80)
    print("Test 1: Extract features from random spectrogram")
    print("=" * 80)
    
    # Create random spectrogram
    batch_size = 4
    time_frames = ESC50_TDIM  # ~500 frames
    n_mels = N_MELS  # 128
    
    random_spec = torch.randn(batch_size, time_frames, n_mels)
    print(f"\nInput spectrogram shape: {random_spec.shape}")
    
    try:
        features = extractor.extract_features(random_spec)
        print(f"Output features shape: {features.shape}")
        print(f"Expected shape: ({batch_size}, {AST_EMBEDDING_DIM})")
        
        if features.shape == (batch_size, AST_EMBEDDING_DIM):
            print("✓ Feature extraction successful!")
        else:
            print(f"✗ Unexpected feature shape!")
    
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with real audio
    print("\n" + "=" * 80)
    print("Test 2: Extract features from real ESC-50 audio")
    print("=" * 80)
    
    try:
        from data.audio_preprocessing import preprocess_audio
        import glob
        
        audio_files = glob.glob(os.path.join(ESC50_AUDIO_DIR, "*.wav"))[:2]
        
        if audio_files:
            spectrograms = []
            for audio_path in audio_files:
                _, spec = preprocess_audio(audio_path)
                spectrograms.append(spec)
            
            spectrograms = torch.stack(spectrograms)
            print(f"Loaded {len(audio_files)} audio files")
            print(f"Spectrogram batch shape: {spectrograms.shape}")
            
            features = extractor.extract_features(spectrograms)
            print(f"Extracted features shape: {features.shape}")
            print(f"Feature statistics:")
            print(f"  Mean: {features.mean().item():.4f}")
            print(f"  Std: {features.std().item():.4f}")
            print(f"  Min: {features.min().item():.4f}")
            print(f"  Max: {features.max().item():.4f}")
            
            print("\n✓ Real audio test successful!")
        else:
            print("No audio files found for testing")
    
    except Exception as e:
        print(f"✗ Real audio test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Feature extractor tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_ast_feature_extractor()
