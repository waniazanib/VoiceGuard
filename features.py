"""
Audio feature engineering and preprocessing pipelines for Constant-Q Transform (CQT) and MFCC.
"""

import numpy as np
import librosa
import torch
import config


def load_audio(path: str) -> np.ndarray:
    """
    Loads any audio file, resamples to 16kHz, converts to mono, trims silence,
    and pads or truncates to exactly 4 seconds.
    
    Args:
        path: Path to the input audio file.
        
    Returns:
        waveform: 1D float array of shape (64000,)
    """
    # Load audio (represented as single channel at target sample rate)
    y, sr = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    
    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) == 0:
        y_trimmed = y  # Fallback if trimming empties the clip
        
    target_length = config.SAMPLE_RATE * config.CLIP_DURATION  # 64000 samples
    
    if len(y_trimmed) < target_length:
        # Pad with zeros
        pad_width = target_length - len(y_trimmed)
        y_processed = np.pad(y_trimmed, (0, pad_width), mode='constant')
    else:
        # Truncate
        y_processed = y_trimmed[:target_length]
        
    return y_processed


def extract_cqt(waveform: np.ndarray) -> np.ndarray:
    """
    Computes the Constant-Q Transform (CQT) with 84 bins and normalizes the log-magnitude
    spectrogram into the [-1, 1] range.
    
    Args:
        waveform: 1D float array of audio samples (typically shape 64000)
        
    Returns:
        normalized_cqt: 2D array of shape (CQT_BINS, T)
    """
    # Computes a Constant-Q Transform
    cqt_complex = librosa.cqt(
        y=waveform,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
        n_bins=config.CQT_BINS,
        bins_per_octave=12
    )
    
    # Calculate magnitude spectrogram
    cqt_mag = np.abs(cqt_complex)
    
    # Transform to decibel scale
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)
    
    # Normalize linearly into range [-1, 1]
    min_val, max_val = cqt_db.min(), cqt_db.max()
    if max_val - min_val > 1e-8:
        normalized_cqt = 2.0 * (cqt_db - min_val) / (max_val - min_val) - 1.0
    else:
        normalized_cqt = np.zeros_like(cqt_db)
        
    return normalized_cqt


def extract_mfcc(waveform: np.ndarray) -> np.ndarray:
    """
    Computes 40 MFCCs plus their delta and delta-delta coefficients,
    giving a 120 x T representation, and normalizes it to [-1, 1].
    
    Args:
        waveform: 1D float array of audio samples.
        
    Returns:
        normalized_mfcc: 2D array of shape (120, T)
    """
    # Calculate MFCC features
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        hop_length=config.HOP_LENGTH
    )
    
    # Compute first-order and second-order derivatives (deltas)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Concatenate features along the bin axis
    mfcc_feats = np.concatenate([mfcc, delta1, delta2], axis=0)
    
    # Normalize into range [-1, 1]
    min_val, max_val = mfcc_feats.min(), mfcc_feats.max()
    if max_val - min_val > 1e-8:
        normalized_mfcc = 2.0 * (mfcc_feats - min_val) / (max_val - min_val) - 1.0
    else:
        normalized_mfcc = np.zeros_like(mfcc_feats)
        
    return normalized_mfcc


def apply_specaugment(spec: np.ndarray) -> np.ndarray:
    """
    Applies SpecAugment data augmentation on a CQT or MFCC 2D array.
    Applies 2 frequency masks (max width 10 bins) and 2 time masks (max width 20 frames).
    
    Args:
        spec: 2D numpy array of shape (bins, frames)
        
    Returns:
        augmented_spec: 2D numpy array with masks applied.
    """
    augmented = spec.copy()
    n_bins, n_frames = augmented.shape
    
    # 2 Frequency masks (each with width dynamically up to 10 bins)
    for _ in range(2):
        w = np.random.randint(1, 11)
        f_start = np.random.randint(0, max(1, n_bins - w))
        augmented[f_start:f_start+w, :] = 0.0
        
    # 2 Time masks (each with width dynamically up to 20 frames)
    for _ in range(2):
        w = np.random.randint(1, 21)
        t_start = np.random.randint(0, max(1, n_frames - w))
        augmented[:, t_start:t_start+w] = 0.0
        
    return augmented


def preprocess_for_inference(audio_path: str) -> torch.Tensor:
    """
    The full inference preprocessing pipeline:
    load audio -> extract CQT -> normalize to [-1, 1] -> add batch & channel -> torch tensor
    
    Args:
        audio_path: File system path of the audio file to process.
        
    Returns:
        input_tensor: PyTorch tensor of shape (1, 1, CQT_BINS, T)
    """
    waveform = load_audio(audio_path)
    cqt_feat = extract_cqt(waveform)
    
    # Add channel and batch dimensions: [bins, frames] -> [1, 1, bins, frames]
    input_numpy = cqt_feat[np.newaxis, np.newaxis, :, :]
    
    return torch.tensor(input_numpy, dtype=torch.float32)
