"""
Audio preprocessing utilities for emotion classification
"""
import librosa
import numpy as np
import torch
from config import *


def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Load audio file and ensure consistent length

    Args:
        file_path: Path to audio file
        sr: Sample rate
        duration: Target duration in seconds

    Returns:
        audio: Audio time series
    """
    try:
        # Load audio file (works with both .wav and .mp3)
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)

        # Ensure consistent length
        target_length = int(sr * duration)
        if len(audio) < target_length:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Truncate if too long
            audio = audio[:target_length]

        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_mel_spectrogram(audio, sr=SAMPLE_RATE, n_mels=N_MELS,
                            n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Extract mel-spectrogram features from audio

    Args:
        audio: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Number of samples between frames

    Returns:
        mel_spec: Mel-spectrogram (n_mels, time)
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def normalize_spectrogram(mel_spec):
    """
    Normalize mel-spectrogram to [0, 1] range

    Args:
        mel_spec: Mel-spectrogram

    Returns:
        normalized_spec: Normalized spectrogram
    """
    # Min-max normalization
    min_val = np.min(mel_spec)
    max_val = np.max(mel_spec)

    if max_val - min_val > 0:
        normalized_spec = (mel_spec - min_val) / (max_val - min_val)
    else:
        normalized_spec = mel_spec

    return normalized_spec


def augment_audio(audio, sr=SAMPLE_RATE):
    """
    Apply random augmentation to audio

    Args:
        audio: Audio time series
        sr: Sample rate

    Returns:
        augmented_audio: Augmented audio
    """
    augmented = audio.copy()

    # Random time stretching
    if np.random.random() > 0.5:
        rate = np.random.uniform(TIME_STRETCH_RATE[0], TIME_STRETCH_RATE[1])
        augmented = librosa.effects.time_stretch(augmented, rate=rate)

        # Ensure consistent length
        target_length = len(audio)
        if len(augmented) < target_length:
            augmented = np.pad(augmented, (0, target_length - len(augmented)), mode='constant')
        else:
            augmented = augmented[:target_length]

    # Random pitch shifting
    if np.random.random() > 0.5:
        n_steps = np.random.uniform(PITCH_SHIFT_STEPS[0], PITCH_SHIFT_STEPS[1])
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    # Add random noise
    if np.random.random() > 0.5:
        noise = np.random.randn(len(augmented))
        augmented = augmented + NOISE_FACTOR * noise

    return augmented


def process_audio_file(file_path, augment=False):
    """
    Complete preprocessing pipeline for audio file

    Args:
        file_path: Path to audio file
        augment: Whether to apply augmentation

    Returns:
        tensor: Preprocessed mel-spectrogram as tensor (1, n_mels, time)
    """
    # Load audio
    audio = load_audio(file_path)
    if audio is None:
        return None

    # Apply augmentation if requested
    if augment and USE_AUGMENTATION:
        audio = augment_audio(audio)

    # Extract mel-spectrogram
    mel_spec = extract_mel_spectrogram(audio)

    # Normalize
    mel_spec = normalize_spectrogram(mel_spec)

    # Ensure consistent time dimension
    if mel_spec.shape[1] < MAX_LENGTH:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, MAX_LENGTH - mel_spec.shape[1])), mode='constant')
    else:
        mel_spec = mel_spec[:, :MAX_LENGTH]

    # Convert to tensor and add channel dimension
    tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, time)

    return tensor


def process_microphone_audio(audio_data, sr=MIC_SAMPLE_RATE):
    """
    Process audio from microphone input

    Args:
        audio_data: Raw audio data from microphone
        sr: Sample rate

    Returns:
        tensor: Preprocessed mel-spectrogram as tensor
    """
    # Ensure correct length
    target_length = int(sr * DURATION)
    if len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:target_length]

    # Extract mel-spectrogram
    mel_spec = extract_mel_spectrogram(audio_data, sr=sr)

    # Normalize
    mel_spec = normalize_spectrogram(mel_spec)

    # Ensure consistent time dimension
    if mel_spec.shape[1] < MAX_LENGTH:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, MAX_LENGTH - mel_spec.shape[1])), mode='constant')
    else:
        mel_spec = mel_spec[:, :MAX_LENGTH]

    # Convert to tensor
    tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, time)

    return tensor