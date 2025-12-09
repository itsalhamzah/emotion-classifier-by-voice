"""
Dataset loader for emotion classification
"""
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import *
from preprocessing import process_audio_file


class EmotionDataset(Dataset):
    """
    Custom Dataset for emotion classification from audio files
    """

    def __init__(self, file_paths, labels, augment=False):
        """
        Args:
            file_paths: List of paths to audio files
            labels: List of emotion labels (integers)
            augment: Whether to apply data augmentation
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and process audio file

        Returns:
            mel_spec: Preprocessed mel-spectrogram tensor (1, n_mels, time)
            label: Emotion label (integer)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Process audio file
        mel_spec = process_audio_file(file_path, augment=self.augment)

        # Handle loading errors
        if mel_spec is None:
            # Return a zero tensor if loading fails
            mel_spec = torch.zeros((1, N_MELS, MAX_LENGTH))

        return mel_spec, label


def load_dataset_paths(dataset_path):
    """
    Load all audio file paths and their labels from dataset directory

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        file_paths: List of file paths
        labels: List of labels (integers)
        emotion_to_idx: Dictionary mapping emotion names to indices
    """
    file_paths = []
    labels = []

    # Create emotion to index mapping
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

    print("Loading dataset paths...")

    # Iterate through emotion folders
    for emotion in EMOTIONS:
        emotion_path = Path(dataset_path) / emotion

        if not emotion_path.exists():
            print(f"Warning: Emotion folder '{emotion}' not found at {emotion_path}")
            continue

        # Get all wav files in emotion folder (UPDATED FOR .WAV FILES)
        audio_files = list(emotion_path.glob("*.wav"))

        # If no .wav files, try .mp3
        if len(audio_files) == 0:
            audio_files = list(emotion_path.glob("*.mp3"))

        print(f"Found {len(audio_files)} files for emotion '{emotion}'")

        # Add to lists
        for audio_file in audio_files:
            file_paths.append(str(audio_file))
            labels.append(emotion_to_idx[emotion])

    print(f"\nTotal files loaded: {len(file_paths)}")
    print(f"Total labels: {len(labels)}")
    print(f"Emotion distribution: {dict(zip(EMOTIONS, [labels.count(i) for i in range(NUM_CLASSES)]))}")

    return file_paths, labels, emotion_to_idx


def create_data_loaders(dataset_path, batch_size=BATCH_SIZE):
    """
    Create train, validation, and test data loaders
    """
    # Load all file paths and labels
    file_paths, labels, emotion_to_idx = load_dataset_paths(dataset_path)

    if len(file_paths) == 0:
        raise ValueError("No audio files found! Please check your DATASET_PATH in config.py")

    # Split data
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels,
        test_size=TEST_SPLIT,
        random_state=42,
        stratify=labels
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT),
        random_state=42,
        stratify=train_val_labels
    )

    print(f"\nDataset splits:")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")

    # Create datasets
    train_dataset = EmotionDataset(train_paths, train_labels, augment=True)
    val_dataset = EmotionDataset(val_paths, val_labels, augment=False)
    test_dataset = EmotionDataset(test_paths, test_labels, augment=False)

    # Optimize num_workers for GPU
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, test_loader, emotion_to_idx
    # Load all file paths and labels
    file_paths, labels, emotion_to_idx = load_dataset_paths(dataset_path)

    if len(file_paths) == 0:
        raise ValueError("No audio files found! Please check your DATASET_PATH in config.py")

    # Split data: train/val/test
    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels,
        test_size=TEST_SPLIT,
        random_state=42,
        stratify=labels
    )

    # Second split: separate train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT),
        random_state=42,
        stratify=train_val_labels
    )

    print(f"\nDataset splits:")
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")

    # Create datasets
    train_dataset = EmotionDataset(train_paths, train_labels, augment=True)
    val_dataset = EmotionDataset(val_paths, val_labels, augment=False)
    test_dataset = EmotionDataset(test_paths, test_labels, augment=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader, test_loader, emotion_to_idx