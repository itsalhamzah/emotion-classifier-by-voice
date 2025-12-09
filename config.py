"""
Configuration file for emotion classification system
"""
import torch

# Dataset Configuration
DATASET_PATH = r"C:\Users\MSI\Desktop\emotion_classification\emotion by voice(dataset)\Voice Emotion Dataset"  # Update this
EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
NUM_CLASSES = len(EMOTIONS)

# Audio Processing Parameters
SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 256
N_FFT = 2048
HOP_LENGTH = 512
MAX_LENGTH = int(SAMPLE_RATE * DURATION / HOP_LENGTH)

# Training Parameters
BATCH_SIZE = 128  # Increase to 64 or 128 if you have good GPU
EPOCHS = 100
LEARNING_RATE = 0.0005
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# GPU Configuration
USE_GPU = True  # Set to False to force CPU
DEVICE = torch.device("cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu")

# Print device information
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Model Configuration
MODEL_PATH = "models/emotion_classifier.pth"

# Microphone Configuration
MIC_DURATION = 4.0
MIC_SAMPLE_RATE = 22050

# Data Augmentation
USE_AUGMENTATION = True
TIME_STRETCH_RATE = [0.8, 1.2]
PITCH_SHIFT_STEPS = [-2, 2]
NOISE_FACTOR = 0.005