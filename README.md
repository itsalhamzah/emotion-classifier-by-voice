✨ Features
🎯 High Accuracy: CNN-based architecture optimized for emotion recognition
🎙️ Real-time Recording: Record directly from the microphone with live waveform visualization
📁 File Upload: Analyze pre-recorded audio files (WAV, MP3)
📊 Confidence Metrics: Visual confidence gauges and probability distributions
🚀 GPU Acceleration: CUDA support for faster training and inference
🎨 Modern GUI: Beautiful Tkinter interface with dark mode

🏗️ Project Structure
emotion_classification/
│
├── config.py                    # Configuration and hyperparameters
├── model.py                     # CNN architecture definition
├── preprocessing.py             # Audio preprocessing utilities
├── dataset.py                   # Dataset loading and data loaders
├── train.py                     # Training script
├── gui_inference.py             # Main GUI application
│
├── check_installation.py        # Verify package installation
├── check_gpu.py                 # Check GPU availability
├── test_setup.py                # Verify dataset setup
│
├── evaluate_model.py            # Detailed model evaluation
├── test_audio_files.py          # Test on audio files
├── test_dataset_samples.py      # Test on dataset samples
│
├── continue_training.py         # Resume training from checkpoint
├── finetune.py                  # Fine-tune with lower learning rate
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file


Notes:
Run these files in order

check_installation.py → Verify environment
check_gpu.py → Confirm GPU setup
test_setup.py → Validate dataset
train.py → Train the model
evaluate_model.py → Check performance
gui_inference.py → Use the application
* download the data set and update the path in config.py


Used dataset: https://www.kaggle.com/datasets/sdeogade/voice-emotion-classification
