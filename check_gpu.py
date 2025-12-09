import torch

print("=" * 60)
print("GPU AVAILABILITY CHECK")
print("=" * 60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device Count: {torch.cuda.device_count()}")
    print(f"Current GPU Device: {torch.cuda.current_device()}")
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✅ GPU is ready to use!")
else:
    print("\n❌ CUDA is not available. Training will use CPU.")
    print("\nTo enable GPU:")
    print("1. Make sure you have an NVIDIA GPU")
    print("2. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
    print("3. Reinstall PyTorch with CUDA support:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")