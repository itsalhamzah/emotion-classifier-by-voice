"""
Verify all required packages are installed correctly
"""


def check_installation():
    print("=" * 60)
    print("CHECKING INSTALLATION")
    print("=" * 60)

    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'torchaudio': 'TorchAudio',
        'librosa': 'Librosa',
        'soundfile': 'SoundFile',
        'sounddevice': 'SoundDevice',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM',
        'numba': 'Numba',
    }

    all_installed = True

    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name:15s} {version}")
        except ImportError:
            print(f"❌ {name:15s} NOT INSTALLED")
            all_installed = False

    print("=" * 60)

    # Check GPU
    try:
        import torch
        print(f"\nGPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
    except:
        pass

    print("=" * 60)

    if all_installed:
        print("✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
    else:
        print("❌ SOME PACKAGES ARE MISSING")
        print("Run: pip install -r requirements.txt")

    print("=" * 60)


if __name__ == "__main__":
    check_installation()