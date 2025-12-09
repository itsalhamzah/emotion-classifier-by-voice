from config import DATASET_PATH, EMOTIONS
import os
from pathlib import Path

print("=" * 60)
print("TESTING SETUP")
print("=" * 60)

print(f"\n1. Dataset path: {DATASET_PATH}")
print(f"2. Path exists: {os.path.exists(DATASET_PATH)}")

if os.path.exists(DATASET_PATH):
    print(f"3. Folders found: {os.listdir(DATASET_PATH)}")

    print("\n4. Checking audio files in each emotion folder:")
    for emotion in EMOTIONS:
        emotion_path = Path(DATASET_PATH) / emotion
        if emotion_path.exists():
            wav_files = list(emotion_path.glob("*.wav"))
            mp3_files = list(emotion_path.glob("*.mp3"))
            total = len(wav_files) + len(mp3_files)
            print(f"   {emotion:8s}: {total} files (.wav: {len(wav_files)}, .mp3: {len(mp3_files)})")
        else:
            print(f"   {emotion:8s}: FOLDER NOT FOUND")

    print("\n✅ Setup looks good!" if total > 0 else "\n❌ No audio files found!")
else:
    print("\n❌ Dataset path does not exist!")
    print("Please update DATASET_PATH in config.py")
