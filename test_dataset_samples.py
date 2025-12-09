"""
Test random samples from your dataset
"""
import random
from pathlib import Path
from test_audio_files import AudioFileTester
from config import DATASET_PATH, EMOTIONS


def test_random_samples(samples_per_emotion=5):
    """Test random samples from each emotion"""

    tester = AudioFileTester()

    print("=" * 60)
    print(f"TESTING {samples_per_emotion} RANDOM SAMPLES PER EMOTION")
    print("=" * 60)

    for emotion in EMOTIONS:
        emotion_path = Path(DATASET_PATH) / emotion

        if not emotion_path.exists():
            print(f"\n❌ Emotion folder not found: {emotion}")
            continue

        # Get all audio files
        audio_files = list(emotion_path.glob("*.wav"))

        if len(audio_files) == 0:
            print(f"\n❌ No audio files found for: {emotion}")
            continue

        # Select random samples
        samples = random.sample(audio_files, min(samples_per_emotion, len(audio_files)))

        print(f"\n📁 Testing {emotion.upper()} samples:")
        print("-" * 60)

        correct = 0
        for i, audio_file in enumerate(samples, 1):
            predicted_emotion, confidence, _ = tester.predict_file(str(audio_file))

            # Check if correct
            is_correct = "✅" if predicted_emotion == emotion else "❌"

            if predicted_emotion == emotion:
                correct += 1

            print(f"{i}. {is_correct} Predicted: {predicted_emotion:8s} ({confidence * 100:5.2f}%) | Actual: {emotion}")

        accuracy = correct / len(samples) * 100
        print(f"Accuracy: {correct}/{len(samples)} ({accuracy:.1f}%)")

    print("\n" + "=" * 60)


def test_specific_emotion_files(emotion, num_files=10):
    """Test specific number of files from one emotion"""

    tester = AudioFileTester()

    emotion_path = Path(DATASET_PATH) / emotion
    audio_files = list(emotion_path.glob("*.wav"))[:num_files]

    print(f"\nTesting {num_files} files from {emotion.upper()}:\n")

    tester.test_multiple_files([str(f) for f in audio_files])


if __name__ == "__main__":
    # Test 5 random samples from each emotion
    test_random_samples(samples_per_emotion=5)

    # Or test specific emotion
    # test_specific_emotion_files("happy", num_files=10)