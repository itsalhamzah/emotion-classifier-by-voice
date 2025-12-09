"""
Test emotion classifier on multiple audio files
"""
import librosa
import os
from pathlib import Path
from model import EmotionCNN
from config import *
from preprocessing import process_microphone_audio


class AudioFileTester:
    """Test model on audio files"""

    def __init__(self, model_path="models/best_acc_model.pth"):
        """Load the trained model"""
        print(f"Loading model from {model_path}...")
        self.device = DEVICE
        self.model = EmotionCNN(num_classes=NUM_CLASSES).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.emotions = EMOTIONS
        print("✅ Model loaded successfully!\n")

    def predict_file(self, audio_path):
        """Predict emotion from audio file"""
        # Load audio
        audio_data, _ = librosa.load(audio_path, sr=MIC_SAMPLE_RATE, duration=MIC_DURATION)

        # Preprocess
        mel_spec = process_microphone_audio(audio_data, sr=MIC_SAMPLE_RATE)
        mel_spec = mel_spec.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        emotion = self.emotions[predicted.item()]
        confidence_score = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]

        return emotion, confidence_score, all_probs

    def test_single_file(self, audio_path):
        """Test a single audio file and display results"""
        if not os.path.exists(audio_path):
            print(f"❌ Error: File not found: {audio_path}")
            return

        print(f"Testing: {audio_path}")
        print("Analyzing...\n")

        emotion, confidence, probabilities = self.predict_file(audio_path)

        # Display results
        print("=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"File: {os.path.basename(audio_path)}")
        print(f"Predicted Emotion: {emotion.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"\nProbability Distribution:")

        prob_pairs = list(zip(self.emotions, probabilities))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)

        for emo, prob in prob_pairs:
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {emo:8s}: {bar} {prob * 100:5.2f}%")
        print("=" * 60)

    def test_multiple_files(self, file_paths):
        """Test multiple audio files"""
        results = []

        print(f"\nTesting {len(file_paths)} audio files...\n")
        print("=" * 60)

        for i, audio_path in enumerate(file_paths, 1):
            if not os.path.exists(audio_path):
                print(f"{i}. ❌ File not found: {audio_path}")
                continue

            emotion, confidence, _ = self.predict_file(audio_path)
            filename = os.path.basename(audio_path)

            print(f"{i}. {filename:30s} → {emotion:8s} ({confidence * 100:5.2f}%)")
            results.append({
                'file': filename,
                'emotion': emotion,
                'confidence': confidence
            })

        print("=" * 60)
        return results

    def test_folder(self, folder_path, extension=".wav"):
        """Test all audio files in a folder"""
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Error: Folder not found: {folder_path}")
            return

        # Get all audio files
        audio_files = list(folder.glob(f"*{extension}"))

        if not audio_files:
            print(f"❌ No {extension} files found in {folder_path}")
            return

        print(f"Found {len(audio_files)} audio files in {folder_path}\n")

        # Test all files
        results = self.test_multiple_files([str(f) for f in audio_files])

        # Summary statistics
        print("\n📊 SUMMARY:")
        print("-" * 60)
        emotion_counts = {}
        for result in results:
            emotion = result['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(results) * 100
            print(f"{emotion:10s}: {count:3d} files ({percentage:5.1f}%)")
        print("-" * 60)


def main():
    """Main testing function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test emotion classifier on audio files")
    parser.add_argument("--file", type=str, help="Path to single audio file")
    parser.add_argument("--folder", type=str, help="Path to folder containing audio files")
    parser.add_argument("--files", nargs='+', help="List of audio file paths")
    parser.add_argument("--model", type=str, default="models/best_acc_model.pth",
                        help="Path to model checkpoint")

    args = parser.parse_args()

    # Initialize tester
    tester = AudioFileTester(args.model)

    # Test based on arguments
    if args.file:
        # Test single file
        tester.test_single_file(args.file)

    elif args.folder:
        # Test all files in folder
        tester.test_folder(args.folder)

    elif args.files:
        # Test multiple specified files
        tester.test_multiple_files(args.files)

    else:
        # Interactive mode - ask for file path
        print("=" * 60)
        print("AUDIO FILE EMOTION TESTER")
        print("=" * 60)
        print("\nEnter audio file path (or 'quit' to exit):")

        while True:
            file_path = input("\n> ").strip().strip('"')

            if file_path.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if file_path:
                tester.test_single_file(file_path)
            else:
                print("Please enter a valid file path")


if __name__ == "__main__":
    main()