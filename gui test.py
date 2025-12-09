"""
Simple GUI for emotion classification
"""
import tkinter as tk
from tkinter import ttk, filedialog
import torch
import sounddevice as sd
import numpy as np
from datetime import datetime
import threading

from config import *
from model import EmotionCNN
from preprocessing import process_microphone_audio


class EmotionClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Emotion Classifier")
        self.root.geometry("600x500")

        # Load model
        self.device = DEVICE
        self.model = EmotionCNN(num_classes=NUM_CLASSES).to(self.device)

        model_path = "models/best_acc_model.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Setup GUI
        self.setup_ui()

    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="🎤 Voice Emotion Classifier",
                         font=("Arial", 20, "bold"))
        title.pack(pady=20)

        # Status
        self.status_label = tk.Label(self.root, text="Ready",
                                     font=("Arial", 12))
        self.status_label.pack(pady=10)

        # Record button
        self.record_btn = tk.Button(self.root, text="🎙️ Record Voice",
                                    font=("Arial", 14), bg="#4CAF50", fg="white",
                                    command=self.record_and_predict, height=2, width=20)
        self.record_btn.pack(pady=20)

        # Result frame
        result_frame = tk.LabelFrame(self.root, text="Prediction Results",
                                     font=("Arial", 12), padx=20, pady=20)
        result_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Emotion label
        self.emotion_label = tk.Label(result_frame, text="---",
                                      font=("Arial", 24, "bold"))
        self.emotion_label.pack(pady=10)

        # Confidence label
        self.confidence_label = tk.Label(result_frame, text="",
                                         font=("Arial", 14))
        self.confidence_label.pack(pady=5)

        # Probabilities
        self.prob_text = tk.Text(result_frame, height=8, width=50,
                                 font=("Courier", 10))
        self.prob_text.pack(pady=10)

    def record_and_predict(self):
        # Disable button during recording
        self.record_btn.config(state="disabled")
        self.status_label.config(text="🎤 Recording... Speak now!")

        # Run in thread to not freeze GUI
        thread = threading.Thread(target=self._record_thread)
        thread.start()

    def _record_thread(self):
        # Record audio
        recording = sd.rec(int(MIC_DURATION * MIC_SAMPLE_RATE),
                           samplerate=MIC_SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        audio_data = recording.flatten()

        # Update status
        self.status_label.config(text="🔄 Analyzing...")

        # Predict
        mel_spec = process_microphone_audio(audio_data, sr=MIC_SAMPLE_RATE)
        mel_spec = mel_spec.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        emotion = EMOTIONS[predicted.item()]
        conf_score = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]

        # Update GUI
        self.root.after(0, self._update_results, emotion, conf_score, all_probs)

    def _update_results(self, emotion, confidence, probabilities):
        # Update emotion
        self.emotion_label.config(text=f"🎭 {emotion.upper()}")

        # Update confidence
        self.confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")

        # Update probabilities
        self.prob_text.delete(1.0, tk.END)
        self.prob_text.insert(tk.END, "Probability Distribution:\n\n")

        prob_pairs = list(zip(EMOTIONS, probabilities))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)

        for emo, prob in prob_pairs:
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            self.prob_text.insert(tk.END, f"{emo:8s}: {bar} {prob * 100:5.2f}%\n")

        # Update status and re-enable button
        self.status_label.config(text="✅ Ready")
        self.record_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionClassifierGUI(root)
    root.mainloop()