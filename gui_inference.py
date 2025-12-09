"""
Enhanced GUI with confidence meter and real-time audio waveform visualization
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import sounddevice as sd
import numpy as np
from datetime import datetime
import threading
import time
import queue
from config import *
from model import EmotionCNN
from preprocessing import process_microphone_audio


class EnhancedEmotionClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Emotion Classifier")
        self.root.geometry("900x850")
        self.root.resizable(False, False)

        # Modern color scheme
        self.bg_color = "#1e1e2e"
        self.accent_color = "#89b4fa"
        self.success_color = "#a6e3a1"
        self.danger_color = "#f38ba8"
        self.warning_color = "#fab387"
        self.text_color = "#cdd6f4"
        self.card_color = "#313244"

        self.root.configure(bg=self.bg_color)

        # Recording state
        self.is_recording = False
        self.countdown_active = False
        self.audio_queue = queue.Queue()
        self.waveform_data = []

        # Load model
        self.load_model()

        # Setup GUI
        self.setup_ui()

    def load_model(self):
        """Load the trained model"""
        try:
            self.device = DEVICE
            self.model = EmotionCNN(num_classes=NUM_CLASSES).to(self.device)

            model_path = "models/best_acc_model.pth"
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.model_loaded = True
            print("✓ Model loaded successfully")
        except Exception as e:
            self.model_loaded = False
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def setup_ui(self):
        """Setup the user interface"""

        # ==================== HEADER ====================
        header_frame = tk.Frame(self.root, bg=self.bg_color)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))

        title_label = tk.Label(
            header_frame,
            text="🎤 Voice Emotion Classifier ",
            font=("Segoe UI", 28, "bold"),
            bg=self.bg_color,
            fg=self.text_color
        )
        title_label.pack()

        subtitle_label = tk.Label(
            header_frame,
            text="AI-Powered Emotion Detection with Real-time Visualization",
            font=("Segoe UI", 11),
            bg=self.bg_color,
            fg=self.accent_color
        )
        subtitle_label.pack(pady=(5, 0))

        # ==================== STATUS CARD ====================
        status_card = tk.Frame(self.root, bg=self.card_color, relief="flat")
        status_card.pack(fill="x", padx=20, pady=10)

        status_inner = tk.Frame(status_card, bg=self.card_color)
        status_inner.pack(padx=20, pady=15)

        self.status_icon = tk.Label(
            status_inner,
            text="✓",
            font=("Segoe UI", 24),
            bg=self.card_color,
            fg=self.success_color
        )
        self.status_icon.pack(side="left", padx=(0, 10))

        self.status_label = tk.Label(
            status_inner,
            text="Ready to record" if self.model_loaded else "Model not loaded!",
            font=("Segoe UI", 14, "bold"),
            bg=self.card_color,
            fg=self.text_color
        )
        self.status_label.pack(side="left")

        # ==================== COUNTDOWN & WAVEFORM ====================
        viz_container = tk.Frame(self.root, bg=self.bg_color)
        viz_container.pack(pady=15)

        # Left: Countdown timer
        countdown_frame = tk.Frame(viz_container, bg=self.bg_color)
        countdown_frame.pack(side="left", padx=(0, 20))

        self.countdown_canvas = tk.Canvas(
            countdown_frame,
            width=180,
            height=180,
            bg=self.bg_color,
            highlightthickness=0
        )
        self.countdown_canvas.pack()

        # Circle background
        self.countdown_canvas.create_oval(
            10, 10, 170, 170,
            outline=self.card_color,
            width=8
        )

        # Progress arc
        self.countdown_arc = self.countdown_canvas.create_arc(
            10, 10, 170, 170,
            start=90,
            extent=0,
            outline=self.accent_color,
            width=8,
            style="arc"
        )

        # Countdown number
        self.countdown_text = self.countdown_canvas.create_text(
            90, 90,
            text="",
            font=("Segoe UI", 48, "bold"),
            fill=self.text_color
        )

        # Countdown label
        self.countdown_label = self.countdown_canvas.create_text(
            90, 130,
            text="",
            font=("Segoe UI", 12),
            fill=self.accent_color
        )

        # Right: Waveform visualization
        waveform_frame = tk.Frame(viz_container, bg=self.card_color)
        waveform_frame.pack(side="left")

        waveform_header = tk.Label(
            waveform_frame,
            text="📊 Live Audio Waveform",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color
        )
        waveform_header.pack(pady=(10, 5))

        self.waveform_canvas = tk.Canvas(
            waveform_frame,
            width=450,
            height=150,
            bg="#1e1e2e",
            highlightthickness=1,
            highlightbackground=self.accent_color
        )
        self.waveform_canvas.pack(padx=15, pady=(0, 15))

        # Draw center line
        self.waveform_canvas.create_line(
            0, 75, 450, 75,
            fill=self.card_color,
            width=1
        )

        # ==================== CONTROL BUTTONS ====================
        button_frame = tk.Frame(self.root, bg=self.bg_color)
        button_frame.pack(pady=20)

        self.record_btn = tk.Button(
            button_frame,
            text="🎙️  Start Recording",
            font=("Segoe UI", 14, "bold"),
            bg=self.accent_color,
            fg="#1e1e2e",
            activebackground="#74c7ec",
            activeforeground="#1e1e2e",
            relief="flat",
            cursor="hand2",
            width=20,
            height=2,
            command=self.start_recording
        )
        self.record_btn.pack(side="left", padx=10)

        file_btn = tk.Button(
            button_frame,
            text="📁  Upload File",
            font=("Segoe UI", 14, "bold"),
            bg=self.card_color,
            fg=self.text_color,
            activebackground="#45475a",
            activeforeground=self.text_color,
            relief="flat",
            cursor="hand2",
            width=20,
            height=2,
            command=self.upload_file
        )
        file_btn.pack(side="left", padx=10)

        # ==================== RESULT CARD ====================
        result_card = tk.Frame(self.root, bg=self.card_color, relief="flat")
        result_card.pack(fill="both", expand=True, padx=20, pady=10)

        result_header = tk.Label(
            result_card,
            text="Prediction Results",
            font=("Segoe UI", 16, "bold"),
            bg=self.card_color,
            fg=self.text_color
        )
        result_header.pack(pady=(20, 10))

        # Emotion display with confidence indicator
        emotion_container = tk.Frame(result_card, bg=self.card_color)
        emotion_container.pack(pady=10)

        # Left: Emoji and emotion
        emotion_left = tk.Frame(emotion_container, bg=self.card_color)
        emotion_left.pack(side="left", padx=30)

        self.emotion_emoji = tk.Label(
            emotion_left,
            text="😐",
            font=("Segoe UI", 64),
            bg=self.card_color
        )
        self.emotion_emoji.pack()

        self.emotion_label = tk.Label(
            emotion_left,
            text="---",
            font=("Segoe UI", 28, "bold"),
            bg=self.card_color,
            fg=self.text_color
        )
        self.emotion_label.pack()

        self.confidence_label = tk.Label(
            emotion_left,
            text="",
            font=("Segoe UI", 14),
            bg=self.card_color,
            fg=self.accent_color
        )
        self.confidence_label.pack(pady=5)

        # Right: Confidence meter
        confidence_right = tk.Frame(emotion_container, bg=self.card_color)
        confidence_right.pack(side="left", padx=30)

        confidence_title = tk.Label(
            confidence_right,
            text="Model Confidence",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color
        )
        confidence_title.pack(pady=(10, 5))

        # Circular confidence gauge
        self.confidence_canvas = tk.Canvas(
            confidence_right,
            width=120,
            height=120,
            bg=self.card_color,
            highlightthickness=0
        )
        self.confidence_canvas.pack()

        # Background circle
        self.confidence_canvas.create_oval(
            10, 10, 110, 110,
            outline="#45475a",
            width=12
        )

        # Progress arc (will be updated)
        self.confidence_arc = self.confidence_canvas.create_arc(
            10, 10, 110, 110,
            start=90,
            extent=0,
            outline=self.success_color,
            width=12,
            style="arc"
        )

        # Confidence percentage
        self.confidence_percent = self.confidence_canvas.create_text(
            60, 60,
            text="0%",
            font=("Segoe UI", 24, "bold"),
            fill=self.text_color
        )

        # Confidence assessment
        self.confidence_assessment = tk.Label(
            confidence_right,
            text="",
            font=("Segoe UI", 10),
            bg=self.card_color,
            fg=self.accent_color
        )
        self.confidence_assessment.pack(pady=(5, 0))

        # ==================== PROBABILITY DISTRIBUTION ====================
        prob_section = tk.Frame(result_card, bg=self.card_color)
        prob_section.pack(fill="x", padx=30, pady=(20, 20))

        prob_title = tk.Label(
            prob_section,
            text="Probability Distribution:",
            font=("Segoe UI", 12, "bold"),
            bg=self.card_color,
            fg=self.text_color,
            anchor="w"
        )
        prob_title.pack(fill="x", pady=(0, 10))

        self.prob_bars = {}
        self.prob_labels = {}

        for emotion in EMOTIONS:
            bar_container = tk.Frame(prob_section, bg=self.card_color)
            bar_container.pack(fill="x", pady=4)

            # Emotion name
            name_label = tk.Label(
                bar_container,
                text=emotion.capitalize(),
                font=("Segoe UI", 11),
                bg=self.card_color,
                fg=self.text_color,
                width=10,
                anchor="w"
            )
            name_label.pack(side="left")

            # Progress bar background
            bar_bg = tk.Frame(bar_container, bg="#45475a", height=24)
            bar_bg.pack(side="left", fill="x", expand=True, padx=10)

            bar = tk.Frame(bar_bg, bg=self.accent_color, height=24)
            bar.place(relx=0, rely=0, relwidth=0, relheight=1)

            self.prob_bars[emotion] = bar

            # Percentage label
            percent_label = tk.Label(
                bar_container,
                text="0.0%",
                font=("Segoe UI", 11, "bold"),
                bg=self.card_color,
                fg=self.text_color,
                width=7,
                anchor="e"
            )
            percent_label.pack(side="left")

            self.prob_labels[emotion] = percent_label

        # ==================== FOOTER ====================
        footer = tk.Label(
            self.root,
            text=f"Model: best_acc_model.pth  |  Device: {DEVICE}  |  Duration: {MIC_DURATION}s  |  Sample Rate: {MIC_SAMPLE_RATE}Hz",
            font=("Segoe UI", 9),
            bg=self.bg_color,
            fg="#6c7086"
        )
        footer.pack(side="bottom", pady=10)

    def update_waveform(self, audio_chunk):
        """Update waveform visualization with new audio data"""
        if len(audio_chunk) == 0:
            return

        # Clear canvas
        self.waveform_canvas.delete("waveform")

        # Downsample for visualization (take every Nth sample)
        stride = max(1, len(audio_chunk) // 450)
        samples = audio_chunk[::stride][:450]

        # Normalize to canvas height
        if len(samples) > 0:
            normalized = (samples / np.max(np.abs(samples) + 1e-10)) * 60  # Scale to ±60 pixels

            # Draw waveform
            for i in range(len(samples) - 1):
                x1 = i
                y1 = 75 - normalized[i]  # Center at y=75
                x2 = i + 1
                y2 = 75 - normalized[i + 1]

                # Color based on amplitude
                amplitude = abs(normalized[i])
                if amplitude > 40:
                    color = self.danger_color
                elif amplitude > 20:
                    color = self.warning_color
                else:
                    color = self.accent_color

                self.waveform_canvas.create_line(
                    x1, y1, x2, y2,
                    fill=color,
                    width=2,
                    tags="waveform"
                )

    def clear_waveform(self):
        """Clear waveform display"""
        self.waveform_canvas.delete("waveform")
        self.waveform_canvas.create_line(
            0, 75, 450, 75,
            fill=self.card_color,
            width=1
        )

    def update_countdown(self, seconds_remaining):
        """Update countdown display"""
        self.countdown_canvas.itemconfig(
            self.countdown_text,
            text=str(seconds_remaining)
        )

        self.countdown_canvas.itemconfig(
            self.countdown_label,
            text="Recording..." if seconds_remaining > 0 else "Processing..."
        )

        progress = (MIC_DURATION - seconds_remaining) / MIC_DURATION
        extent = -360 * progress

        if seconds_remaining <= 1:
            color = self.danger_color
        elif seconds_remaining <= 2:
            color = self.warning_color
        else:
            color = self.accent_color

        self.countdown_canvas.itemconfig(
            self.countdown_arc,
            extent=extent,
            outline=color
        )

    def reset_countdown(self):
        """Reset countdown display"""
        self.countdown_canvas.itemconfig(self.countdown_text, text="")
        self.countdown_canvas.itemconfig(self.countdown_label, text="")
        self.countdown_canvas.itemconfig(self.countdown_arc, extent=0)

    def update_confidence_meter(self, confidence):
        """Update the confidence gauge"""
        # Update percentage text
        self.confidence_canvas.itemconfig(
            self.confidence_percent,
            text=f"{int(confidence * 100)}%"
        )

        # Update arc (0% to 100% = 0° to 360°)
        extent = -360 * confidence

        # Color based on confidence level
        if confidence >= 0.8:
            color = self.success_color
            assessment = "Very Confident ✓"
            assessment_color = self.success_color
        elif confidence >= 0.6:
            color = self.accent_color
            assessment = "Confident"
            assessment_color = self.accent_color
        elif confidence >= 0.4:
            color = self.warning_color
            assessment = "Moderate"
            assessment_color = self.warning_color
        else:
            color = self.danger_color
            assessment = "Low Confidence ⚠"
            assessment_color = self.danger_color

        self.confidence_canvas.itemconfig(
            self.confidence_arc,
            extent=extent,
            outline=color
        )

        self.confidence_assessment.config(
            text=assessment,
            fg=assessment_color
        )

    def start_recording(self):
        """Start recording with countdown and waveform"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded!")
            return

        if self.is_recording:
            return

        self.is_recording = True
        self.record_btn.config(
            state="disabled",
            bg="#45475a",
            text="⏺️  Recording..."
        )

        self.update_status("🎤", "Recording your voice...", self.danger_color)

        # Clear waveform
        self.clear_waveform()

        # Start recording thread
        thread = threading.Thread(target=self._record_with_visualization)
        thread.start()

    def _record_with_visualization(self):
        """Record with live countdown and waveform visualization"""
        start_time = time.time()
        chunks = []

        # Audio callback for real-time visualization
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(status)
            chunks.append(indata.copy())
            # Update waveform in main thread
            self.root.after(0, self.update_waveform, indata[:, 0])

        # Start recording with callback
        stream = sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            dtype='float32'
        )

        with stream:
            # Countdown loop
            while time.time() - start_time < MIC_DURATION:
                elapsed = time.time() - start_time
                remaining = max(0, MIC_DURATION - elapsed)

                self.root.after(0, self.update_countdown, int(remaining) + 1)
                time.sleep(0.1)

        # Combine all chunks
        recording = np.concatenate(chunks, axis=0)
        audio_data = recording.flatten()

        # Show processing
        self.root.after(0, self.update_countdown, 0)
        self.root.after(0, self.update_status, "⚙️", "Analyzing emotion...", self.accent_color)

        # Process audio
        emotion, confidence, probabilities = self.predict(audio_data)

        # Update UI with results
        self.root.after(0, self._update_results, emotion, confidence, probabilities)
        self.root.after(0, self.reset_countdown)
        self.root.after(0, self.clear_waveform)

        # Re-enable button
        self.is_recording = False
        self.root.after(0, self.record_btn.config, {
            "state": "normal",
            "bg": self.accent_color,
            "text": "🎙️  Start Recording"
        })

        self.root.after(0, self.update_status, "✓", "Analysis complete!", self.success_color)

    def upload_file(self):
        """Upload and analyze audio file"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded!")
            return

        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac"),
                ("WAV Files", "*.wav"),
                ("MP3 Files", "*.mp3"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.update_status("⚙️", "Loading and analyzing file...", self.accent_color)

        thread = threading.Thread(target=self._process_file, args=(file_path,))
        thread.start()

    def _process_file(self, file_path):
        """Process uploaded file"""
        try:
            import librosa

            audio_data, _ = librosa.load(file_path, sr=MIC_SAMPLE_RATE, duration=MIC_DURATION)

            # Show waveform of loaded file
            self.root.after(0, self.update_waveform, audio_data)

            emotion, confidence, probabilities = self.predict(audio_data)

            self.root.after(0, self._update_results, emotion, confidence, probabilities)
            self.root.after(0, self.update_status, "✓", "File analyzed successfully!", self.success_color)
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Failed to process file:\n{e}")
            self.root.after(0, self.update_status, "✗", "Analysis failed", self.danger_color)

    def predict(self, audio_data):
        """Predict emotion from audio data"""
        mel_spec = process_microphone_audio(audio_data, sr=MIC_SAMPLE_RATE)
        mel_spec = mel_spec.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        emotion = EMOTIONS[predicted.item()]
        confidence_score = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]

        return emotion, confidence_score, all_probs

    def _update_results(self, emotion, confidence, probabilities):
        """Update result display"""
        emoji_map = {
            'anger': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢'
        }

        # Update emotion display
        self.emotion_emoji.config(text=emoji_map.get(emotion, '😐'))
        self.emotion_label.config(text=emotion.upper())
        self.confidence_label.config(text=f"Prediction: {confidence * 100:.2f}%")

        # Update confidence meter
        self.update_confidence_meter(confidence)

        # Update probability bars
        for i, emo in enumerate(EMOTIONS):
            prob = probabilities[i]

            self.prob_bars[emo].place(relwidth=prob)
            self.prob_labels[emo].config(text=f"{prob * 100:.1f}%")

            if emo == emotion:
                self.prob_bars[emo].config(bg=self.success_color)
                self.prob_labels[emo].config(fg=self.success_color, font=("Segoe UI", 11, "bold"))
            else:
                self.prob_bars[emo].config(bg=self.accent_color)
                self.prob_labels[emo].config(fg=self.text_color, font=("Segoe UI", 11))

    def update_status(self, icon, text, color):
        """Update status display"""
        self.status_icon.config(text=icon, fg=color)
        self.status_label.config(text=text)


def main():
    """Main function"""
    root = tk.Tk()
    app = EnhancedEmotionClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()