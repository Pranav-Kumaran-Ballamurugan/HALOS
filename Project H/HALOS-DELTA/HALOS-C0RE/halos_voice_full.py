import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import os
import time
import queue
from datetime import datetime
import webbrowser
import openai
from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr
import whisper
import sounddevice as sd
import numpy as np

# Load env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Whisper model singleton
class WhisperModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = whisper.load_model("tiny")
        return cls._instance

# Audio circular buffer
class AudioBuffer:
    def __init__(self, seconds=5, rate=16000):
        self.buffer = np.zeros(seconds * rate, dtype=np.float32)
        self.size = seconds * rate
        self.index = 0
        self.lock = threading.Lock()
        self.sample_rate = rate

    def add_data(self, indata):
        with self.lock:
            available = self.size - self.index
            if len(indata) <= available:
                self.buffer[self.index:self.index + len(indata)] = indata
                self.index += len(indata)
            else:
                remain = len(indata) - available
                self.buffer[self.index:] = indata[:available]
                self.buffer[:remain] = indata[available:]
                self.index = remain

    def get_last(self, seconds):
        samples = int(seconds * self.sample_rate)
        with self.lock:
            if self.index >= samples:
                return self.buffer[self.index - samples:self.index]
            else:
                return np.concatenate((
                    self.buffer[self.size - (samples - self.index):],
                    self.buffer[:self.index]
                ))

# HALOS with full voice features
class HALOSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HALOS AI Assistant")
        self.root.geometry("1000x700")
        self.tts = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.audio_buffer = AudioBuffer()
        self.model = WhisperModel().model
        self.authenticated = True

        self.setup_ui()
        self.start_audio_capture()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="HALOS", font=("Helvetica", 24, "bold")).pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Assistant")

        self.chat_display = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled', height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)

        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=tk.X, pady=5)

        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.process_command)

        ttk.Button(input_frame, text="Send", command=self.process_command).pack(side=tk.RIGHT)
        ttk.Button(input_frame, text="🎙️ Speak", command=self.listen_from_buffer).pack(side=tk.RIGHT, padx=5)

    def process_command(self, event=None):
        command = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        if command:
            self.display_message(f"You: {command}", "user")
            self.handle_command(command)

    def handle_command(self, command):
        if command.lower().startswith("play "):
            query = command[5:].strip().replace(" ", "+")
            url = f"https://www.youtube.com/results?search_query={query}"
            webbrowser.open(url)
            self.display_message(f"HALOS: 🎵 Searching YouTube for {command[5:].strip()}", "halos")
            self.speak(f"Searching YouTube for {command[5:].strip()}")
            return

        try:
            response = openai.ChatCompletion.create(
                model=os.getenv("GPT_MODEL", "gpt-4-turbo"),
                messages=[
                    {"role": "system", "content": "You're a helpful assistant."},
                    {"role": "user", "content": command}
                ]
            )
            reply = response.choices[0].message.content.strip()
            self.display_message(f"HALOS: {reply}", "halos")
            self.speak(reply)
        except Exception as e:
            self.display_message(f"HALOS: Error - {str(e)}", "halos")

    def display_message(self, message, sender):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def speak(self, text):
        self.tts.say(text)
        self.tts.runAndWait()

    def start_audio_capture(self):
        def callback(indata, frames, time, status):
            self.audio_buffer.add_data(indata[:, 0])
        self.stream = sd.InputStream(callback=callback, channels=1, samplerate=16000)
        self.stream.start()

    def listen_from_buffer(self):
        audio = self.audio_buffer.get_last(5)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(self.model, mel, options)
        text = result.text.strip()
        if text:
            self.display_message(f"You (spoken): {text}", "user")
            self.handle_command(text)
        else:
            self.display_message("HALOS: I didn't catch that.", "halos")
            self.speak("I didn't catch that")

if __name__ == "__main__":
    root = tk.Tk()
    app = HALOSApp(root)
    root.mainloop()
