import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import time
import queue
import json
import openai
import stripe
from dotenv import load_dotenv
from datetime import datetime
import webbrowser
import pyttsx3
import whisper
import sounddevice as sd
import numpy as np
import hashlib
import logging

# Load configuration
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv("STRIPE_LIVE_KEY")

logging.basicConfig(level=logging.INFO, filename="halos.log", format="%(asctime)s - %(levelname)s - %(message)s")

class WhisperModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = whisper.load_model("tiny")
        return cls._instance

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

class HALOSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HALOS AI Assistant")
        self.root.geometry("1000x800")

        self.tts = pyttsx3.init()
        self.recognizer = whisper.DecodingOptions(fp16=False)
        self.model = WhisperModel().model
        self.audio_buffer = AudioBuffer()

        self.setup_ui()
        self.start_audio_capture()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_assistant_tab()
        self.setup_codefixer_tab()
        self.setup_payments_tab()
        self.setup_security_tab()

    def setup_assistant_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Assistant")

        self.chat_display = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled', height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)

        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=tk.X)

        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.process_command)

        ttk.Button(input_frame, text="Send", command=self.process_command).pack(side=tk.RIGHT)
        ttk.Button(input_frame, text="🎙️ Speak", command=self.listen_from_buffer).pack(side=tk.RIGHT, padx=5)

    def setup_codefixer_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Code Fixer")

        self.code_input = scrolledtext.ScrolledText(tab, wrap=tk.WORD, height=15)
        self.code_input.pack(fill=tk.BOTH, expand=True)

        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(button_frame, text="Fix Code", command=self.fix_code).pack(side=tk.LEFT)

    def setup_payments_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Payments")

        ttk.Label(tab, text="Amount (USD):").pack()
        self.amount_entry = ttk.Entry(tab)
        self.amount_entry.pack()

        ttk.Label(tab, text="Recipient Email:").pack()
        self.recipient_entry = ttk.Entry(tab)
        self.recipient_entry.pack()

        ttk.Button(tab, text="Process Payment", command=self.process_payment).pack(pady=10)
        self.payment_log = scrolledtext.ScrolledText(tab, height=10)
        self.payment_log.pack(fill=tk.BOTH, expand=True)

    def setup_security_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Security Lab")

        frame = ttk.LabelFrame(tab, text="Hash Cracking Lab")
        frame.pack(fill=tk.X, pady=10)

        ttk.Label(frame, text="Hash:").grid(row=0, column=0, sticky=tk.W)
        self.hash_input = ttk.Entry(frame, width=60)
        self.hash_input.grid(row=0, column=1, padx=5)

        ttk.Label(frame, text="Dictionary File:").grid(row=1, column=0, sticky=tk.W)
        self.dict_path = ttk.Entry(frame, width=60)
        self.dict_path.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="Browse", command=self.browse_dict).grid(row=1, column=2)

        ttk.Button(frame, text="Crack Hash", command=self.crack_hash).grid(row=2, columnspan=3, pady=5)

        self.hash_result = ttk.Label(frame, text="Result: ")
        self.hash_result.grid(row=3, columnspan=3, sticky=tk.W, pady=5)

        self.progress = ttk.Progressbar(frame, mode='determinate')
        self.progress.grid(row=4, columnspan=3, sticky=tk.EW, pady=5)

    def speak(self, text):
        self.tts.say(text)
        self.tts.runAndWait()

    def display_message(self, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def process_command(self, event=None):
        command = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        if command:
            self.display_message(f"You: {command}")
            self.handle_command(command)

    def handle_command(self, command):
        if command.lower().startswith("play "):
            query = command[5:].strip().replace(" ", "+")
            url = f"https://www.youtube.com/results?search_query={query}"
            webbrowser.open(url)
            self.display_message(f"HALOS: 🎵 Searching YouTube for {command[5:].strip()}")
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
            self.display_message(f"HALOS: {reply}")
            self.speak(reply)
        except Exception as e:
            self.display_message(f"HALOS: Error - {str(e)}")
            logging.error(f"Chat command failed: {str(e)}")

    def fix_code(self):
        code = self.code_input.get("1.0", tk.END).strip()
        if not code:
            messagebox.showinfo("Info", "Please paste some code to fix.")
            return
        try:
            response = openai.ChatCompletion.create(
                model=os.getenv("GPT_MODEL", "gpt-4-turbo"),
                messages=[
                    {"role": "system", "content": "Fix this Python code."},
                    {"role": "user", "content": code}
                ]
            )
            fixed = response.choices[0].message.content.strip()
            self.code_input.delete("1.0", tk.END)
            self.code_input.insert("1.0", fixed)
            self.speak("Code fix complete.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def process_payment(self):
        try:
            amount = float(self.amount_entry.get())
            recipient = self.recipient_entry.get().strip()
            if not recipient:
                raise ValueError("Recipient required")

            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),
                currency="usd",
                description=f"Payment to {recipient}"
            )
            self.payment_log.insert(tk.END, f"$ {amount:.2f} to {recipient} - Intent: {intent.id}\n")
            self.speak("Payment created successfully.")
        except Exception as e:
            messagebox.showerror("Payment Error", str(e))

    def browse_dict(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path:
            self.dict_path.delete(0, tk.END)
            self.dict_path.insert(0, path)
            logging.info(f"Selected dictionary file: {path}")

    def crack_hash(self):
        target_hash = self.hash_input.get().strip().lower()
        dict_file = self.dict_path.get().strip()

        if not target_hash:
            self.hash_result.config(text="Result: Please enter a hash")
            return
        if not dict_file or not os.path.exists(dict_file):
            self.hash_result.config(text="Result: Valid dictionary file required")
            return

        threading.Thread(target=self._crack_hash_thread, args=(target_hash, dict_file), daemon=True).start()

    def _crack_hash_thread(self, target_hash, dict_file):
        self.root.after(0, lambda: self.hash_result.config(text="Processing..."))
        self.root.after(0, lambda: self.progress.config(value=0))

        try:
            algo = self.detect_algorithm(target_hash)
            if not algo:
                self.root.after(0, lambda: self.hash_result.config(text="Result: Unknown hash algorithm"))
                logging.warning(f"Unknown hash: {target_hash}")
                return

            with open(dict_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            total = len(lines)
            start = time.time()

            for i, line in enumerate(lines):
                word = line.strip()
                h = getattr(hashlib, algo)(word.encode()).hexdigest()
                if h == target_hash:
                    self.root.after(0, lambda: self.hash_result.config(
                        text=f"Result: Found match → '{word}' in {time.time()-start:.2f}s"))
                    return
                if i % 100 == 0:
                    p = (i / total) * 100
                    self.root.after(0, lambda v=p: self.progress.config(value=v))

            self.root.after(0, lambda: self.hash_result.config(text="Result: No match found"))
        except Exception as e:
            logging.error(f"Hash crack failed: {str(e)}")
            self.root.after(0, lambda: self.hash_result.config(text=f"Result: Error - {str(e)}"))

    def detect_algorithm(self, h):
        l = len(h)
        return {
            32: 'md5',
            40: 'sha1',
            64: 'sha256',
            96: 'sha384',
            128: 'sha512'
        }.get(l, None)

    def start_audio_capture(self):
        def callback(indata, frames, time, status):
            self.audio_buffer.add_data(indata[:, 0])
        self.stream = sd.InputStream(callback=callback, channels=1, samplerate=16000)
        self.stream.start()

    def listen_from_buffer(self):
        audio = self.audio_buffer.get_last(5)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        result = whisper.decode(self.model, mel, self.recognizer)
        text = result.text.strip()
        if text:
            self.display_message(f"You (spoken): {text}")
            self.handle_command(text)
        else:
            self.display_message("HALOS: I didn't catch that.")
            self.speak("I didn't catch that")

if __name__ == "__main__":
    root = tk.Tk()
    app = HALOSApp(root)
    root.mainloop()
