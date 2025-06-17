# HALOS V4: Full Multimodal + Story + Drawing + Assistant + All Prior Features + Neurodivergent Design + Finance AI

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, Canvas
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
from collections import deque
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv("STRIPE_LIVE_KEY")

class WhisperModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = whisper.load_model("base")
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
        self.root.title("HALOS V4: Multimodal AI Companion")
        self.root.geometry("1300x950")

        self.tts = pyttsx3.init()
        self.recognizer = whisper.DecodingOptions(fp16=False)
        self.model = WhisperModel().model
        self.audio_buffer = AudioBuffer()

        self.high_contrast_mode = False
        self.dyslexia_font_enabled = False

        self.memory_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chat_history = []
        self.memory_db = []
        self.emotion_classifier = pipeline("text-classification", model="finiteautomata/bertweet-base-emotion-analysis")

        self.setup_ui()
        self.start_audio_capture()
        self.start_services()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_assistant_tab()
        self.setup_codefixer_tab()
        self.setup_payments_tab()
        self.setup_security_tab()
        self.setup_learning_tab()
        self.setup_drawing_tab()
        self.setup_accessibility_tab()
        self.setup_finance_tab()

    def setup_assistant_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🧠 Assistant")
        self.assistant_output = scrolledtext.ScrolledText(tab, height=20)
        self.assistant_output.pack(fill=tk.BOTH, expand=True)
        self.assistant_input = ttk.Entry(tab)
        self.assistant_input.pack(fill=tk.X)
        ttk.Button(tab, text="Ask", command=self.run_assistant).pack()

    def run_assistant(self):
        query = self.assistant_input.get()
        if not query: return
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": query}]
            )
            answer = response.choices[0].message.content.strip()
            self.assistant_output.insert(tk.END, f"You: {query}\nHALOS: {answer}\n")
        except Exception as e:
            self.assistant_output.insert(tk.END, f"Error: {str(e)}\n")

    def setup_codefixer_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🛠 Code Fixer")
        self.code_input = scrolledtext.ScrolledText(tab, height=10)
        self.code_input.pack(fill=tk.BOTH, expand=True)
        ttk.Button(tab, text="Fix Code", command=self.fix_code).pack()
        self.code_output = scrolledtext.ScrolledText(tab, height=10)
        self.code_output.pack(fill=tk.BOTH, expand=True)

    def fix_code(self):
        code = self.code_input.get("1.0", tk.END)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Fix this Python code:\n{code}"}]
        )
        self.code_output.insert(tk.END, response.choices[0].message.content)

    def setup_payments_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="💳 Payments")
        ttk.Label(tab, text="Amount:").pack()
        self.pay_amount = ttk.Entry(tab)
        self.pay_amount.pack()
        ttk.Label(tab, text="Description:").pack()
        self.pay_desc = ttk.Entry(tab)
        self.pay_desc.pack()
        ttk.Button(tab, text="Create PaymentIntent", command=self.create_payment).pack()
        self.pay_output = scrolledtext.ScrolledText(tab, height=10)
        self.pay_output.pack()

    def create_payment(self):
        try:
            amount = int(float(self.pay_amount.get()) * 100)
            desc = self.pay_desc.get()
            intent = stripe.PaymentIntent.create(amount=amount, currency="usd", description=desc)
            self.pay_output.insert(tk.END, f"Client Secret: {intent.client_secret}\n")
        except Exception as e:
            self.pay_output.insert(tk.END, f"Error: {str(e)}\n")

    def setup_security_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔐 Security")
        ttk.Label(tab, text="Enter Hash:").pack()
        self.hash_entry = ttk.Entry(tab)
        self.hash_entry.pack()
        ttk.Button(tab, text="Crack Hash", command=self.crack_hash).pack()
        self.hash_result = scrolledtext.ScrolledText(tab, height=10)
        self.hash_result.pack()

    def crack_hash(self):
        hash_to_crack = self.hash_entry.get()
        for word in ["password", "123456", "admin"]:
            if hashlib.md5(word.encode()).hexdigest() == hash_to_crack:
                self.hash_result.insert(tk.END, f"Match found: {word}\n")
                return
        self.hash_result.insert(tk.END, "No match found\n")

    def setup_learning_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📚 Learn")
        self.learn_topic = ttk.Entry(tab)
        self.learn_topic.pack(fill=tk.X)
        ttk.Button(tab, text="Explain", command=self.learn_explain).pack()
        self.learn_output = scrolledtext.ScrolledText(tab, height=20)
        self.learn_output.pack()

    def learn_explain(self):
        topic = self.learn_topic.get()
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Explain: {topic}"}]
        )
        self.learn_output.insert(tk.END, response.choices[0].message.content)

    def setup_drawing_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎨 Drawing")
        self.canvas = Canvas(tab, width=800, height=600, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.drawing_output = scrolledtext.ScrolledText(tab)
        self.drawing_output.pack()
        ttk.Button(tab, text="Analyze Drawing", command=self.analyze_drawing).pack()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+4, y+4, fill="black")

    def analyze_drawing(self):
        self.drawing_output.insert(tk.END, "[Mocked] Drawing analyzed\n")

    def setup_accessibility_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="♿ Accessibility")
        ttk.Button(tab, text="Toggle High Contrast", command=self.toggle_contrast).pack()
        ttk.Button(tab, text="Toggle Dyslexia Font", command=self.toggle_dyslexia_font).pack()

    def toggle_contrast(self):
        self.high_contrast_mode = not self.high_contrast_mode
        bg = "black" if self.high_contrast_mode else "SystemButtonFace"
        fg = "white" if self.high_contrast_mode else "SystemWindowText"
        self.root.tk_setPalette(background=bg, foreground=fg)

    def toggle_dyslexia_font(self):
        self.dyslexia_font_enabled = not self.dyslexia_font_enabled
        font = ('OpenDyslexic', 12) if self.dyslexia_font_enabled else ('Helvetica', 12)
        self.root.option_add("*Font", font)

    def setup_finance_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="💰 Finance AI")
        ttk.Label(tab, text="Expense Description:").pack()
        self.expense_desc = ttk.Entry(tab, width=80)
        self.expense_desc.pack(pady=5)
        ttk.Label(tab, text="Amount (USD):").pack()
        self.expense_amount = ttk.Entry(tab, width=20)
        self.expense_amount.pack(pady=5)
        ttk.Button(tab, text="Add Transaction", command=self.add_transaction).pack(pady=5)
        self.finance_output = scrolledtext.ScrolledText(tab, height=15)
        self.finance_output.pack(fill=tk.BOTH, expand=True)
        self.transactions = []

    def add_transaction(self):
        desc = self.expense_desc.get().strip()
        try:
            amount = float(self.expense_amount.get().strip())
        except ValueError:
            self.finance_output.insert(tk.END, "Invalid amount\n")
            return
        self.transactions.append({"description": desc, "amount": amount})
        self.finance_output.insert(tk.END, f"Logged: ${amount:.2f} - {desc}\n")
        self.expense_desc.delete(0, tk.END)
        self.expense_amount.delete(0, tk.END)

    def start_audio_capture(self):
        def callback(indata, frames, time, status):
            self.audio_buffer.add_data(indata[:, 0])
        threading.Thread(
            target=sd.InputStream(
                callback=callback,
                channels=1,
                samplerate=16000,
                dtype='float32'
            ).start,
            daemon=True
        ).start()

    def start_services(self):
        pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    app = HALOSApp(root)
    root.mainloop()
