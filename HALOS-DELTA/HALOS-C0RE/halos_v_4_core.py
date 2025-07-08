# HALOS V4: Full Multimodal + Story + Drawing + Assistant + All Prior Features + Neurodivergent Design + Upgrades

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
import chromadb
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

class AIMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="chroma_data")
        self.collection = self.client.get_or_create_collection("memories")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def store(self, text: str) -> None:
        embedding = self.embedder.encode(text)
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[text],
            ids=[str(time.time())]
        )

    def search(self, query: str, n_results: int = 3) -> List[str]:
        query_embed = self.embedder.encode(query)
        result = self.collection.query(
            query_embeddings=[query_embed.tolist()],
            n_results=n_results
        )
        return result['documents'][0] if result['documents'] else []

class HALOSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HALOS V4 Enhanced")
        self.root.geometry("1300x950")

        self.tts = pyttsx3.init()
        self.recognizer = whisper.DecodingOptions(fp16=False)
        self.model = WhisperModel().model
        self.audio_buffer = AudioBuffer()

        self.memory_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory = AIMemory()
        self.chat_history = []

        self.emotion_classifier = pipeline("text-classification", model="finiteautomata/bertweet-base-emotion-analysis")

        self.setup_ui()
        self.start_audio_capture()
        self.start_services()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_assistant_tab()
        self.setup_memory_tab()
        self.setup_finance_tab()

    def setup_assistant_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Assistant")

        self.chat_display = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled', height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True)

        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=tk.X)

        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.process_command)

        ttk.Button(input_frame, text="Send", command=self.process_command).pack(side=tk.RIGHT)

    def setup_memory_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Memory Timeline")

        ttk.Label(tab, text="Search Memory:").pack(anchor='w')
        self.memory_query = ttk.Entry(tab)
        self.memory_query.pack(fill=tk.X)
        ttk.Button(tab, text="Search", command=self.search_memory).pack()

        self.memory_result = scrolledtext.ScrolledText(tab)
        self.memory_result.pack(fill=tk.BOTH, expand=True)

    def setup_finance_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Finance")

        self.data = {'Food': 50, 'Transport': 30, 'Misc': 20}

        ttk.Button(tab, text="Visualize", command=self.visualize_spending).pack()
        self.finance_canvas = tk.Canvas(tab, width=500, height=400)
        self.finance_canvas.pack()

    def visualize_spending(self):
        fig, ax = plt.subplots()
        categories = list(self.data.keys())
        values = list(self.data.values())
        ax.pie(values, labels=categories, autopct='%1.1f%%')
        ax.set_title("Spending Breakdown")

        for widget in self.finance_canvas.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.finance_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def process_command(self, event=None):
        query = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        self.chat_display.config(state='normal')

        emotion = self.emotion_classifier(query)[0]['label']
        tone = "empathetic" if emotion in ["sadness", "anger"] else "friendly"

        prompt = f"Respond in a {tone} tone: {query}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        self.chat_display.insert(tk.END, f"You: {query}\nHALOS: {response}\n")
        self.chat_display.config(state='disabled')

        self.memory.store(f"{query}\n{response}")

    def search_memory(self):
        query = self.memory_query.get().strip()
        results = self.memory.search(query)
        self.memory_result.delete(1.0, tk.END)
        self.memory_result.insert(tk.END, "\n---\n".join(results))

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
