# HALOS Version 3 - Multimodal Input (Text + Drawing + Voice) + Memory

import tkinter as tk
from tkinter import ttk, scrolledtext, Canvas
import openai
import os
import pyttsx3
import speech_recognition as sr
import threading
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class HalosV3(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HALOS V3 - Multimodal AI")
        self.geometry("1000x800")

        self.memory_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory = []

        self.tts = pyttsx3.init()
        self.recognizer = sr.Recognizer()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_text_tab()
        self.setup_drawing_tab()

    def setup_text_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Text/Voice")

        self.display = scrolledtext.ScrolledText(tab, height=20)
        self.display.pack(fill=tk.BOTH, expand=True)

        entry_frame = ttk.Frame(tab)
        entry_frame.pack(fill=tk.X)

        self.input = ttk.Entry(entry_frame)
        self.input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input.bind("<Return>", self.ask)

        ttk.Button(entry_frame, text="Ask", command=self.ask).pack(side=tk.RIGHT)
        ttk.Button(entry_frame, text="🎤", command=self.ask_voice).pack(side=tk.RIGHT)

    def setup_drawing_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Drawing")

        self.canvas = Canvas(tab, bg="white", width=800, height=500)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

    def ask(self, event=None):
        text = self.input.get().strip()
        if not text: return
        self.display.insert(tk.END, f"You: {text}\n")
        self.input.delete(0, tk.END)
        threading.Thread(target=self.query_gpt, args=(text,), daemon=True).start()

    def ask_voice(self):
        with sr.Microphone() as source:
            self.display.insert(tk.END, "Listening...\n")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio)
            self.display.insert(tk.END, f"You (voice): {text}\n")
            self.query_gpt(text)
        except Exception as e:
            self.display.insert(tk.END, f"Voice error: {str(e)}\n")

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+4, y+4, fill="black")

    def query_gpt(self, text):
        try:
            embedding = self.memory_model.encode(text)
            self.memory.append((text, embedding))
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": text}]
            )
            answer = response.choices[0].message.content.strip()
            self.display.insert(tk.END, f"HALOS: {answer}\n")
            self.tts.say(answer)
            self.tts.runAndWait()
        except Exception as e:
            self.display.insert(tk.END, f"Error: {str(e)}\n")

if __name__ == '__main__':
    app = HalosV3()
    app.mainloop()
