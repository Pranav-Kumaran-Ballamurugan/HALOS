# HALOS Version 2 - Assistant + Voice Input + TTS

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import openai
import os
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class HalosV2(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HALOS V2 - GPT + Voice")
        self.geometry("800x600")

        self.chat_display = scrolledtext.ScrolledText(self, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.input_field = ttk.Entry(self)
        self.input_field.pack(fill=tk.X, padx=10)
        self.input_field.bind("<Return>", self.ask_gpt)

        self.ask_button = ttk.Button(self, text="Ask", command=self.ask_gpt)
        self.ask_button.pack(pady=5)

        self.voice_button = ttk.Button(self, text="🎤 Speak", command=self.ask_voice)
        self.voice_button.pack(pady=5)

        self.tts = pyttsx3.init()
        self.recognizer = sr.Recognizer()

    def ask_gpt(self, event=None):
        prompt = self.input_field.get().strip()
        if not prompt:
            return
        self.input_field.delete(0, tk.END)
        self.append_chat(f"You: {prompt}")

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()
            self.append_chat(f"HALOS: {answer}")
            self.tts.say(answer)
            self.tts.runAndWait()
        except Exception as e:
            self.append_chat(f"Error: {str(e)}")

    def ask_voice(self):
        with sr.Microphone() as source:
            self.append_chat("Listening...")
            audio = self.recognizer.listen(source)
        try:
            prompt = self.recognizer.recognize_google(audio)
            self.append_chat(f"You (voice): {prompt}")
            self.input_field.delete(0, tk.END)
            self.input_field.insert(0, prompt)
            self.ask_gpt()
        except sr.UnknownValueError:
            self.append_chat("Could not understand audio")
        except Exception as e:
            self.append_chat(f"Voice Error: {str(e)}")

    def append_chat(self, text):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text + "\n")
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)

if __name__ == '__main__':
    app = HalosV2()
    app.mainloop()
