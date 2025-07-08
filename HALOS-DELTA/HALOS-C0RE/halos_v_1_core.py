# HALOS Version 1 - Minimal Assistant with Basic GPT Response

import tkinter as tk
from tkinter import ttk, scrolledtext
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class HalosV1(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HALOS V1 - Basic GPT Assistant")
        self.geometry("800x600")

        self.chat_display = scrolledtext.ScrolledText(self, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.input_field = ttk.Entry(self)
        self.input_field.pack(fill=tk.X, padx=10)
        self.input_field.bind("<Return>", self.ask_gpt)

        self.ask_button = ttk.Button(self, text="Ask", command=self.ask_gpt)
        self.ask_button.pack(pady=5)

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
        except Exception as e:
            self.append_chat(f"Error: {str(e)}")

    def append_chat(self, text):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text + "\n")
        self.chat_display.config(state='disabled')
        self.chat_display.yview(tk.END)

if __name__ == '__main__':
    app = HalosV1()
    app.mainloop()
