import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import time
import json
import queue
import hashlib
from datetime import datetime
import webbrowser
import openai
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "OPENAI_KEY": os.getenv("OPENAI_API_KEY"),
    "GPT_MODEL": "gpt-4-turbo"
}

openai.api_key = CONFIG["OPENAI_KEY"]

class HALOSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HALOS AI Assistant")
        self.root.geometry("1000x700")

        self.authenticated = True
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="HALOS", font=("Helvetica", 24, "bold")).pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_assistant_tab()
        self.setup_security_tab()

    def setup_assistant_tab(self):
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

    def setup_security_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Security Lab")

        # Hash cracking UI
        hash_frame = ttk.LabelFrame(tab, text="Hash Cracking Lab")
        hash_frame.pack(fill=tk.X, pady=10)

        ttk.Label(hash_frame, text="Hash:").grid(row=0, column=0, sticky=tk.W)
        self.hash_input = ttk.Entry(hash_frame, width=60)
        self.hash_input.grid(row=0, column=1, padx=5)

        ttk.Label(hash_frame, text="Dictionary File:").grid(row=1, column=0, sticky=tk.W)
        self.dict_path = ttk.Entry(hash_frame, width=60)
        self.dict_path.grid(row=1, column=1, padx=5)
        ttk.Button(hash_frame, text="Browse", command=self.browse_dict).grid(row=1, column=2)

        ttk.Button(hash_frame, text="Crack Hash", command=self.crack_hash).grid(row=2, columnspan=3, pady=5)

        self.hash_result = ttk.Label(hash_frame, text="Result: ")
        self.hash_result.grid(row=3, columnspan=3, sticky=tk.W, pady=5)

    def process_command(self, event=None):
        command = self.user_input.get()
        self.user_input.delete(0, tk.END)
        self.display_message(f"You: {command}", "user")
        reply = self.process_natural_command(command)
        self.display_message(f"HALOS: {reply}", "halos")

    def process_natural_command(self, command):
        if command.lower().startswith("play "):
            search = command[5:].strip().replace(" ", "+")
            url = f"https://www.youtube.com/results?search_query={search}"
            webbrowser.open(url)
            return f"🎵 Searching YouTube for: {command[5:].strip()}"

        try:
            response = openai.ChatCompletion.create(
                model=CONFIG["GPT_MODEL"],
                messages=[
                    {"role": "system", "content": "You're a helpful AI assistant."},
                    {"role": "user", "content": command}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def display_message(self, message, sender):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def browse_dict(self):
        path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if path:
            self.dict_path.delete(0, tk.END)
            self.dict_path.insert(0, path)

    def crack_hash(self):
        target_hash = self.hash_input.get().strip().lower()
        dict_file = self.dict_path.get().strip()

        if not os.path.exists(dict_file):
            self.hash_result.config(text="Result: Dictionary file not found")
            return

        algo = self.detect_algorithm(target_hash)
        if not algo:
            self.hash_result.config(text="Result: Unknown hash algorithm")
            return

        with open(dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                h = getattr(hashlib, algo)(word.encode()).hexdigest()
                if h == target_hash:
                    self.hash_result.config(text=f"Result: Found match → '{word}'")
                    return

        self.hash_result.config(text="Result: No match found")

    def detect_algorithm(self, hash_str):
        length = len(hash_str)
        if length == 32:
            return 'md5'
        elif length == 40:
            return 'sha1'
        elif length == 64:
            return 'sha256'
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = HALOSApp(root)
    root.mainloop()
