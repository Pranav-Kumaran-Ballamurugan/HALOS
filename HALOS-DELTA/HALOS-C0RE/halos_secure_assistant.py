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
import logging
from collections import deque

load_dotenv()

# Configure logging
logging.basicConfig(
    filename='halos.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RateLimiter:
    def __init__(self, max_calls, time_frame):
        self.max_calls = max_calls
        self.time_frame = time_frame
        self.timestamps = deque()

    def __call__(self):
        now = time.time()
        while self.timestamps and now - self.timestamps[0] > self.time_frame:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_calls:
            return False
        self.timestamps.append(now)
        return True

CONFIG = {
    "GPT_MODEL": "gpt-4-turbo"
}

class HALOSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HALOS AI Assistant")
        self.root.geometry("1000x700")
        
        self.logger = logging.getLogger('HALOS')
        self.rate_limiter = RateLimiter(max_calls=5, time_frame=60)  # 5 requests per minute
        
        # Load API key securely
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            del os.environ["OPENAI_API_KEY"]  # Remove from environment after loading
        else:
            self.logger.error("OpenAI API key not found in environment variables")
            messagebox.showerror("Error", "OpenAI API key not configured")
            self.root.destroy()
            return

        self.authenticated = True
        self.setup_ui()
        self.logger.info("HALOS application initialized")

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

        self.progress = ttk.Progressbar(hash_frame, mode='determinate')
        self.progress.grid(row=4, columnspan=3, sticky=tk.EW, pady=5)

    def process_command(self, event=None):
        command = self.user_input.get().strip()
        if not command:
            return
            
        self.user_input.delete(0, tk.END)
        self.display_message(f"You: {command}", "user")
        
        if not self.rate_limiter():
            self.display_message("HALOS: Rate limit exceeded - please wait", "halos")
            return
            
        if command.lower().startswith("play "):
            search = command[5:].strip().replace(" ", "+")
            url = f"https://www.youtube.com/results?search_query={search}"
            webbrowser.open(url)
            self.display_message(f"HALOS: 🎵 Searching YouTube for: {command[5:].strip()}", "halos")
            return

        try:
            response = openai.ChatCompletion.create(
                model=CONFIG["GPT_MODEL"],
                messages=[
                    {"role": "system", "content": "You're a helpful AI assistant."},
                    {"role": "user", "content": command}
                ],
                timeout=10
            )
            reply = response.choices[0].message.content.strip()
            self.display_message(f"HALOS: {reply}", "halos")
            self.logger.info(f"Processed command: {command}")
        except openai.error.AuthenticationError:
            self.display_message("HALOS: Error: Invalid API key", "halos")
            self.logger.error("OpenAI authentication failed")
        except openai.error.RateLimitError:
            self.display_message("HALOS: Error: Rate limit exceeded", "halos")
            self.logger.warning("OpenAI rate limit exceeded")
        except openai.error.Timeout:
            self.display_message("HALOS: Error: Request timed out", "halos")
            self.logger.warning("OpenAI request timeout")
        except Exception as e:
            error_msg = f"HALOS: Error: {str(e)}"
            self.display_message(error_msg, "halos")
            self.logger.error(f"Command processing failed: {str(e)}")

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
            self.logger.info(f"Selected dictionary file: {path}")

    def crack_hash(self):
        target_hash = self.hash_input.get().strip().lower()
        dict_file = self.dict_path.get().strip()

        if not target_hash:
            self.hash_result.config(text="Result: Please enter a hash")
            return
        if not dict_file:
            self.hash_result.config(text="Result: Please select a dictionary file")
            return
        if not os.path.exists(dict_file):
            self.hash_result.config(text="Result: Dictionary file not found")
            self.logger.error(f"Dictionary file not found: {dict_file}")
            return

        threading.Thread(
            target=self._crack_hash_thread,
            args=(target_hash, dict_file),
            daemon=True
        ).start()

    def _crack_hash_thread(self, target_hash, dict_file):
        self.root.after(0, lambda: self.hash_result.config(text="Processing..."))
        self.root.after(0, lambda: self.progress.config(value=0))

        try:
            algo = self.detect_algorithm(target_hash)
            if not algo:
                self.root.after(0, lambda: self.hash_result.config(
                    text="Result: Unknown hash algorithm"))
                self.logger.warning(f"Unknown hash algorithm for: {target_hash}")
                return

            total_lines = sum(1 for _ in open(dict_file, 'r', encoding='utf-8'))
            if total_lines == 0:
                self.root.after(0, lambda: self.hash_result.config(
                    text="Result: Empty dictionary file"))
                return

            processed = 0
            start_time = time.time()

            with open(dict_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    h = getattr(hashlib, algo)(word.encode()).hexdigest()
                    if h == target_hash:
                        elapsed = time.time() - start_time
                        self.root.after(0, lambda: self.hash_result.config(
                            text=f"Result: Found match → '{word}' (in {elapsed:.2f}s)"))
                        self.logger.info(f"Hash cracked: {target_hash} → {word}")
                        return

                    processed += 1
                    if processed % 1000 == 0:
                        progress = (processed/total_lines)*100
                        self.root.after(0, lambda: self.progress.config(value=progress))
                        self.root.after(0, lambda: self.hash_result.config(
                            text=f"Checking... {progress:.1f}%"))

            self.root.after(0, lambda: self.hash_result.config(
                text="Result: No match found"))
            self.logger.info(f"No match found for hash: {target_hash}")

        except Exception as e:
            self.root.after(0, lambda: self.hash_result.config(
                text=f"Result: Error - {str(e)}"))
            self.logger.error(f"Hash cracking failed: {str(e)}")

    def detect_algorithm(self, hash_str):
        hash_str = hash_str.strip().lower()
        length = len(hash_str)

        algorithms = {
            32: ['md5', 'md4', 'md2'],
            40: ['sha1', 'ripemd160'],
            56: ['sha224'],
            64: ['sha256', 'sha3-256'],
            96: ['sha384', 'sha3-384'],
            128: ['sha512', 'sha3-512']
        }

        if length in algorithms:
            for algo in algorithms[length]:
                if hasattr(hashlib, algo):
                    return algo
        return None

if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = HALOSApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application crashed: {str(e)}")
        raise
