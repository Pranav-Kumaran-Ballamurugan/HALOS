import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, Canvas
import threading
import os
import time
import io
import base64
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
from llama_cpp import Llama
from PIL import Image
import random
import chromadb
from typing import List, Dict, Optional

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler('halos.log'), 
                            logging.StreamHandler()])

class Config:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.stripe_key = os.getenv("STRIPE_LIVE_KEY")
        self.llama_path = os.getenv("LLAMA_PATH", "models/llama-3-8b-instruct.Q4_K_M.gguf")
        self.validate()
        
    def validate(self):
        if not self.openai_key:
            raise ValueError("OpenAI API key missing")

# --- Core AI Systems ---
class VisionAnalyzer:
    def __init__(self):
        self.client = openai.Client()
        
    def analyze_image(self, image: Image.Image) -> str:
        """Analyze drawings using GPT-4 Vision"""
        try:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this drawing in detail"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }}
                    ]
                }],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Vision analysis failed: {str(e)}")
            return f"Couldn't analyze drawing: {str(e)}"

class AIModel:
    def __init__(self):
        self.config = Config()
        self.local_llm = None
        self.init_local_llm()
        
    def init_local_llm(self):
        try:
            if os.path.exists(self.config.llama_path):
                self.local_llm = Llama(
                    model_path=self.config.llama_path,
                    n_ctx=2048,
                    n_threads=4
                )
                logging.info("Local LLM initialized")
        except Exception as e:
            logging.warning(f"Local LLM failed: {str(e)}")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response with local/cloud fallback"""
        try:
            if self.local_llm:
                response = self.local_llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response['choices'][0]['message']['content']
                
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"AI generation failed: {str(e)}")
            return "I'm having trouble generating a response right now."

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
        
    def search(self, query: str, n_results: int = 3) -> Dict:
        query_embed = self.embedder.encode(query)
        return self.collection.query(
            query_embeddings=[query_embed.tolist()],
            n_results=n_results
        )

# --- Main Application ---
class HALOSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HALOS V5.0 ★")
        self.geometry("1400x1000")
        self.config = Config()
        
        # Core systems
        self.ai = AIModel()
        self.memory = AIMemory()
        self.vision = VisionAnalyzer()
        self.emotion = pipeline("text-classification", 
                              model="finiteautomata/bertweet-base-emotion-analysis")
        
        # UI state
        self.high_contrast_mode = False
        self.dyslexia_font_enabled = False
        self.transactions = []
        self.categories = {
            'Food': ['restaurant', 'groceries', 'coffee'],
            'Transport': ['uber', 'gas', 'parking'],
            'Utilities': ['electric', 'water', 'internet']
        }
        
        self.setup_ui()
        self.setup_services()
        
    def setup_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Setup all tabs
        self.setup_assistant_tab()
        self.setup_codefixer_tab()
        self.setup_payments_tab()
        self.setup_security_tab()
        self.setup_learning_tab()
        self.setup_drawing_tab()
        self.setup_finance_tab()
        self.setup_accessibility_tab()
        
        # Status bar
        self.status = ttk.Label(self, text="Ready ★", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X)
    
    # [Previous tab setup methods with these key enhancements...]
    
    def setup_finance_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="💰 Finance Pro")
        
        # Input frame
        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(input_frame, text="Description:").grid(row=0, column=0, sticky=tk.W)
        self.expense_desc = ttk.Entry(input_frame, width=40)
        self.expense_desc.grid(row=0, column=1)
        
        ttk.Label(input_frame, text="Amount:").grid(row=0, column=2, padx=5)
        self.expense_amount = ttk.Entry(input_frame, width=10)
        self.expense_amount.grid(row=0, column=3)
        
        ttk.Label(input_frame, text="Category:").grid(row=0, column=4, padx=5)
        self.expense_category = ttk.Combobox(input_frame, 
                                           values=list(self.categories.keys()))
        self.expense_category.grid(row=0, column=5)
        
        # Button frame
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add", command=self.add_transaction).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Auto-Categorize", 
                  command=self.auto_categorize).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Generate Report", 
                  command=self.generate_report).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Visualize", 
                  command=self.visualize_finances).pack(side=tk.LEFT, padx=5)
        
        # Output
        self.finance_output = scrolledtext.ScrolledText(tab)
        self.finance_output.pack(fill=tk.BOTH, expand=True)
    
    def auto_categorize(self):
        desc = self.expense_desc.get().lower()
        for cat, keywords in self.categories.items():
            if any(kw in desc for kw in keywords):
                self.expense_category.set(cat)
                return
        self.expense_category.set("Other")
    
    def add_transaction(self):
        try:
            amount = float(self.expense_amount.get())
            desc = self.expense_desc.get()
            category = self.expense_category.get() or "Uncategorized"
            
            self.transactions.append({
                "timestamp": datetime.now().isoformat(),
                "description": desc,
                "amount": amount,
                "category": category
            })
            
            self.finance_output.insert(tk.END, 
                f"Added: ${amount:.2f} - {desc} ({category})\n")
            
            # Clear inputs
            self.expense_desc.delete(0, tk.END)
            self.expense_amount.delete(0, tk.END)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid amount")
    
    def generate_report(self):
        if not self.transactions:
            return
            
        # Prepare data for AI
        transactions_str = "\n".join(
            f"{t['timestamp']} | ${t['amount']:.2f} | {t['category']} | {t['description']}"
            for t in self.transactions
        )
        
        prompt = f"""Analyze these financial transactions:
        {transactions_str}
        
        Provide:
        1. Spending by category breakdown
        2. Weekly/Monthly patterns
        3. Budgeting recommendations
        4. Any unusual spending flags"""
        
        try:
            report = self.ai.generate(prompt, max_tokens=800)
            self.finance_output.insert(tk.END, "\n=== AI FINANCIAL REPORT ===\n")
            self.finance_output.insert(tk.END, report)
            self.finance_output.insert(tk.END, "\n" + "="*40 + "\n")
        except Exception as e:
            self.finance_output.insert(tk.END, f"Error generating report: {str(e)}\n")
    
    def visualize_finances(self):
        # This would use matplotlib in production
        by_category = {}
        for t in self.transactions:
            cat = t['category']
            by_category[cat] = by_category.get(cat, 0) + t['amount']
            
        total = sum(by_category.values())
        visualization = "\n".join(
            f"{cat}: ${amt:.2f} ({amt/total:.1%})" 
            for cat, amt in sorted(by_category.items(), key=lambda x: -x[1])
        )
        
        self.finance_output.insert(tk.END, "\n=== SPENDING VISUALIZATION ===\n")
        self.finance_output.insert(tk.END, visualization)
        self.finance_output.insert(tk.END, "\n" + "="*40 + "\n")
    
    def setup_drawing_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🎨 Drawing+")
        
        # Drawing canvas
        self.canvas = Canvas(tab, bg="white", width=800, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Controls
        ctrl_frame = ttk.Frame(tab)
        ctrl_frame.pack(fill=tk.X)
        
        ttk.Button(ctrl_frame, text="Clear", 
                  command=self.clear_canvas).pack(side=tk.LEFT)
        ttk.Button(ctrl_frame, text="Analyze", 
                  command=self.analyze_drawing).pack(side=tk.LEFT, padx=5)
        
        # Output
        self.drawing_output = scrolledtext.ScrolledText(tab, height=10)
        self.drawing_output.pack(fill=tk.BOTH, expand=True)
    
    def paint(self, event):
        x, y = event.x, event.y
        r = 3  # radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing_output.delete(1.0, tk.END)
    
    def analyze_drawing(self):
        try:
            # Convert canvas to image
            ps = self.canvas.postscript(colormode='color')
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            
            # Get AI analysis
            analysis = self.vision.analyze_image(img)
            
            # Store and display
            self.memory.store(f"DRAWING: {analysis}")
            self.drawing_output.delete(1.0, tk.END)
            self.drawing_output.insert(tk.END, "AI Analysis:\n" + analysis)
            
        except Exception as e:
            self.drawing_output.insert(tk.END, f"Error: {str(e)}\n")
    
    # [Other tab setups remain similar but use self.ai.generate()]
    
    def setup_services(self):
        # Audio processing
        def audio_callback(indata, frames, time, status):
            self.audio_buffer.add_data(indata[:, 0])
            
        self.audio_buffer = AudioBuffer()
        self.audio_thread = threading.Thread(
            target=sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=16000
            ).start,
            daemon=True
        )
        self.audio_thread.start()
        
        # Memory consolidation
        self.memory_thread = threading.Thread(
            target=self.memory_consolidation,
            daemon=True
        )
        self.memory_thread.start()
    
    def memory_consolidation(self):
        while True:
            time.sleep(60)  # Every minute
            if hasattr(self, 'chat_history') and self.chat_history:
                last_msg = self.chat_history[-1]
                self.memory.store(
                    f"CONVERSATION: {last_msg['role']}: {last_msg['content']}"
                )

if __name__ == "__main__":
    try:
        app = HALOSApp()
        app.mainloop()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        raise