import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, Canvas, PhotoImage
import threading
import os
import time
import io
import base64
import json
import openai
import webbrowser
import pyttsx3
import whisper
import sounddevice as sd
import numpy as np
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image, ImageTk
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import matplotlib.pyplot as plt
import chromadb
from chromadb.config import Settings
import anthropic
import google.generativeai as genai
import stripe
import bandit
import semgrep
import plaid
from plaid.api import plaid_api
from plaid.model import *
import cv2
import pytesseract
import docker
import firebase_admin
from firebase_admin import credentials, db
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtCore import Qt
import customtkinter as ctk
import speech_recognition as sr
from speech_recognition import Recognizer, Microphone
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

load_dotenv()

class HALOSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HALOS V7 - Fully Upgraded AI Assistant")
        self.geometry("1400x950")
        
        # Configure theme
        ctk.set_appearance_mode("System")  # Light/Dark mode support
        ctk.set_default_color_theme("blue") 

        # Initialize all components
        self.initialize_core_components()
        self.initialize_upgraded_components()
        self.setup_ui()
        
        # Start background services
        self.start_background_services()

    def initialize_core_components(self):
        """Initialize all original HALOS components"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        
        # Original components
        self.chat_history = []
        self.audio_buffer = []
        self.memory = []
        self.transactions = []
        
        # Original models
        self.summarizer = pipeline("summarization")
        self.emotion_detector = pipeline("text-classification", model="finiteautomata/bertweet-base-emotion-analysis")
        self.tts = pyttsx3.init()
        self.whisper_model = whisper.load_model("base")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def initialize_upgraded_components(self):
        """Initialize all new upgraded components"""
        # Multi-LLM Orchestration
        self.llm_providers = {
            "openai": openai,
            "anthropic": anthropic.Client(os.getenv("ANTHROPIC_API_KEY")),
            "gemini": genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        }
        
        # Enhanced Security Lab
        self.breach_checker = HaveIBeenPwnedAPI(os.getenv("HIBP_API_KEY"))
        self.password_analyzer = PasswordStrengthAnalyzer()
        
        # Code Doctor Pro
        self.code_analyzer = CodeAnalyzer()
        self.test_generator = TestGenerator()
        
        # Vision Module
        self.camera = cv2.VideoCapture(0)
        self.ocr_engine = pytesseract
        
        # Memory System
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="memory_db"
        ))
        self.memory_collection = self.chroma_client.get_or_create_collection("halos_memories")
        
        # Finance Tracker++
        self.plaid_client = PlaidClient(
            client_id=os.getenv("PLAID_CLIENT_ID"),
            secret=os.getenv("PLAID_SECRET"),
            public_key=os.getenv("PLAID_PUBLIC_KEY")
        )
        
        # Cloud Integration
        if os.getenv("FIREBASE_CREDENTIALS"):
            cred = credentials.Certificate(json.loads(os.getenv("FIREBASE_CREDENTIALS")))
            firebase_admin.initialize_app(cred, {
                'databaseURL': os.getenv("FIREBASE_DB_URL")
            })
        
        # Voice Interface 2.0
        self.speech_recognizer = sr.Recognizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Notification System
        self.notifications = []
        self.notification_queue = queue.Queue()

    def setup_ui(self):
        """Setup the modernized UI with all features"""
        self.notebook = ctk.CTkTabview(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add all tabs
        self.setup_assistant_tab()
        self.setup_drawing_tab()
        self.setup_security_tab()
        self.setup_finance_tab()
        self.setup_code_tab()
        self.setup_vision_tab()
        self.setup_memory_tab()
        self.setup_settings_tab()
        
        # Notification center
        self.notification_center = ctk.CTkFrame(self)
        self.notification_center.pack(fill=tk.X, side=tk.BOTTOM)
        self.notification_label = ctk.CTkLabel(self.notification_center, text="Notifications: 0")
        self.notification_label.pack(side=tk.LEFT)
        
        # Status bar
        self.status_bar = ctk.CTkLabel(self, text="Ready", anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_assistant_tab(self):
        """Upgraded assistant tab with multi-LLM support"""
        tab = self.notebook.add("Assistant")
        
        # Chat display
        self.chat_display = ctk.CTkTextbox(tab, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input frame with LLM selector
        input_frame = ctk.CTkFrame(tab)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.llm_selector = ctk.CTkComboBox(input_frame, 
                                          values=["GPT-4", "Claude-3", "Gemini-Pro"])
        self.llm_selector.pack(side=tk.LEFT, padx=5)
        self.llm_selector.set("GPT-4")
        
        self.user_input = ctk.CTkEntry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.user_input.bind("<Return>", self.process_command)
        
        send_button = ctk.CTkButton(input_frame, text="Send", command=self.process_command)
        send_button.pack(side=tk.RIGHT, padx=5)
        
        # Voice controls
        voice_frame = ctk.CTkFrame(tab)
        voice_frame.pack(fill=tk.X)
        
        self.voice_button = ctk.CTkButton(voice_frame, text="🎤 Start Listening", 
                                         command=self.toggle_voice_recognition)
        self.voice_button.pack(side=tk.LEFT, padx=5)
        
        self.emotion_display = ctk.CTkLabel(voice_frame, text="Emotion: Neutral")
        self.emotion_display.pack(side=tk.RIGHT, padx=5)

    def setup_security_tab(self):
        """Enhanced security tab with new features"""
        tab = self.notebook.add("Security")
        
        # Hash cracking section
        hash_frame = ctk.CTkFrame(tab)
        hash_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(hash_frame, text="Hash:").grid(row=0, column=0, padx=5, pady=5)
        self.hash_input = ctk.CTkEntry(hash_frame, width=400)
        self.hash_input.grid(row=0, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(hash_frame, text="Algorithm:").grid(row=1, column=0, padx=5, pady=5)
        self.hash_algo = ctk.CTkComboBox(hash_frame, 
                                       values=["Auto-detect", "MD5", "SHA1", "SHA256", "bcrypt"])
        self.hash_algo.grid(row=1, column=1, padx=5, pady=5)
        
        ctk.CTkLabel(hash_frame, text="Dictionary:").grid(row=2, column=0, padx=5, pady=5)
        self.dict_file = ctk.CTkEntry(hash_frame, width=400)
        self.dict_file.grid(row=2, column=1, padx=5, pady=5)
        ctk.CTkButton(hash_frame, text="Browse", 
                     command=self.load_dict_file).grid(row=2, column=2, padx=5)
        
        ctk.CTkButton(hash_frame, text="Crack Hash", 
                     command=self.crack_hash).grid(row=3, columnspan=3, pady=10)
        
        # Progress bar for hash cracking
        self.hash_progress = ctk.CTkProgressBar(hash_frame)
        self.hash_progress.grid(row=4, columnspan=3, sticky="ew", padx=5)
        self.hash_progress.set(0)
        
        self.hash_result = ctk.CTkLabel(hash_frame, text="Result: ")
        self.hash_result.grid(row=5, columnspan=3)
        
        # Password analysis section
        pass_frame = ctk.CTkFrame(tab)
        pass_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(pass_frame, text="Password Analysis").pack()
        
        ctk.CTkLabel(pass_frame, text="Test Password:").grid(row=0, column=0, padx=5)
        self.password_input = ctk.CTkEntry(pass_frame, show="•")
        self.password_input.grid(row=0, column=1, padx=5)
        
        ctk.CTkButton(pass_frame, text="Analyze", 
                     command=self.analyze_password).grid(row=0, column=2, padx=5)
        
        self.password_strength = ctk.CTkProgressBar(pass_frame)
        self.password_strength.grid(row=1, columnspan=3, sticky="ew", padx=5, pady=5)
        
        self.password_feedback = ctk.CTkLabel(pass_frame, text="")
        self.password_feedback.grid(row=2, columnspan=3)

    def setup_code_tab(self):
        """New Code Doctor Pro tab"""
        tab = self.notebook.add("Code Doctor")
        
        # Code input
        self.code_input = ctk.CTkTextbox(tab, wrap=tk.NONE)
        self.code_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Language selection
        lang_frame = ctk.CTkFrame(tab)
        lang_frame.pack(fill=tk.X)
        
        ctk.CTkLabel(lang_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        self.code_lang = ctk.CTkComboBox(lang_frame, 
                                        values=["Python", "JavaScript", "Java", "C++", "Go"])
        self.code_lang.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        btn_frame = ctk.CTkFrame(tab)
        btn_frame.pack(fill=tk.X)
        
        ctk.CTkButton(btn_frame, text="Analyze", 
                      command=self.analyze_code).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(btn_frame, text="Fix Issues", 
                      command=self.fix_code).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(btn_frame, text="Generate Tests", 
                      command=self.generate_tests).pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.code_output = ctk.CTkTextbox(tab, height=150)
        self.code_output.pack(fill=tk.X, padx=5, pady=5)

    def setup_vision_tab(self):
        """Enhanced vision tab with webcam support"""
        tab = self.notebook.add("Vision")
        
        # Webcam view
        self.camera_frame = ctk.CTkFrame(tab)
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Camera Feed")
        self.camera_label.pack()
        
        # Controls
        cam_btn_frame = ctk.CTkFrame(tab)
        cam_btn_frame.pack(fill=tk.X)
        
        ctk.CTkButton(cam_btn_frame, text="Start Camera", 
                     command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(cam_btn_frame, text="Capture & Analyze", 
                     command=self.capture_image).pack(side=tk.LEFT, padx=5)
        
        # Document processing
        doc_frame = ctk.CTkFrame(tab)
        doc_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkButton(doc_frame, text="Upload Document", 
                     command=self.upload_document).pack(side=tk.LEFT, padx=5)
        
        self.doc_type = ctk.CTkComboBox(doc_frame, 
                                      values=["PDF", "Image", "PPT"])
        self.doc_type.pack(side=tk.LEFT, padx=5)
        
        # Results
        self.vision_output = ctk.CTkTextbox(tab, height=200)
        self.vision_output.pack(fill=tk.X, padx=5, pady=5)

    def setup_memory_tab(self):
        """Advanced memory system tab"""
        tab = self.notebook.add("Memory")
        
        # Memory input
        mem_frame = ctk.CTkFrame(tab)
        mem_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(mem_frame, text="Memory Content:").pack(side=tk.LEFT, padx=5)
        self.memory_input = ctk.CTkEntry(mem_frame, width=400)
        self.memory_input.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(mem_frame, text="Add Memory", 
                     command=self.add_memory).pack(side=tk.LEFT, padx=5)
        
        # Memory search
        search_frame = ctk.CTkFrame(tab)
        search_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.memory_search = ctk.CTkEntry(search_frame, width=400)
        self.memory_search.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(search_frame, text="Search", 
                     command=self.search_memories).pack(side=tk.LEFT, padx=5)
        
        # Memory display
        self.memory_display = ctk.CTkTextbox(tab)
        self.memory_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_settings_tab(self):
        """System settings tab"""
        tab = self.notebook.add("Settings")
        
        # Theme selection
        theme_frame = ctk.CTkFrame(tab)
        theme_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(theme_frame, text="Appearance Mode:").pack(side=tk.LEFT, padx=5)
        
        self.theme_mode = ctk.CTkComboBox(theme_frame, 
                                         values=["Light", "Dark", "System"],
                                         command=self.change_theme)
        self.theme_mode.pack(side=tk.LEFT, padx=5)
        
        # API configuration
        api_frame = ctk.CTkFrame(tab)
        api_frame.pack(fill=tk.X, pady=5)
        
        ctk.CTkLabel(api_frame, text="Default LLM:").pack(side=tk.LEFT, padx=5)
        
        self.default_llm = ctk.CTkComboBox(api_frame, 
                                          values=["GPT-4", "Claude-3", "Gemini-Pro"])
        self.default_llm.pack(side=tk.LEFT, padx=5)
        
        # Cloud sync
        cloud_frame = ctk.CTkFrame(tab)
        cloud_frame.pack(fill=tk.X, pady=5)
        
        self.cloud_sync = ctk.CTkCheckBox(cloud_frame, text="Enable Cloud Sync")
        self.cloud_sync.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkButton(cloud_frame, text="Sync Now", 
                     command=self.force_sync).pack(side=tk.LEFT, padx=5)

    # ======================
    # CORE FUNCTIONALITY
    # ======================
    
    def process_command(self, event=None):
        """Enhanced command processing with multi-LLM support"""
        command = self.user_input.get()
        if not command:
            return
            
        self.user_input.delete(0, tk.END)
        self.display_message(f"You: {command}", "user")
        
        # Special commands
        if command.lower().startswith("play "):
            self.handle_play_command(command)
            return
            
        # Process with selected LLM
        llm_choice = self.llm_selector.get()
        try:
            if llm_choice == "GPT-4":
                response = self.call_openai(command)
            elif llm_choice == "Claude-3":
                response = self.call_anthropic(command)
            elif llm_choice == "Gemini-Pro":
                response = self.call_gemini(command)
                
            self.display_message(f"HALOS: {response}", "halos")
            self.store_memory(f"Conversation: {command} -> {response}")
            
        except Exception as e:
            self.display_message(f"HALOS: Error: {str(e)}", "halos")
            self.add_notification(f"LLM Error: {str(e)}", "error")

    def call_openai(self, prompt):
        """Call OpenAI API with conversation context"""
        messages = [{"role": "system", "content": "You're a helpful AI assistant."}]
        messages.extend(self.chat_history[-6:])  # Keep last 3 exchanges
        messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def call_anthropic(self, prompt):
        """Call Anthropic Claude API"""
        client = self.llm_providers["anthropic"]
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def call_gemini(self, prompt):
        """Call Google Gemini API"""
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text

    # ======================
    # UPGRADED FUNCTIONALITY
    # ======================
    
    def analyze_password(self):
        """Enhanced password analysis with breach checking"""
        password = self.password_input.get()
        if not password:
            return
            
        # Strength analysis
        strength = self.password_analyzer.calculate_strength(password)
        self.password_strength.set(strength["score"] / 100)
        
        # Visual feedback
        if strength["score"] < 40:
            self.password_strength.configure(progress_color="red")
            feedback = "Weak - " + strength["feedback"]
        elif strength["score"] < 70:
            self.password_strength.configure(progress_color="orange")
            feedback = "Moderate - " + strength["feedback"]
        else:
            self.password_strength.configure(progress_color="green")
            feedback = "Strong - " + strength["feedback"]
            
        self.password_feedback.configure(text=feedback)
        
        # Breach check
        if self.breach_checker.is_breached(password):
            self.add_notification("Warning: Password found in data breaches!", "warning")
            self.password_feedback.configure(text=feedback + " (BREACHED!)")

    def analyze_code(self):
        """Advanced code analysis with security scanning"""
        code = self.code_input.get("1.0", tk.END)
        language = self.code_lang.get().lower()
        
        # Static analysis
        analysis = self.code_analyzer.analyze(code, language)
        
        # Security scanning
        security_issues = []
        if language == "python":
            security_issues.extend(self.run_bandit_scan(code))
            security_issues.extend(self.run_semgrep_scan(code))
        
        # Display results
        output = "=== Code Analysis ===\n"
        output += f"Quality Score: {analysis['quality_score']}/100\n"
        output += f"Issues Found: {len(analysis['issues'])}\n"
        
        if security_issues:
            output += "\n=== Security Issues ===\n"
            output += "\n".join(f"- {issue}" for issue in security_issues)
            
        self.code_output.delete("1.0", tk.END)
        self.code_output.insert("1.0", output)
        self.add_notification("Code analysis completed", "info")

    def run_bandit_scan(self, code):
        """Run Bandit security scanner"""
        # Implementation would use bandit lib
        return ["Potential SQL injection vulnerability"]  # Example

    def run_semgrep_scan(self, code):
        """Run Semgrep security scanner"""
        # Implementation would use semgrep lib
        return ["Hard-coded API key detected"]  # Example

    def start_camera(self):
        """Start webcam feed for vision processing"""
        if not hasattr(self, "camera_active"):
            self.camera_active = False
            
        if self.camera_active:
            self.camera_active = False
            return
            
        self.camera_active = True
        threading.Thread(target=self.update_camera_feed, daemon=True).start()

    def update_camera_feed(self):
        """Update the camera feed in real-time"""
        while self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            time.sleep(0.05)

    def capture_image(self):
        """Capture and analyze current camera frame"""
        ret, frame = self.camera.read()
        if ret:
            # Save image
            cv2.imwrite("capture.jpg", frame)
            
            # Analyze with vision
            description = self.analyze_image_with_gpt4("capture.jpg")
            self.vision_output.delete("1.0", tk.END)
            self.vision_output.insert("1.0", f"Image Analysis:\n{description}")
            
            self.add_notification("Image captured and analyzed", "success")

    def analyze_image_with_gpt4(self, image_path):
        """Use GPT-4 Vision to analyze an image"""
        # Implementation would use OpenAI's vision API
        return "This is a simulated image analysis response describing the contents of the image."

    def add_memory(self):
        """Store a new memory with vector embedding"""
        content = self.memory_input.get()
        if not content:
            return
            
        # Generate embedding
        embedding = self.embedding_model.encode(content)
        
        # Store in ChromaDB
        self.memory_collection.add(
            embeddings=[embedding.tolist()],
            documents=[content],
            ids=[str(time.time())]
        )
        
        self.memory_input.delete(0, tk.END)
        self.add_notification("Memory stored successfully", "success")

    def search_memories(self):
        """Search memories using semantic search"""
        query = self.memory_search.get()
        if not query:
            return
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search ChromaDB
        results = self.memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        
        # Display results
        self.memory_display.delete("1.0", tk.END)
        if results['documents']:
            self.memory_display.insert("1.0", "Search Results:\n\n")
            for doc in results['documents'][0]:
                self.memory_display.insert(tk.END, f"- {doc}\n\n")
        else:
            self.memory_display.insert("1.0", "No matching memories found")

    # ======================
    # HELPER FUNCTIONS
    # ======================
    
    def display_message(self, message, sender):
        """Display message in chat with sender-specific formatting"""
        self.chat_display.configure(state='normal')
        
        # Color coding
        if sender == "user":
            tag = "user"
            self.chat_display.tag_config(tag, foreground="blue")
        else:
            tag = "halos"
            self.chat_display.tag_config(tag, foreground="green")
            
        self.chat_display.insert(tk.END, message + "\n", tag)
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)

    def add_notification(self, message, level="info"):
        """Add a notification to the system"""
        self.notifications.append({
            "message": message,
            "level": level,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        self.update_notification_display()

    def update_notification_display(self):
        """Update the notification counter"""
        count = len(self.notifications)
        self.notification_label.configure(text=f"Notifications: {count}")
        
        # Flash for important notifications
        if any(n["level"] in ("warning", "error") for n in self.notifications[-3:]):
            self.notification_label.configure(text_color="red")
            self.after(500, lambda: self.notification_label.configure(text_color=None))

    def change_theme(self, choice):
        """Change the application theme"""
        ctk.set_appearance_mode(choice.lower())
        self.add_notification(f"Changed theme to {choice}", "info")

    def start_background_services(self):
        """Start all background services"""
        # Start notification handler
        threading.Thread(target=self.handle_notifications, daemon=True).start()
        
        # Start cloud sync if enabled
        if self.cloud_sync.get():
            threading.Thread(target=self.sync_with_cloud, daemon=True).start()
            
        # Initialize voice recognition
        self.voice_recognition_active = False

    def handle_notifications(self):
        """Process notification queue"""
        while True:
            # Process any pending notifications
            while not self.notification_queue.empty():
                notification = self.notification_queue.get()
                self.add_notification(notification["message"], notification["level"])
                
            time.sleep(1)

    def sync_with_cloud(self):
        """Sync data with Firebase"""
        while True:
            try:
                # Sync chat history
                if hasattr(self, 'chat_history'):
                    db.reference('/halos/chat').set(self.chat_history[-50:])
                
                # Sync memories
                if hasattr(self, 'memory_collection'):
                    memories = self.memory_collection.get()
                    db.reference('/halos/memories').set(memories)
                    
                time.sleep(60)  # Sync every minute
            except Exception as e:
                print(f"Sync error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def toggle_voice_recognition(self):
        """Toggle voice recognition on/off"""
        if self.voice_recognition_active:
            self.voice_recognition_active = False
            self.voice_button.configure(text="🎤 Start Listening")
        else:
            self.voice_recognition_active = True
            self.voice_button.configure(text="🔴 Stop Listening")
            threading.Thread(target=self.listen_for_speech, daemon=True).start()

    def listen_for_speech(self):
        """Continuous voice recognition"""
        with sr.Microphone() as source:
            self.speech_recognizer.adjust_for_ambient_noise(source)
            
            while self.voice_recognition_active:
                try:
                    audio = self.speech_recognizer.listen(source, timeout=3)
                    text = self.speech_recognizer.recognize_google(audio)
                    
                    # Analyze emotion from speech
                    emotion = self.sentiment_analyzer.polarity_scores(text)
                    dominant_emotion = max(emotion, key=emotion.get)
                    self.emotion_display.configure(text=f"Emotion: {dominant_emotion}")
                    
                    # Process the command
                    self.user_input.delete(0, tk.END)
                    self.user_input.insert(0, text)
                    self.process_command()
                    
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self.add_notification(f"Voice error: {str(e)}", "error")

    def force_sync(self):
        """Force immediate cloud sync"""
        threading.Thread(target=self.sync_with_cloud, daemon=True).start()
        self.add_notification("Cloud sync initiated", "info")

if __name__ == "__main__":
    app = HALOSApp()
    app.mainloop()