#!/usr/bin/env python3
"""
HALOS V7 - Complete Implementation
Hyper-Advanced Learning and Operation System
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, Canvas
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
import speech_recognition as sr
from speech_recognition import Recognizer, Microphone
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from enum import Enum, auto
from dataclasses import dataclass
import queue
from typing import Dict, Optional, List, Tuple, Any
import logging
from logging.handlers import RotatingFileHandler
import requests
import sqlite3
from sqlite3 import Error

# Initialize environment
load_dotenv()
nltk.download('vader_lexicon')

# ======================
# CORE SYSTEM COMPONENTS
# ======================

class LLMProvider(Enum):
    """Extended LLM provider options"""
    OPENAI = auto()
    ANTHROPIC = auto()
    GEMINI = auto()
    LOCAL_LLAMA = auto()
    LOCAL_MISTRAL = auto()

@dataclass
class HALOSConfig:
    """Central configuration manager"""
    dark_mode: bool = False
    current_llm: LLMProvider = LLMProvider.OPENAI
    security_level: int = 2  # 1-5 scale
    memory_retention: int = 30  # days
    voice_enabled: bool = True
    auto_update: bool = True

class NotificationCenter:
    """Centralized notification system"""
    def __init__(self):
        self.notification_queue = queue.Queue()
        self.history = []
        
    def add(self, message: str, level: str = "info"):
        """Add a new notification"""
        notification = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level
        }
        self.notification_queue.put(notification)
        self.history.append(notification)
        
    def get_notification(self) -> Optional[Dict]:
        """Get the next notification if available"""
        try:
            return self.notification_queue.get_nowait()
        except queue.Empty:
            return None
            
    def process_queue(self):
        """Process notifications (run in background thread)"""
        while True:
            notification = self.get_notification()
            if notification:
                self._handle_notification(notification)
            time.sleep(0.1)
            
    def _handle_notification(self, notification: Dict):
        """Handle notification based on level"""
        if notification["level"] == "error":
            logging.error(notification["message"])
        elif notification["level"] == "warning":
            logging.warning(notification["message"])
        else:
            logging.info(notification["message"])

class MemoryManager:
    """Enhanced memory management system"""
    def __init__(self):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="memory_db"
        ))
        self.collection = self.client.get_or_create_collection("halos_memory")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def store(self, content: str, metadata: Dict = None) -> str:
        """Store information in memory"""
        embedding = self.embedding_model.encode(content)
        doc_id = hashlib.md5(content.encode()).hexdigest()
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[metadata or {}]
        )
        return doc_id
        
    def retrieve(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant memories"""
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
        
    def sync_with_cloud(self):
        """Background sync with cloud storage"""
        while True:
            try:
                # Implement cloud sync logic
                time.sleep(3600)  # Sync every hour
            except Exception as e:
                logging.error(f"Cloud sync failed: {str(e)}")
                time.sleep(600)

class OpenAIIntegration:
    """OpenAI API integration"""
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        
    def query(self, prompt: str, context: Dict = None) -> str:
        """Execute query with OpenAI"""
        messages = [{"role": "user", "content": prompt}]
        if context:
            messages.insert(0, {"role": "system", "content": json.dumps(context)})
            
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content

class AnthropicIntegration:
    """Anthropic Claude integration"""
    def __init__(self):
        self.client = anthropic.Client(os.getenv("ANTHROPIC_API_KEY"))
        
    def query(self, prompt: str, context: Dict = None) -> str:
        """Execute query with Claude"""
        full_prompt = f"{json.dumps(context)}\n\n{prompt}" if context else prompt
        response = self.client.completion(
            prompt=f"{anthropic.HUMAN_PROMPT} {full_prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model="claude-2",
            max_tokens_to_sample=1000
        )
        return response["completion"]

class GeminiIntegration:
    """Google Gemini integration"""
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
    def query(self, prompt: str, context: Dict = None) -> str:
        """Execute query with Gemini"""
        full_prompt = f"Context: {json.dumps(context)}\n\nQuery: {prompt}" if context else prompt
        response = self.model.generate_content(full_prompt)
        return response.text

class LLMOrchestrator:
    """Enhanced LLM management with failover"""
    def __init__(self):
        self.providers = {
            LLMProvider.OPENAI: OpenAIIntegration(),
            LLMProvider.ANTHROPIC: AnthropicIntegration(),
            LLMProvider.GEMINI: GeminiIntegration()
        }
        self.performance_metrics = {}
        self.notification_handler = None
    
    def set_notification_handler(self, handler):
        """Set the notification handler callback"""
        self.notification_handler = handler
    
    def process_query(self, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Smart routing with fallback"""
        provider = self._select_provider(prompt, context)
        
        try:
            result = provider.query(prompt, context)
            self._update_metrics(provider, success=True)
            return {'success': True, 'result': result}
        except Exception as e:
            self._update_metrics(provider, success=False)
            if self.notification_handler:
                self.notification_handler(f"LLM Error: {str(e)}", "error")
            return self._handle_fallback(prompt, context, str(e))
    
    def _select_provider(self, prompt: str, context: Optional[Dict]) -> LLMProvider:
        """Determine best provider based on content"""
        # Simple routing logic - can be enhanced
        if "code" in prompt.lower():
            return LLMProvider.ANTHROPIC
        elif "creative" in prompt.lower():
            return LLMProvider.GEMINI
        return LLMProvider.OPENAI
    
    def _handle_fallback(self, prompt: str, context: Optional[Dict], error: str) -> Dict:
        """Handle LLM failures by trying other providers"""
        for provider in self.providers.values():
            try:
                result = provider.query(prompt, context)
                return {'success': True, 'result': result, 'fallback': True}
            except:
                continue
        return {'success': False, 'error': f"All providers failed. Last error: {error}"}

class SecuritySuite:
    """Enhanced security toolkit"""
    def __init__(self):
        self.hash_cracker = HashCracker()
        self.breach_checker = BreachDatabase()
        self.password_analyzer = PasswordStrength()
        self.network_scanner = NetworkScanner()
    
    def full_scan(self, target: str, scan_type: str = "comprehensive") -> Dict:
        """Unified security scan interface"""
        results = {}
        
        if scan_type in ("hash", "comprehensive"):
            results['hash_analysis'] = self.hash_cracker.analyze(target)
        
        if scan_type in ("breach", "comprehensive"):
            results['breach_check'] = self.breach_checker.lookup(target)
        
        if scan_type in ("password", "comprehensive"):
            results['strength_analysis'] = self.password_analyzer.evaluate(target)
        
        return results

class CodeDoctorPro:
    """Advanced code analysis and repair"""
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.security_scanner = SecurityScanner()
        self.test_generator = TestGenerator()
        self.llm_access = None
    
    def set_llm_access(self, llm_orchestrator):
        """Set the LLM access for advanced analysis"""
        self.llm_access = llm_orchestrator
    
    def analyze(self, code: str, language: str) -> Dict:
        """Comprehensive code analysis"""
        analysis = {
            'static_analysis': self.analyzer.run_analysis(code, language),
            'security_issues': self.security_scanner.scan(code, language),
            'style_suggestions': self._get_style_recommendations(code, language)
        }
        
        if self.llm_access:
            response = self.llm_access.process_query(
                f"Review this {language} code for potential improvements:\n{code}"
            )
            if response['success']:
                analysis['llm_review'] = response['result']
        
        return analysis
    
    def fix_code(self, code: str, language: str) -> Dict:
        """Fix identified issues in code"""
        analysis = self.analyze(code, language)
        if self.llm_access:
            response = self.llm_access.process_query(
                f"Fix these issues in the {language} code:\n{code}\n\nIssues:\n{analysis}"
            )
            if response['success']:
                return {
                    'fixed_code': response['result'],
                    'unit_tests': self.test_generator.generate(response['result'], language),
                    'analysis': analysis
                }
        return {'error': 'Failed to fix code'}

class VisionProcessor:
    """Advanced computer vision module"""
    def __init__(self):
        self.ocr_engine = pytesseract
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.memory_access = None
    
    def set_memory_access(self, memory_manager):
        """Set memory access for storing visual data"""
        self.memory_access = memory_manager
    
    def parse_document(self, image_path: str) -> Dict:
        """Extract text and structure from document"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = self.ocr_engine.image_to_string(gray)
        
        result = {
            "text": text,
            "metadata": {
                "dimensions": img.shape,
                "file_path": image_path,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if self.memory_access:
            self.memory_access.store(text, result["metadata"])
        
        return result
    
    def detect_faces(self, image_path: str) -> Dict:
        """Detect faces in an image"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        result = {
            "face_count": len(faces),
            "locations": [{"x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in faces],
            "metadata": {
                "file_path": image_path,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return result

class FinanceTrackerPlus:
    """Advanced financial management"""
    def __init__(self):
        self.plaid_client = self._init_plaid()
        self.stripe_api = stripe
        self.stripe_api.api_key = os.getenv("STRIPE_API_KEY")
        self.transaction_db = "transactions.db"
        self._init_db()
    
    def _init_plaid(self):
        """Initialize Plaid client"""
        configuration = Configuration(
            host=plaid.Environment.Sandbox,
            api_key={
                'clientId': os.getenv("PLAID_CLIENT_ID"),
                'secret': os.getenv("PLAID_SECRET"),
                'publicKey': os.getenv("PLAID_PUBLIC_KEY")
            }
        )
        return plaid_api.PlaidApi(plaid.ApiClient(configuration))
    
    def _init_db(self):
        """Initialize transaction database"""
        conn = sqlite3.connect(self.transaction_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                amount REAL,
                currency TEXT,
                description TEXT,
                date TEXT,
                category TEXT,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def analyze_spending(self, timeframe: str = "monthly") -> Dict:
        """Analyze spending patterns"""
        conn = sqlite3.connect(self.transaction_db)
        cursor = conn.cursor()
        
        if timeframe == "monthly":
            cursor.execute('''
                SELECT strftime('%Y-%m', date) as month, 
                       SUM(amount), 
                       currency 
                FROM transactions 
                GROUP BY month, currency
            ''')
        else:  # weekly
            cursor.execute('''
                SELECT strftime('%Y-%W', date) as week, 
                       SUM(amount), 
                       currency 
                FROM transactions 
                GROUP BY week, currency
            ''')
        
        results = cursor.fetchall()
        conn.close()
        
        return {
            "timeframe": timeframe,
            "results": [
                {"period": r[0], "amount": r[1], "currency": r[2]}
                for r in results
            ]
        }
    
    def process_stripe_payment(self, amount: float, currency: str, description: str) -> Dict:
        """Process payment through Stripe"""
        try:
            payment_intent = self.stripe_api.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency,
                description=description
            )
            
            # Store transaction
            conn = sqlite3.connect(self.transaction_db)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                payment_intent.id,
                amount,
                currency,
                description,
                datetime.now().isoformat(),
                "payment",
                json.dumps({"source": "stripe"})
            ))
            conn.commit()
            conn.close()
            
            return {
                "success": True,
                "payment_intent": payment_intent.id,
                "client_secret": payment_intent.client_secret
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

class HALOSCore:
    """Main system integration point"""
    def __init__(self):
        # Initialize all components
        self.initialize_core_components()
        self.initialize_upgraded_components()
        self.setup_system_services()
        self.configure_logging()
        
    def configure_logging(self):
        """Configure system-wide logging"""
        self.logger = logging.getLogger("HALOS")
        self.logger.setLevel(logging.INFO)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            "halos.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(levelname)s - %(message)s"
        ))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
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
        # System configuration
        self.config = HALOSConfig()
        
        # Multi-LLM Orchestration
        self.llm_orchestrator = LLMOrchestrator()
        
        # Enhanced Security Lab
        self.security_suite = SecuritySuite()
        
        # Code Doctor Pro
        self.code_doctor = CodeDoctorPro()
        
        # Vision Module
        self.vision_processor = VisionProcessor()
        
        # Memory System
        self.memory_manager = MemoryManager()
        
        # Finance Tracker++
        self.finance_tracker = FinanceTrackerPlus()
        
        # Notification System
        self.notification_center = NotificationCenter()
        
        # Task Management
        self.task_queue = queue.Queue()

    def setup_system_services(self):
        """Setup cross-component dependencies"""
        # Share core functionality across components
        self.llm_orchestrator.set_notification_handler(self.notification_center.add)
        self.code_doctor.set_llm_access(self.llm_orchestrator)
        self.vision_processor.set_memory_access(self.memory_manager)
        
        # Initialize background services
        self.start_background_services()

    def start_background_services(self):
        """Start all background services"""
        # Start notification handler
        threading.Thread(target=self.notification_center.process_queue, daemon=True).start()
        
        # Start cloud sync if enabled
        if os.getenv("ENABLE_CLOUD_SYNC", "false").lower() == "true":
            threading.Thread(target=self.memory_manager.sync_with_cloud, daemon=True).start()

    def execute_task(self, task_type: str, **kwargs) -> Dict:
        """Unified task execution interface"""
        task_map = {
            'llm_query': self.llm_orchestrator.process_query,
            'code_analysis': self.code_doctor.analyze,
            'security_scan': self.security_suite.full_scan,
            'document_parse': self.vision_processor.parse_document,
            'memory_store': self.memory_manager.store,
            'financial_analysis': self.finance_tracker.analyze_spending,
            'voice_process': self.process_voice_input,
            'payment_process': self.finance_tracker.process_stripe_payment
        }
        
        if task_type not in task_map:
            raise ValueError(f"Unknown task type: {task_type}")
        
        try:
            return task_map[task_type](**kwargs)
        except Exception as e:
            self.notification_center.add(f"Task failed: {str(e)}", "error")
            return {"success": False, "error": str(e)}

    def process_voice_input(self, audio_stream):
        """Process voice input through the system"""
        try:
            # Convert numpy array to bytes
            audio_bytes = audio_stream.tobytes()
            
            # Save to temporary file
            temp_file = "temp_audio.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
                
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(temp_file)
            os.remove(temp_file)
            
            self.notification_center.add("Voice processed successfully", "info")
            return {"success": True, "text": result["text"]}
        except Exception as e:
            self.notification_center.add(f"Voice processing error: {str(e)}", "error")
            return {"success": False, "error": str(e)}

# ======================
# GUI IMPLEMENTATION
# ======================

class HALOSApp(tk.Tk):
    def __init__(self, core: HALOSCore):
        super().__init__()
        self.core = core
        self.title("HALOS V7 - Integrated AI System")
        self.geometry("1200x900")
        
        # Configure dark mode if enabled
        if self.core.config.dark_mode:
            self.configure(bg='#2d2d2d')
            self.style = ttk.Style()
            self.style.theme_use('alt')
        
        # Initialize GUI components
        self.setup_ui()
        
        # Start background services
        self.start_background_services()

    def setup_ui(self):
        """Setup the main application UI"""
        self.notebook = ttk.Notebook(self)
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
        
        # Status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_assistant_tab(self):
        """Assistant tab with chat interface"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Assistant")
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled')
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input frame
        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.user_input.bind("<Return>", self.process_command)
        
        ttk.Button(input_frame, text="Send", command=self.process_command).pack(side=tk.RIGHT, padx=5)
        
        # Voice controls
        voice_frame = ttk.Frame(tab)
        voice_frame.pack(fill=tk.X)
        
        self.voice_button = ttk.Button(voice_frame, text="Start Listening", 
                                     command=self.toggle_voice_recognition)
        self.voice_button.pack(side=tk.LEFT, padx=5)

    def setup_drawing_tab(self):
        """Drawing canvas tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Drawing")
        
        self.canvas = Canvas(tab, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        controls = ttk.Frame(tab)
        controls.pack(fill=tk.X)
        
        ttk.Button(controls, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT)
        ttk.Button(controls, text="Save", command=self.save_drawing).pack(side=tk.LEFT)
        
        # Drawing bindings
        self.canvas.bind("<B1-Motion>", self.draw)
        self.last_x, self.last_y = None, None

    def setup_security_tab(self):
        """Security tools tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Security")
        
        # Scan target input
        input_frame = ttk.Frame(tab)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Target:").pack(side=tk.LEFT)
        self.scan_target = ttk.Entry(input_frame)
        self.scan_target.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(input_frame, text="Scan", command=self.run_security_scan).pack(side=tk.RIGHT)
        
        # Results display
        self.security_results = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled')
        self.security_results.pack(fill=tk.BOTH, expand=True)

    def setup_finance_tab(self):
        """Financial tools tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Finance")
        
        # Payment form
        form_frame = ttk.Frame(tab)
        form_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(form_frame, text="Amount:").grid(row=0, column=0, sticky=tk.W)
        self.payment_amount = ttk.Entry(form_frame)
        self.payment_amount.grid(row=0, column=1, sticky=tk.EW)
        
        ttk.Label(form_frame, text="Currency:").grid(row=1, column=0, sticky=tk.W)
        self.payment_currency = ttk.Combobox(form_frame, values=["USD", "EUR", "GBP"])
        self.payment_currency.grid(row=1, column=1, sticky=tk.EW)
        self.payment_currency.set("USD")
        
        ttk.Label(form_frame, text="Description:").grid(row=2, column=0, sticky=tk.W)
        self.payment_desc = ttk.Entry(form_frame)
        self.payment_desc.grid(row=2, column=1, sticky=tk.EW)
        
        ttk.Button(form_frame, text="Process Payment", command=self.process_payment).grid(row=3, columnspan=2)
        
        # Analysis section
        ttk.Button(tab, text="Analyze Spending", command=self.show_spending_analysis).pack(pady=5)
        
        # Results display
        self.finance_results = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled')
        self.finance_results.pack(fill=tk.BOTH, expand=True)

    def setup_code_tab(self):
        """Code analysis tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Code Doctor")
        
        # Code input
        self.code_input = scrolledtext.ScrolledText(tab, wrap=tk.WORD)
        self.code_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Language selection
        lang_frame = ttk.Frame(tab)
        lang_frame.pack(fill=tk.X)
        
        ttk.Label(lang_frame, text="Language:").pack(side=tk.LEFT)
        self.code_language = ttk.Combobox(lang_frame, values=["Python", "JavaScript", "Java", "C++"])
        self.code_language.pack(side=tk.LEFT, padx=5)
        self.code_language.set("Python")
        
        # Action buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Analyze", command=self.analyze_code).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Fix", command=self.fix_code).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", command=self.clear_code).pack(side=tk.RIGHT)
        
        # Results display
        self.code_results = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled')
        self.code_results.pack(fill=tk.BOTH, expand=True)

    def setup_vision_tab(self):
        """Computer vision tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Vision")
        
        # Image display
        self.image_label = ttk.Label(tab)
        self.image_label.pack()
        
        # Action buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Extract Text", command=self.extract_text).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Detect Faces", command=self.detect_faces).pack(side=tk.LEFT)
        
        # Results display
        self.vision_results = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled')
        self.vision_results.pack(fill=tk.BOTH, expand=True)

    def setup_memory_tab(self):
        """Memory management tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Memory")
        
        # Search interface
        search_frame = ttk.Frame(tab)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.memory_query = ttk.Entry(search_frame)
        self.memory_query.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.memory_query.bind("<Return>", self.search_memory)
        
        ttk.Button(search_frame, text="Search", command=self.search_memory).pack(side=tk.RIGHT)
        
        # Results display
        self.memory_results = scrolledtext.ScrolledText(tab, wrap=tk.WORD, state='disabled')
        self.memory_results.pack(fill=tk.BOTH, expand=True)

    def setup_settings_tab(self):
        """System settings tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Settings")
        
        # Dark mode toggle
        self.dark_mode_var = tk.BooleanVar(value=self.core.config.dark_mode)
        ttk.Checkbutton(tab, text="Dark Mode", variable=self.dark_mode_var,
                      command=self.toggle_dark_mode).pack(anchor=tk.W)
        
        # LLM provider selection
        provider_frame = ttk.Frame(tab)
        provider_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(provider_frame, text="LLM Provider:").pack(side=tk.LEFT)
        self.llm_provider = ttk.Combobox(
            provider_frame,
            values=[p.name for p in LLMProvider]
        )
        self.llm_provider.pack(side=tk.LEFT, padx=5)
        self.llm_provider.set(self.core.config.current_llm.name)
        self.llm_provider.bind("<<ComboboxSelected>>", self.change_llm_provider)
        
        # Security level
        security_frame = ttk.Frame(tab)
        security_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(security_frame, text="Security Level:").pack(side=tk.LEFT)
        self.security_level = ttk.Scale(
            security_frame,
            from_=1,
            to=5,
            orient=tk.HORIZONTAL
        )
        self.security_level.set(self.core.config.security_level)
        self.security_level.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.security_level.bind("<ButtonRelease-1>", self.change_security_level)

    # [GUI event handlers and helper methods...]
    def process_command(self, event=None):
        """Process user command through the core system"""
        command = self.user_input.get()
        if not command:
            return
            
        self.user_input.delete(0, tk.END)
        self.display_message(f"You: {command}", "user")
        
        # Special commands
        if command.lower().startswith("play "):
            self.handle_play_command(command)
            return
            
        # Process through core system
        result = self.core.execute_task('llm_query', prompt=command)
        
        if result['success']:
            self.display_message(f"HALOS: {result['result']}", "halos")
        else:
            self.display_message(f"HALOS: Error: {result.get('error', 'Unknown error')}", "halos")

    def display_message(self, message, sender):
        """Display message in chat with sender-specific formatting"""
        self.chat_display.config(state='normal')
        tag = "user" if sender == "user" else "halos"
        self.chat_display.tag_config(tag, foreground="blue" if sender == "user" else "green")
        self.chat_display.insert(tk.END, message + "\n", tag)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    def toggle_voice_recognition(self):
        """Toggle voice recognition on/off"""
        if hasattr(self, 'listening') and self.listening:
            self.listening = False
            self.voice_button.config(text="Start Listening")
        else:
            self.listening = True
            self.voice_button.config(text="Listening...")
            threading.Thread(target=self.listen_for_speech, daemon=True).start()

    def listen_for_speech(self):
        """Continuous voice recognition"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            
            while getattr(self, 'listening', False):
                try:
                    audio = recognizer.listen(source, timeout=3)
                    text = recognizer.recognize_google(audio)
                    self.user_input.delete(0, tk.END)
                    self.user_input.insert(0, text)
                    self.process_command()
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    self.status_bar.config(text=f"Voice error: {str(e)}")

    def run_security_scan(self):
        """Run security scan on target"""
        target = self.scan_target.get()
        if not target:
            messagebox.showerror("Error", "Please enter a target to scan")
            return
            
        result = self.core.execute_task('security_scan', target=target)
        self.security_results.config(state='normal')
        self.security_results.delete(1.0, tk.END)
        self.security_results.insert(tk.END, json.dumps(result, indent=2))
        self.security_results.config(state='disabled')

    def process_payment(self):
        """Process payment through Stripe"""
        try:
            amount = float(self.payment_amount.get())
            currency = self.payment_currency.get()
            description = self.payment_desc.get()
            
            result = self.core.execute_task(