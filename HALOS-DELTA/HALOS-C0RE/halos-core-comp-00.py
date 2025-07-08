#!/usr/bin/env python3
"""
HALOS - Hyper-Advanced Learning and Operation System
Integrated Tkinter GUI with Core System Architecture
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
from datetime import datetime, timedelta
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
from typing import Dict, Optional, List
from collections import defaultdict

load_dotenv()

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

class PaymentMethod(Enum):
    """Payment method options"""
    STRIPE = auto()
    UPI = auto()
    PAYPAL = auto()
    BANK_TRANSFER = auto()

class PaymentStatus(Enum):
    """Payment status tracking"""
    PENDING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REFUNDED = auto()

@dataclass
class Transaction:
    """Enhanced transaction data structure"""
    amount: float
    currency: str
    method: PaymentMethod
    status: PaymentStatus
    timestamp: datetime
    description: str
    metadata: Optional[Dict] = None
    transaction_id: Optional[str] = None

@dataclass
class HALOSConfig:
    """Central configuration manager"""
    dark_mode: bool = False
    current_llm: LLMProvider = LLMProvider.OPENAI
    security_level: int = 2
    memory_retention: int = 30  # days
    default_currency: str = "USD"

# ======================
# PAYMENT SYSTEM COMPONENTS
# ======================

class StripeProcessor:
    """Stripe payment processor with multi-currency support"""
    def __init__(self):
        stripe.api_key = os.getenv('STRIPE_API_KEY')
        
    def process(self, amount: float, currency: str, description: str) -> Dict:
        try:
            payment_intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency.lower(),
                description=description
            )
            return {
                'success': True,
                'transaction_id': payment_intent.id,
                'metadata': {
                    'stripe_object': payment_intent
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class UPIProcessor:
    """UPI payment processor (simulated)"""
    def __init__(self):
        self.api_key = os.getenv('UPI_API_KEY', 'demo_key')
        
    def process(self, amount: float, currency: str, description: str) -> Dict:
        try:
            if currency != 'INR':
                return {'success': False, 'error': 'UPI only supports INR'}
                
            # Simulate UPI processing
            time.sleep(1)  # Simulate network delay
            return {
                'success': True,
                'transaction_id': f"UPI{int(time.time())}",
                'metadata': {
                    'amount': amount,
                    'currency': currency,
                    'description': description,
                    'status': 'completed'
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class FinanceAnalytics:
    """Financial analytics and reporting engine"""
    def __init__(self):
        self.data = []
        
    def update(self, transaction: Transaction):
        """Update analytics with new transaction"""
        self.data.append(transaction)
    
    def generate_report(self, period: str) -> Dict:
        """Generate analytics report for the given period"""
        now = datetime.now()
        filtered = []
        
        if period == 'daily':
            filtered = [t for t in self.data if t.timestamp.date() == now.date()]
        elif period == 'weekly':
            start = now - timedelta(days=now.weekday())
            filtered = [t for t in self.data if t.timestamp.date() >= start.date()]
        elif period == 'monthly':
            filtered = [t for t in self.data if t.timestamp.month == now.month]
        else:
            filtered = self.data
            
        return {
            'period': period,
            'total_transactions': len(filtered),
            'total_amount': sum(t.amount for t in filtered),
            'payment_methods': self._aggregate_by_method(filtered),
            'categories': self._aggregate_by_category(filtered)
        }
    
    def _aggregate_by_method(self, transactions: List[Transaction]) -> Dict:
        """Aggregate transactions by payment method"""
        result = defaultdict(float)
        for t in transactions:
            result[t.method.name] += t.amount
        return dict(result)
    
    def _aggregate_by_category(self, transactions: List[Transaction]) -> Dict:
        """Aggregate transactions by category (from description)"""
        result = defaultdict(float)
        for t in transactions:
            category = self._extract_category(t.description)
            result[category] += t.amount
        return dict(result)
    
    def _extract_category(self, description: str) -> str:
        """Extract category from description (simplified)"""
        desc = description.lower()
        if 'food' in desc or 'restaurant' in desc:
            return 'Food'
        elif 'rent' in desc:
            return 'Housing'
        elif 'transport' in desc or 'uber' in desc or 'taxi' in desc:
            return 'Transportation'
        return 'Other'

class FinanceTrackerPlus:
    """Enhanced financial management with payment processing"""
    def __init__(self):
        self.transactions = []
        self.payment_processors = {
            PaymentMethod.STRIPE: StripeProcessor(),
            PaymentMethod.UPI: UPIProcessor()
        }
        self.analytics_engine = FinanceAnalytics()
        self.balance = 0.0
        
    def process_payment(self, amount: float, currency: str, 
                       method: PaymentMethod, description: str) -> Dict:
        """Process a payment through the selected method"""
        processor = self.payment_processors.get(method)
        if not processor:
            return {'success': False, 'error': 'Payment method not supported'}
            
        try:
            result = processor.process(amount, currency, description)
            transaction = Transaction(
                amount=amount,
                currency=currency,
                method=method,
                status=PaymentStatus.COMPLETED if result['success'] else PaymentStatus.FAILED,
                timestamp=datetime.now(),
                description=description,
                metadata=result.get('metadata'),
                transaction_id=result.get('transaction_id')
            )
            self._record_transaction(transaction)
            return {'success': result['success'], 'transaction': transaction}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _record_transaction(self, transaction: Transaction):
        """Record transaction and update balance"""
        self.transactions.append(transaction)
        if transaction.status == PaymentStatus.COMPLETED:
            self.balance += transaction.amount
        self.analytics_engine.update(transaction)
    
    def analyze_spending(self, period: str = 'monthly') -> Dict:
        """Generate spending analytics for the specified period"""
        return self.analytics_engine.generate_report(period)
    
    def get_balance(self) -> float:
        """Get current balance"""
        return self.balance
        
    def get_transactions(self, limit: int = 100) -> List[Transaction]:
        """Get recent transactions"""
        return sorted(self.transactions, key=lambda x: x.timestamp, reverse=True)[:limit]

class HALOSCore:
    """Main system integration point"""
    def __init__(self):
        # Initialize all components
        self.initialize_core_components()
        self.initialize_upgraded_components()
        self.setup_system_services()
        
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
            'financial_operation': self._process_financial_operation,
            'voice_process': self.process_voice_input
        }
        
        if task_type not in task_map:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return task_map[task_type](**kwargs)
    
    def _process_financial_operation(self, operation: str, **kwargs) -> Dict:
        """Handle financial operations"""
        if operation == 'process_payment':
            return self.finance_tracker.process_payment(
                amount=kwargs.get('amount'),
                currency=kwargs.get('currency'),
                method=kwargs.get('method'),
                description=kwargs.get('description')
            )
        elif operation == 'get_balance':
            return {'success': True, 'balance': self.finance_tracker.get_balance()}
        elif operation == 'get_transactions':
            return {'success': True, 'transactions': self.finance_tracker.get_transactions()}
        else:
            return {'success': False, 'error': f'Unknown financial operation: {operation}'}

    def process_voice_input(self, audio_stream):
        """Process voice input through the system"""
        try:
            # First try upgraded voice processing
            result = self.voice_processor.process(audio_stream)
            self.notification_center.add("Voice processed successfully", "info")
            return result
        except Exception as e:
            self.notification_center.add(f"Voice processing error: {str(e)}", "error")
            # Fallback to legacy system
            return self.legacy_voice_process(audio_stream)

# ======================
# UPGRADED COMPONENTS
# ======================

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
            if provider == self._select_provider(prompt, context):
                continue  # Skip the one that already failed
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

    def setup_finance_tab(self):
        """Finance tab with payment processing and analytics"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Finance")
        
        # Create notebook within the tab
        finance_notebook = ttk.Notebook(tab)
        finance_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Payment Tab
        payment_frame = ttk.Frame(finance_notebook)
        self.setup_payment_controls(payment_frame)
        finance_notebook.add(payment_frame, text="Payments")
        
        # Analytics Tab
        analytics_frame = ttk.Frame(finance_notebook)
        self.setup_analytics_controls(analytics_frame)
        finance_notebook.add(analytics_frame, text="Analytics")
        
        # Transactions Tab
        transactions_frame = ttk.Frame(finance_notebook)
        self.setup_transactions_table(transactions_frame)
        finance_notebook.add(transactions_frame, text="Transactions")

    def setup_payment_controls(self, parent):
        """Setup payment processing controls"""
        frame = ttk.LabelFrame(parent, text="Process Payment")
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Amount
        ttk.Label(frame, text="Amount:").grid(row=0, column=0, sticky=tk.W)
        self.payment_amount = ttk.Entry(frame)
        self.payment_amount.grid(row=0, column=1)
        
        # Currency
        ttk.Label(frame, text="Currency:").grid(row=1, column=0, sticky=tk.W)
        self.payment_currency = ttk.Combobox(frame, values=['USD', 'EUR', 'GBP', 'INR'])
        self.payment_currency.set(self.core.config.default_currency)
        self.payment_currency.grid(row=1, column=1)
        
        # Method
        ttk.Label(frame, text="Method:").grid(row=2, column=0, sticky=tk.W)
        self.payment_method = ttk.Combobox(frame, values=[m.name for m in PaymentMethod])
        self.payment_method.set('STRIPE')
        self.payment_method.grid(row=2, column=1)
        
        # Description
        ttk.Label(frame, text="Description:").grid(row=3, column=0, sticky=tk.W)
        self.payment_description = ttk.Entry(frame)
        self.payment_description.grid(row=3, column=1)
        
        # Process Button
        ttk.Button(frame, text="Process Payment", command=self.process_payment).grid(row=4, columnspan=2)
        
        # Balance display
        balance_frame = ttk.Frame(parent)
        balance_frame.pack(fill=tk.X, pady=10)
        ttk.Label(balance_frame, text="Current Balance:").pack(side=tk.LEFT)
        self.balance_label = ttk.Label(balance_frame, text="0.00")
        self.balance_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(balance_frame, text="Refresh", command=self.update_balance).pack(side=tk.RIGHT)

    def setup_analytics_controls(self, parent):
        """Setup financial analytics controls"""
        frame = ttk.LabelFrame(parent, text="Financial Analytics")
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Period selection
        period_frame = ttk.Frame(frame)
        period_frame.pack(fill=tk.X)
        ttk.Label(period_frame, text="Report Period:").pack(side=tk.LEFT)
        self.analytics_period = ttk.Combobox(period_frame, values=['daily', 'weekly', 'monthly', 'all'])
        self.analytics_period.set('monthly')
        self.analytics_period.pack(side=tk.LEFT, padx=5)
        ttk.Button(period_frame, text="Generate", command=self.generate_analytics).pack(side=tk.LEFT)
        
        # Report display
        self.analytics_display = scrolledtext.ScrolledText(frame, wrap=tk.WORD, state='disabled')
        self.analytics_display.pack(fill=tk.BOTH, expand=True)

    def setup_transactions_table(self, parent):
        """Setup transactions history table"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for transactions
        columns = ("date", "amount", "currency", "method", "status", "description")
        self.transactions_tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        for col in columns:
            self.transactions_tree.heading(col, text=col.capitalize())
            self.transactions_tree.column(col, width=100)
        
        self.transactions_tree.pack(fill=tk.BOTH, expand=True)
        
        # Load transactions
        self.refresh_transactions()

    def process_payment(self):
        """Process payment from GUI"""
        try:
            amount = float(self.payment_amount.get())
            currency = self.payment_currency.get()
            method = PaymentMethod[self.payment_method.get()]
            description = self.payment_description.get()
            
            result = self.core.execute_task('financial_operation', 
                                          operation='process_payment',
                                          amount=amount,
                                          currency=currency,
                                          method=method,
                                          description=description)
            
            if result['success']:
                messagebox.showinfo("Success", "Payment processed successfully")
                self.refresh_transactions()
                self.update_balance()
            else:
                messagebox.showerror("Error", result.get('error', 'Payment failed'))
        except ValueError:
            messagebox.showerror("Error", "Invalid amount")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_analytics(self):
        """Generate and display financial analytics"""
        period = self.analytics_period.get()
        result = self.core.execute_task('financial_analysis', period=period)
        
        self.analytics_display.config(state='normal')
        self.analytics_display.delete(1.0, tk.END)
        
        if result['success']:
            report = result['result']
            text = f"=== {report['period'].upper()} REPORT ===\n"
            text += f"Total Transactions: {report['total_transactions']}\n"
            text += f"Total Amount: {report['total_amount']:.2f}\n\n"
            
            text += "By Payment Method:\n"
            for method, amount in report['payment_methods'].items():
                text += f"- {method}: {amount:.2f}\n"
                
            text += "\nBy Category:\n"
            for category, amount in report['categories'].items():
                text += f"- {category}: {amount:.2f}\n"
                
            self.analytics_display.insert(tk.END, text)
        else:
            self.analytics_display.insert(tk.END, f"Error: {result.get('error', 'Unknown error')}")
        
        self.analytics_display.config(state='disabled')

    def refresh_transactions(self):
        """Refresh transactions table"""
        # Clear existing data
        for item in self.transactions_tree.get_children():
            self.transactions_tree.delete(item)
        
        # Get transactions from core
        result = self.core.execute_task('financial_operation', operation='get_transactions')
        
        if result['success']:
            transactions = result['transactions']
            # Add to treeview
            for t in transactions:
                self.transactions_tree.insert("", tk.END, values=(
                    t.timestamp.strftime("%Y-%m-%d %H:%M"),
                    f"{t.amount:.2f}",
                    t.currency,
                    t.method.name,
                    t.status.name,
                    t.description
                ))

    def update_balance(self):
        """Update the balance display"""
        result = self.core.execute_task('financial_operation', operation='get_balance')
        if result['success']:
            self.balance_label.config(text=f"{result['balance']:.2f}")

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

    def start_background_services(self):
        """Start background services"""
        # Start notification monitor
        threading.Thread(target=self.monitor_notifications, daemon=True).start()
        
        # Start voice listener
        self.listening = False

    def monitor_notifications(self):
        """Monitor and display notifications from core"""
        while True:
            notification = self.core.notification_center.get_notification()
            if notification:
                self.status_bar.config(text=f"Notification: {notification['message']}")
                if notification['level'] == "error":
                    messagebox.showerror("Error", notification['message'])
            time.sleep(0.1)

# ======================
# MAIN APPLICATION
# ======================

if __name__ == "__main__":
    # Initialize core system
    core_system = HALOSCore()
    
    # Start GUI
    app = HALOSApp(core_system)
    
    # Configure dark mode if enabled
    if core_system.config.dark_mode:
        style = ttk.Style()
        style.theme_use('alt')
    
    app.mainloop()