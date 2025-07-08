# halos_finance_integration.py
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime
import threading
import json
import sqlite3
from enum import Enum

# Import the existing analytics class
from finance_analytics import FinanceAnalytics, TimePeriod

class PaymentMethod(Enum):
    """Payment method options"""
    STRIPE = "stripe"
    UPI = "upi"
    BANK = "bank"
    CRYPTO = "crypto"

class HALOSFinanceSystem:
    """Integrated finance system for HALOS"""
    
    def __init__(self, db_path: str = "halos_finance.db"):
        """
        Initialize the finance system with database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.analytics = FinanceAnalytics(db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    date TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    user_email TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_email 
                ON transactions(user_email)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_date 
                ON transactions(date)
            ''')
            
            conn.commit()

    def add_transaction(self, transaction: Dict) -> bool:
        """
        Add a new transaction to the system
        
        Args:
            transaction: Dictionary containing transaction data
            
        Returns:
            True if successful, False otherwise
        """
        required_fields = {
            'id', 'date', 'amount', 'currency', 
            'method', 'status', 'user_email'
        }
        
        if not all(field in transaction for field in required_fields):
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO transactions 
                    (id, date, amount, currency, method, status, user_email, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction['id'],
                    transaction['date'],
                    transaction['amount'],
                    transaction['currency'],
                    transaction['method'],
                    transaction['status'],
                    transaction['user_email'],
                    json.dumps(transaction.get('metadata', {}))
                ))
                conn.commit()
                return True
        except sqlite3.Error:
            return False

    def get_transactions(self, limit: int = 100) -> List[Dict]:
        """
        Get recent transactions
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM transactions
                ORDER BY date DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

class HALOSFinanceTab(ttk.Frame):
    """Integrated finance tab for HALOS GUI"""
    
    def __init__(self, notebook, dark_mode=False):
        """
        Initialize the finance tab
        
        Args:
            notebook: ttk.Notebook to attach this tab to
            dark_mode: Whether to use dark mode styling
        """
        super().__init__(notebook)
        self.notebook = notebook
        self.dark_mode = dark_mode
        self.finance_system = HALOSFinanceSystem()
        self.payment_in_progress = False
        
        # Configure style
        self.style = ttk.Style()
        if self.dark_mode:
            self.style.configure('Finance.TFrame', background='#2d2d2d')
            self.configure(style='Finance.TFrame')
        
        self.setup_ui()

    def setup_ui(self):
        """Initialize all UI components"""
        # Create notebook for different finance sections
        self.finance_notebook = ttk.Notebook(self)
        self.finance_notebook.pack(fill=tk.BOTH, expand=True)

        # Add tabs
        self.setup_payment_tab()
        self.setup_subscription_tab()
        self.setup_transaction_tab()
        self.setup_analytics_tab()

    def setup_payment_tab(self):
        """Payment processing tab"""
        tab = ttk.Frame(self.finance_notebook)
        self.finance_notebook.add(tab, text="Payments")

        # Header
        ttk.Label(tab, text="Process Payment", font=("Helvetica", 14)).pack(pady=10)

        # Payment method selector
        method_frame = ttk.Frame(tab)
        method_frame.pack(pady=5, fill=tk.X)

        ttk.Label(method_frame, text="Payment Method:").pack(side=tk.LEFT)

        self.payment_method = tk.StringVar(value=PaymentMethod.STRIPE.value)
        methods = [
            ("Stripe (Cards, Apple Pay)", PaymentMethod.STRIPE.value),
            ("UPI (India)", PaymentMethod.UPI.value),
            ("Bank Transfer", "bank"),
            ("Crypto", "crypto")
        ]
        
        for text, value in methods:
            ttk.Radiobutton(
                method_frame,
                text=text,
                variable=self.payment_method,
                value=value
            ).pack(side=tk.LEFT, padx=5)

        # Amount entry
        form_frame = ttk.Frame(tab)
        form_frame.pack(pady=5, fill=tk.X)

        ttk.Label(form_frame, text="Amount:").grid(row=0, column=0, sticky=tk.W)
        self.amount_entry = ttk.Entry(form_frame)
        self.amount_entry.grid(row=0, column=1, padx=5)
        self.amount_entry.insert(0, "49.99")

        ttk.Label(form_frame, text="Currency:").grid(row=1, column=0, sticky=tk.W)
        self.currency_var = tk.StringVar(value="USD")
        currencies = ["USD", "EUR", "GBP", "INR", "JPY"]
        ttk.Combobox(
            form_frame,
            textvariable=self.currency_var,
            values=currencies,
            state="readonly"
        ).grid(row=1, column=1, padx=5, sticky=tk.W)

        # Customer info
        ttk.Label(form_frame, text="Email:").grid(row=2, column=0, sticky=tk.W)
        self.email_entry = ttk.Entry(form_frame)
        self.email_entry.grid(row=2, column=1, padx=5, pady=5)
        self.email_entry.insert(0, "customer@example.com")

        # Pay button
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(pady=10)

        ttk.Button(
            btn_frame,
            text="Process Payment",
            command=self.process_payment
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Verify Payment",
            command=self.verify_payment
        ).pack(side=tk.LEFT)

        # Status display
        self.payment_status = ttk.Label(tab, text="Ready", relief=tk.SUNKEN)
        self.payment_status.pack(fill=tk.X, pady=5)

    def setup_subscription_tab(self):
        """Subscription management tab"""
        tab = ttk.Frame(self.finance_notebook)
        self.finance_notebook.add(tab, text="Subscriptions")

        ttk.Label(tab, text="Subscription Management", font=("Helvetica", 14)).pack(pady=10)

        # Subscription plans
        plans_frame = ttk.Frame(tab)
        plans_frame.pack(pady=10)

        self.subscription_plan = tk.StringVar(value="pro")
        plans = [
            ("Basic - $9.99/month", "basic"),
            ("Pro - $49.99/month", "pro"),
            ("Enterprise - $199/month", "enterprise")
        ]

        for text, value in plans:
            ttk.Radiobutton(
                plans_frame,
                text=text,
                variable=self.subscription_plan,
                value=value
            ).pack(anchor=tk.W)

        # Action buttons
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(pady=10)

        ttk.Button(
            btn_frame,
            text="Subscribe",
            command=self.subscribe
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="Cancel Subscription",
            command=self.cancel_subscription
        ).pack(side=tk.LEFT)

        # Subscription status
        self.sub_status = scrolledtext.ScrolledText(tab, height=5, state='disabled')
        self.sub_status.pack(fill=tk.BOTH, expand=True, pady=5)
        self.update_subscription_status("No active subscription")

    def setup_transaction_tab(self):
        """Transaction history tab"""
        tab = ttk.Frame(self.finance_notebook)
        self.finance_notebook.add(tab, text="Transactions")

        # Header and refresh button
        header_frame = ttk.Frame(tab)
        header_frame.pack(fill=tk.X, pady=5)

        ttk.Label(header_frame, text="Transaction History", font=("Helvetica", 14)).pack(side=tk.LEFT)
        ttk.Button(header_frame, text="Refresh", command=self.load_transactions).pack(side=tk.RIGHT)

        # Transaction treeview
        columns = ("id", "date", "amount", "method", "status")
        self.transaction_tree = ttk.Treeview(
            tab,
            columns=columns,
            show="headings",
            selectmode="browse"
        )

        # Configure columns
        self.transaction_tree.heading("id", text="ID")
        self.transaction_tree.heading("date", text="Date")
        self.transaction_tree.heading("amount", text="Amount")
        self.transaction_tree.heading("method", text="Method")
        self.transaction_tree.heading("status", text="Status")

        self.transaction_tree.column("id", width=100)
        self.transaction_tree.column("date", width=150)
        self.transaction_tree.column("amount", width=100)
        self.transaction_tree.column("method", width=100)
        self.transaction_tree.column("status", width=100)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.transaction_tree.yview)
        self.transaction_tree.configure(yscrollcommand=scrollbar.set)
        self.transaction_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Details panel
        self.transaction_details = scrolledtext.ScrolledText(tab, height=8, state='disabled')
        self.transaction_details.pack(fill=tk.BOTH, expand=False)

        # Bind selection event
        self.transaction_tree.bind("<<TreeviewSelect>>", self.show_transaction_details)

        # Load initial data
        self.load_transactions()

    def setup_analytics_tab(self):
        """Financial analytics tab"""
        tab = ttk.Frame(self.finance_notebook)
        self.finance_notebook.add(tab, text="Analytics")

        ttk.Label(tab, text="Financial Analytics", font=("Helvetica", 14)).pack(pady=10)

        # Time period selector
        period_frame = ttk.Frame(tab)
        period_frame.pack(pady=5)

        ttk.Label(period_frame, text="Time Period:").pack(side=tk.LEFT)
        self.period_var = tk.StringVar(value="monthly")
        ttk.Combobox(
            period_frame,
            textvariable=self.period_var,
            values=["daily", "weekly", "monthly", "yearly"],
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(period_frame, text="Generate", command=self.generate_analytics).pack(side=tk.LEFT)

        # Analytics display
        self.analytics_display = scrolledtext.ScrolledText(tab, state='disabled')
        self.analytics_display.pack(fill=tk.BOTH, expand=True, pady=5)

    def process_payment(self):
        """Process payment in a background thread"""
        if self.payment_in_progress:
            messagebox.showwarning("Warning", "Payment already in progress")
            return

        try:
            amount = float(self.amount_entry.get())
            if amount <= 0:
                raise ValueError("Amount must be positive")

            method = PaymentMethod(self.payment_method.get())
            currency = self.currency_var.get()
            email = self.email_entry.get()

            if not email or "@" not in email:
                raise ValueError("Please enter a valid email")

            transaction = {
                "id": f"txn_{int(datetime.now().timestamp())}",
                "date": datetime.now().isoformat(),
                "amount": amount,
                "currency": currency,
                "method": method.value,
                "status": "pending",
                "user_email": email,
                "metadata": {
                    "source": "HALOS GUI",
                    "description": "Payment processing"
                }
            }

            self.payment_in_progress = True
            self.payment_status.config(text="Processing payment...")

            # Run in background thread to keep UI responsive
            threading.Thread(
                target=self._process_payment_background,
                args=(method, amount, transaction),
                daemon=True
            ).start()

        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.payment_status.config(text="Payment failed")

    def _process_payment_background(self, method: PaymentMethod, amount: float, transaction: Dict):
        """Background payment processing"""
        try:
            # In a real implementation, this would call the payment processor
            # For now, we'll simulate a successful payment after a delay
            time.sleep(2)  # Simulate processing time
            
            # Update transaction status
            transaction['status'] = "completed"
            self.finance_system.add_transaction(transaction)
            
            self.after(0, lambda: self.payment_status.config(
                text=f"Payment completed - {transaction['id']}"
            ))
            
        except Exception as e:
            transaction['status'] = "failed"
            transaction['metadata']['error'] = str(e)
            self.finance_system.add_transaction(transaction)
            
            self.after(0, lambda: messagebox.showerror("Payment Error", str(e)))
            self.after(0, lambda: self.payment_status.config(
                text=f"Payment failed: {str(e)}"
            ))
            
        finally:
            self.payment_in_progress = False
            self.after(0, self.load_transactions)

    def verify_payment(self):
        """Verify payment status"""
        selection = self.transaction_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a transaction to verify")
            return

        item = self.transaction_tree.item(selection[0])
        txn_id = item['values'][0]  # First column is transaction ID

        try:
            # In a real implementation, this would verify with the payment processor
            # For now, we'll just update the status in our database
            transaction = self.finance_system.get_transaction(txn_id)
            if not transaction:
                raise ValueError("Transaction not found")
            
            # Simulate verification
            transaction['status'] = "completed"
            self.finance_system.update_transaction(transaction)
            
            messagebox.showinfo("Success", "Payment verified successfully")
            self.load_transactions()
                
        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

    def subscribe(self):
        """Handle subscription process"""
        plan = self.subscription_plan.get()
        amount = 9.99 if plan == "basic" else 49.99 if plan == "pro" else 199.00
        
        try:
            # Create subscription transaction
            transaction = {
                "id": f"sub_{int(datetime.now().timestamp())}",
                "date": datetime.now().isoformat(),
                "amount": amount,
                "currency": "USD",
                "method": "stripe",
                "status": "active",
                "user_email": self.email_entry.get(),
                "metadata": {
                    "plan": plan,
                    "type": "subscription"
                }
            }
            
            self.finance_system.add_transaction(transaction)
            
            self.update_subscription_status(f"Subscribed to {plan} plan at ${amount}/month")
            messagebox.showinfo("Success", f"Subscribed to {plan} plan")
            self.load_transactions()
            
        except Exception as e:
            messagebox.showerror("Error", f"Subscription failed: {str(e)}")

    def cancel_subscription(self):
        """Cancel active subscription"""
        if messagebox.askyesno("Confirm", "Are you sure you want to cancel your subscription?"):
            self.update_subscription_status("Subscription cancelled")
            messagebox.showinfo("Success", "Subscription cancelled")

    def load_transactions(self):
        """Load transactions from database into the treeview"""
        # Clear existing data
        for item in self.transaction_tree.get_children():
            self.transaction_tree.delete(item)
        
        # Get transactions from database
        transactions = self.finance_system.get_transactions()
        
        # Add to treeview
        for txn in transactions:
            self.transaction_tree.insert("", "end", values=(
                txn['id'],
                txn['date'],
                f"{txn['amount']} {txn['currency']}",
                txn['method'],
                txn['status']
            ))

    def show_transaction_details(self, event):
        """Show details for selected transaction"""
        selection = self.transaction_tree.selection()
        if not selection:
            return

        item = self.transaction_tree.item(selection[0])
        txn_id = item['values'][0]
        
        # Get transaction from database
        txn = self.finance_system.get_transaction(txn_id)
        if not txn:
            return

        # Format details
        details = json.dumps(txn, indent=2)
        self.transaction_details.config(state='normal')
        self.transaction_details.delete(1.0, tk.END)
        self.transaction_details.insert(tk.END, details)
        self.transaction_details.config(state='disabled')

    def generate_analytics(self):
        """Generate financial analytics report"""
        period = self.period_var.get()
        
        try:
            # Get analytics data
            if period == "daily":
                data = self.finance_system.analytics.get_totals_by_period(TimePeriod.DAILY)
            elif period == "weekly":
                data = self.finance_system.analytics.get_totals_by_period(TimePeriod.WEEKLY)
            elif period == "monthly":
                data = self.finance_system.analytics.get_totals_by_period(TimePeriod.MONTHLY)
            else:  # yearly
                data = self.finance_system.analytics.get_totals_by_period(TimePeriod.YEARLY)
            
            # Format report
            report = f"Analytics Report - {period.capitalize()}\n\n"
            report += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for item in data:
                report += (f"{item['period_label']}: "
                          f"${item['total_amount']:,.2f} "
                          f"({item['transaction_count']} transactions)\n")
            
            # Display report
            self.analytics_display.config(state='normal')
            self.analytics_display.delete(1.0, tk.END)
            self.analytics_display.insert(tk.END, report)
            self.analytics_display.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate analytics: {str(e)}")

    def update_subscription_status(self, message: str):
        """Update subscription status display"""
        self.sub_status.config(state='normal')
        self.sub_status.delete(1.0, tk.END)
        self.sub_status.insert(tk.END, message)
        self.sub_status.config(state='disabled')

# Integration with main HALOS system
class HALOSCore:
    """Main HALOS system with integrated finance module"""
    
    def __init__(self):
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all system components"""
        # Other system components would be initialized here
        self.finance_system = HALOSFinanceSystem()
        
    def get_finance_tab(self, notebook, dark_mode=False):
        """Get the finance tab for the main GUI"""
        return HALOSFinanceTab(notebook, dark_mode)