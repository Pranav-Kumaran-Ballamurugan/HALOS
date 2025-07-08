# upgrade_finance_tab.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from payments import start_payment, verify_payment, PaymentMethod
from datetime import datetime
import threading
import json
from typing import Dict, Optional

class EnhancedFinanceTab:
    def __init__(self, notebook, dark_mode=False):
        """
        Enhanced finance tab with payment processing, transaction history, and subscription management
        
        Args:
            notebook: ttk.Notebook to attach this tab to
            dark_mode: Whether to use dark mode styling
        """
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="Finance+")
        
        self.dark_mode = dark_mode
        self.payment_in_progress = False
        self.transaction_history = []
        self.setup_ui()

    def setup_ui(self):
        """Initialize all UI components"""
        # Configure style
        self.style = ttk.Style()
        if self.dark_mode:
            self.style.configure('Finance.TFrame', background='#2d2d2d')
            self.tab.configure(style='Finance.TFrame')

        # Create notebook for different finance sections
        self.finance_notebook = ttk.Notebook(self.tab)
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

            user_info = {
                "email": email,
                "name": "Customer",
                "currency": currency
            }

            self.payment_in_progress = True
            self.payment_status.config(text="Processing payment...")

            # Run in background thread to keep UI responsive
            threading.Thread(
                target=self._process_payment_background,
                args=(method, amount, user_info),
                daemon=True
            ).start()

        except ValueError as e:
            messagebox.showerror("Error", str(e))
            self.payment_status.config(text="Payment failed")

    def _process_payment_background(self, method: PaymentMethod, amount: float, user_info: Dict):
        """Background payment processing"""
        try:
            # Start payment process
            start_payment(method, amount, user_info)
            
            # Record transaction
            transaction = {
                "id": f"txn_{int(time.time())}",
                "date": datetime.now().isoformat(),
                "amount": amount,
                "currency": user_info.get("currency", "USD"),
                "method": method.value,
                "status": "initiated",
                "user_email": user_info["email"]
            }
            
            self.transaction_history.append(transaction)
            self.tab.after(0, lambda: self.payment_status.config(text=f"Payment initiated - {transaction['id']}"))
            
        except Exception as e:
            self.tab.after(0, lambda: messagebox.showerror("Payment Error", str(e)))
            self.tab.after(0, lambda: self.payment_status.config(text=f"Payment failed: {str(e)}"))
            
        finally:
            self.payment_in_progress = False
            self.tab.after(0, self.load_transactions)

    def verify_payment(self):
        """Verify payment status"""
        selection = self.transaction_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a transaction to verify")
            return

        item = self.transaction_tree.item(selection[0])
        txn_id = item['values'][0]  # First column is transaction ID

        try:
            # Determine payment method from transaction data
            method_str = item['values'][3].lower()  # Method column
            method = PaymentMethod(method_str) if method_str in PaymentMethod._value2member_map_ else None
            
            if not method:
                raise ValueError("Cannot verify this payment method")

            result = verify_payment(method, txn_id)
            
            if result.get('verified'):
                messagebox.showinfo("Success", "Payment verified successfully")
                # Update transaction status
                for txn in self.transaction_history:
                    if txn['id'] == txn_id:
                        txn['status'] = "completed"
                        break
                self.load_transactions()
            else:
                messagebox.showwarning("Warning", "Payment not yet verified")
                
        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

    def subscribe(self):
        """Handle subscription process"""
        plan = self.subscription_plan.get()
        amount = 9.99 if plan == "basic" else 49.99 if plan == "pro" else 199.00
        
        try:
            # In a real implementation, this would call the subscription API
            self.update_subscription_status(f"Subscribed to {plan} plan at ${amount}/month")
            messagebox.showinfo("Success", f"Subscribed to {plan} plan")
            
            # Record subscription transaction
            transaction = {
                "id": f"sub_{int(time.time())}",
                "date": datetime.now().isoformat(),
                "amount": amount,
                "currency": "USD",
                "method": "stripe",
                "status": "active",
                "user_email": self.email_entry.get(),
                "plan": plan
            }
            self.transaction_history.append(transaction)
            self.load_transactions()
            
        except Exception as e:
            messagebox.showerror("Error", f"Subscription failed: {str(e)}")

    def cancel_subscription(self):
        """Cancel active subscription"""
        if messagebox.askyesno("Confirm", "Are you sure you want to cancel your subscription?"):
            self.update_subscription_status("Subscription cancelled")
            messagebox.showinfo("Success", "Subscription cancelled")

    def load_transactions(self):
        """Load transactions into the treeview"""
        # Clear existing data
        for item in self.transaction_tree.get_children():
            self.transaction_tree.delete(item)
        
        # Add transactions (in a real app, this would come from a database)
        for txn in self.transaction_history:
            self.transaction_tree.insert("", "end", values=(
                txn['id'],
                txn['date'],
                f"{txn['amount']} {txn.get('currency', 'USD')}",
                txn['method'],
                txn.get('status', 'pending')
            ))

    def show_transaction_details(self, event):
        """Show details for selected transaction"""
        selection = self.transaction_tree.selection()
        if not selection:
            return

        item = self.transaction_tree.item(selection[0])
        txn_id = item['values'][0]
        
        # Find the transaction in history
        txn = next((t for t in self.transaction_history if t['id'] == txn_id), None)
        if not txn:
            return

        # Format details
        details = json.dumps(txn, indent=2)
        self.transaction_details.config(state='normal')
        self.transaction_details.delete(1.0, tk.END)
        self.transaction_details.insert(tk.END, details)
        self.transaction_details.config(state='disabled')

    def generate_analytics(self):
        """Generate financial analytics"""
        period = self.period_var.get()
        
        # In a real implementation, this would query transaction data
        analytics_text = f"Analytics for {period} period:\n\n"
        
        if period == "daily":
            analytics_text += "Total payments today: $0.00\n"
            analytics_text += "Average transaction: $0.00\n"
        elif period == "weekly":
            analytics_text += "Total payments this week: $0.00\n"
            analytics_text += "Average transaction: $0.00\n"
        elif period == "monthly":
            analytics_text += "Total payments this month: $0.00\n"
            analytics_text += "Average transaction: $0.00\n"
        else:  # yearly
            analytics_text += "Total payments this year: $0.00\n"
            analytics_text += "Average transaction: $0.00\n"
        
        self.analytics_display.config(state='normal')
        self.analytics_display.delete(1.0, tk.END)
        self.analytics_display.insert(tk.END, analytics_text)
        self.analytics_display.config(state='disabled')

    def update_subscription_status(self, message: str):
        """Update subscription status display"""
        self.sub_status.config(state='normal')
        self.sub_status.delete(1.0, tk.END)
        self.sub_status.insert(tk.END, message)
        self.sub_status.config(state='disabled')

    def add_sample_transactions(self):
        """Add sample transactions for demonstration"""
        sample_txns = [
            {
                "id": "txn_12345",
                "date": "2023-01-15T10:30:00",
                "amount": 49.99,
                "currency": "USD",
                "method": "stripe",
                "status": "completed",
                "user_email": "customer@example.com"
            },
            {
                "id": "txn_67890",
                "date": "2023-02-20T14:45:00",
                "amount": 29.99,
                "currency": "EUR",
                "method": "bank",
                "status": "pending",
                "user_email": "customer@example.com"
            }
        ]
        self.transaction_history.extend(sample_txns)
        self.load_transactions()