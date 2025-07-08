# finance_tab.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from transaction_manager import TransactionManager
from finance_analytics import FinanceAnalytics
from payments import start_payment, verify_payment, PaymentMethod
from datetime import datetime
import threading
import json
import time

class EnhancedFinanceTab:
    def __init__(self, notebook, dark_mode=False):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="Finance+")

        self.dark_mode = dark_mode
        self.transaction_manager = TransactionManager()
        self.analytics_engine = FinanceAnalytics()

        self.payment_in_progress = False
        self.setup_ui()

    def setup_ui(self):
        self.finance_notebook = ttk.Notebook(self.tab)
        self.finance_notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_payment_tab()
        self.setup_transaction_tab()
        self.setup_analytics_tab()

    def setup_payment_tab(self):
        tab = ttk.Frame(self.finance_notebook)
        self.finance_notebook.add(tab, text="Payments")

        ttk.Label(tab, text="Process Payment", font=("Helvetica", 14)).pack(pady=10)

        form_frame = ttk.Frame(tab)
        form_frame.pack(pady=5, fill=tk.X)

        ttk.Label(form_frame, text="Amount:").grid(row=0, column=0, sticky=tk.W)
        self.amount_entry = ttk.Entry(form_frame)
        self.amount_entry.grid(row=0, column=1, padx=5)
        self.amount_entry.insert(0, "49.99")

        ttk.Label(form_frame, text="Email:").grid(row=1, column=0, sticky=tk.W)
        self.email_entry = ttk.Entry(form_frame)
        self.email_entry.grid(row=1, column=1, padx=5)
        self.email_entry.insert(0, "customer@example.com")

        self.payment_method = tk.StringVar(value=PaymentMethod.STRIPE.value)
        methods = [("Stripe", PaymentMethod.STRIPE.value), ("UPI", PaymentMethod.UPI.value)]

        method_frame = ttk.Frame(tab)
        method_frame.pack(pady=5, fill=tk.X)

        ttk.Label(method_frame, text="Method:").pack(side=tk.LEFT)
        for text, val in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.payment_method, value=val).pack(side=tk.LEFT, padx=5)

        ttk.Button(tab, text="Pay Now", command=self.process_payment).pack(pady=10)

        self.payment_status = ttk.Label(tab, text="Ready", relief=tk.SUNKEN)
        self.payment_status.pack(fill=tk.X, pady=5)

    def setup_transaction_tab(self):
        tab = ttk.Frame(self.finance_notebook)
        self.finance_notebook.add(tab, text="Transactions")

        self.transaction_tree = ttk.Treeview(tab, columns=("id", "date", "amount", "method", "status"), show="headings")
        for col in ("id", "date", "amount", "method", "status"):
            self.transaction_tree.heading(col, text=col.capitalize())
            self.transaction_tree.column(col, width=120)

        self.transaction_tree.pack(fill=tk.BOTH, expand=True)
        self.transaction_tree.bind("<<TreeviewSelect>>", self.show_transaction_details)

        self.transaction_details = scrolledtext.ScrolledText(tab, height=6, state='disabled')
        self.transaction_details.pack(fill=tk.BOTH, pady=5)

        ttk.Button(tab, text="Refresh", command=self.load_transactions).pack(pady=5)

        self.load_transactions()

    def setup_analytics_tab(self):
        tab = ttk.Frame(self.finance_notebook)
        self.finance_notebook.add(tab, text="Analytics")

        ttk.Label(tab, text="Financial Analytics", font=("Helvetica", 14)).pack(pady=10)

        self.analytics_display = scrolledtext.ScrolledText(tab, state='disabled')
        self.analytics_display.pack(fill=tk.BOTH, expand=True)

        ttk.Button(tab, text="Generate Report", command=self.generate_report).pack(pady=5)

    def process_payment(self):
        if self.payment_in_progress:
            messagebox.showwarning("Warning", "Payment already in progress")
            return

        try:
            amount = float(self.amount_entry.get())
            email = self.email_entry.get()
            method = PaymentMethod(self.payment_method.get())

            txn = {
                "id": f"txn_{int(time.time())}",
                "date": datetime.now().isoformat(),
                "amount": amount,
                "currency": "USD",
                "method": method.value,
                "status": "initiated",
                "user_email": email,
                "metadata": {}
            }

            success, msg = self.transaction_manager.add_transaction(txn)
            if not success:
                raise RuntimeError(msg)

            self.payment_in_progress = True
            self.payment_status.config(text="Processing payment...")

            threading.Thread(target=self._process_payment_background, args=(method, amount, email, txn["id"]), daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _process_payment_background(self, method, amount, email, txn_id):
        try:
            start_payment(method, amount, {"email": email})
            self.transaction_manager.update_transaction_status(txn_id, "completed")
            self.payment_status.config(text=f"Payment completed: {txn_id}")
        except Exception as e:
            self.transaction_manager.update_transaction_status(txn_id, "failed")
            self.payment_status.config(text=f"Failed: {str(e)}")
        finally:
            self.payment_in_progress = False
            self.load_transactions()

    def load_transactions(self):
        for i in self.transaction_tree.get_children():
            self.transaction_tree.delete(i)

        for txn in self.transaction_manager.get_recent_transactions():
            self.transaction_tree.insert("", "end", values=(txn.id, txn.date, f"{txn.amount} {txn.currency}", txn.method, txn.status))

    def show_transaction_details(self, event):
        selected = self.transaction_tree.selection()
        if not selected:
            return
        item = self.transaction_tree.item(selected[0])
        txn_id = item['values'][0]
        txn = self.transaction_manager.get_transaction(txn_id)

        self.transaction_details.config(state='normal')
        self.transaction_details.delete(1.0, tk.END)
        self.transaction_details.insert(tk.END, json.dumps(txn.__dict__, indent=2))
        self.transaction_details.config(state='disabled')

    def generate_report(self):
        report = self.analytics_engine.generate_report("monthly")
        self.analytics_display.config(state='normal')
        self.analytics_display.delete(1.0, tk.END)
        self.analytics_display.insert(tk.END, json.dumps(report, indent=2))
        self.analytics_display.config(state='disabled')
