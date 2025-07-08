# halos_main.py - fully integrated

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import json
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from enhanced_finance_sqlite import TransactionManager
from finance_analytics import FinanceAnalytics
from payments_enhanced import start_payment, verify_payment, PaymentMethod, PaymentStatus

class HALOSCore:
    def __init__(self):
        self.transaction_manager = TransactionManager()
        self.finance_analytics = FinanceAnalytics()

class HALOSApp(tk.Tk):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.title("HALOS V8 - AI + Finance Suite")
        self.geometry("1200x900")
        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.setup_finance_tab()

    def setup_finance_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Finance+")
        notebook = ttk.Notebook(tab)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.setup_payment_tab(notebook)
        self.setup_transaction_tab(notebook)
        self.setup_analytics_tab(notebook)

    def setup_payment_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Payments")

        ttk.Label(tab, text="Process Payment", font=("Helvetica", 14)).pack(pady=10)

        frame = ttk.Frame(tab)
        frame.pack(pady=5)

        self.payment_method = tk.StringVar(value="stripe")
        for text, value in [("Stripe", "stripe"), ("UPI", "upi")]:
            ttk.Radiobutton(frame, text=text, variable=self.payment_method, value=value).pack(side=tk.LEFT, padx=5)

        self.amount_entry = ttk.Entry(frame)
        self.amount_entry.insert(0, "49.99")
        self.amount_entry.pack(side=tk.LEFT, padx=5)

        self.email_entry = ttk.Entry(frame)
        self.email_entry.insert(0, "customer@example.com")
        self.email_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(tab, text="Pay Now", command=self.process_payment).pack(pady=10)

    def process_payment(self):
        amount = Decimal(self.amount_entry.get())
        email = self.email_entry.get()
        method = PaymentMethod(self.payment_method.get())

        txn_id = f"txn_{int(time.time())}"
        txn = {
            "id": txn_id,
            "date": datetime.now().isoformat(),
            "amount": float(amount),
            "currency": "USD",
            "method": method.value,
            "status": "initiated",
            "user_email": email,
            "metadata": {}
        }

        ok, msg = self.core.transaction_manager.add_transaction(txn)
        if not ok:
            messagebox.showerror("Error", msg)
            return

        self.status = ttk.Label(self, text="Processing payment...")
        self.status.pack()

        threading.Thread(
            target=self._process_payment_bg,
            args=(method, amount, email, txn_id),
            daemon=True
        ).start()

    def _process_payment_bg(self, method, amount, email, txn_id):
        try:
            payment_id, status = start_payment(method, amount, {"email": email})
            self.core.transaction_manager.update_transaction_status(txn_id, status.value)
            self.status.config(text=f"Payment: {status.name} (ID: {payment_id})")

            if status == PaymentStatus.PENDING:
                time.sleep(5)
                verify_result = verify_payment(method, payment_id)
                self.core.transaction_manager.update_transaction_status(txn_id, verify_result.value)
                self.status.config(text=f"Verified: {verify_result.name}")

        except Exception as e:
            self.core.transaction_manager.update_transaction_status(txn_id, "failed")
            self.status.config(text=f"Payment failed: {e}")

        self.load_transactions()

    def setup_transaction_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Transactions")
        columns = ("id", "date", "amount", "method", "status")
        self.transaction_tree = ttk.Treeview(tab, columns=columns, show="headings")
        for col in columns:
            self.transaction_tree.heading(col, text=col.capitalize())
        self.transaction_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(tab, text="Refresh", command=self.load_transactions).pack(pady=5)
        self.load_transactions()

    def load_transactions(self):
        for i in self.transaction_tree.get_children():
            self.transaction_tree.delete(i)
        txns = self.core.transaction_manager.get_recent_transactions(100)
        for t in txns:
            self.transaction_tree.insert("", "end", values=(t.id, t.date, t.amount, t.method, t.status))

    def setup_analytics_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Analytics")
        st = scrolledtext.ScrolledText(tab)
        st.pack(fill=tk.BOTH, expand=True)
        report = self.core.finance_analytics.generate_report()
        st.insert(tk.END, json.dumps(report, indent=2))

if __name__ == "__main__":
    core = HALOSCore()
    app = HALOSApp(core)
    app.mainloop()
