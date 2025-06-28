# upgrade_finance_tab.py

import tkinter as tk
from tkinter import ttk, messagebox
from payments import start_payment, PaymentMethod

class HybridFinanceTab:
    def __init__(self, notebook):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="Upgrade")

        ttk.Label(self.tab, text="Upgrade to HALOS Pro", font=("Helvetica", 16)).pack(pady=10)

        # Payment method selector
        method_frame = ttk.Frame(self.tab)
        method_frame.pack(pady=5)

        ttk.Label(method_frame, text="Select Payment Method:").pack(side=tk.LEFT)

        self.payment_method = tk.StringVar(value="stripe")
        ttk.Radiobutton(method_frame, text="Stripe (Cards, ApplePay)", variable=self.payment_method, value="stripe").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(method_frame, text="UPI (India)", variable=self.payment_method, value="upi").pack(side=tk.LEFT, padx=5)

        # Amount entry
        form_frame = ttk.Frame(self.tab)
        form_frame.pack(pady=5)

        ttk.Label(form_frame, text="Amount:").grid(row=0, column=0, sticky=tk.W)
        self.amount_entry = ttk.Entry(form_frame)
        self.amount_entry.grid(row=0, column=1, padx=5)
        self.amount_entry.insert(0, "49.99")

        # Pay button
        ttk.Button(self.tab, text="Pay Now", command=self.start_payment).pack(pady=15)

    def start_payment(self):
        method = PaymentMethod.STRIPE if self.payment_method.get() == "stripe" else PaymentMethod.UPI
        try:
            amount = float(self.amount_entry.get())
            user_info = {
                "email": "customer@example.com",
                "name": "Customer Name",
                "contact": "+919999999999",
                "user_id": "12345"
            }
            start_payment(method, amount, user_info)
        except Exception as e:
            messagebox.showerror("Payment Error", str(e))
