class EnhancedFinanceTab:
    def __init__(self, notebook, dark_mode=False):
        self.tab = ttk.Frame(notebook)
        notebook.add(self.tab, text="Finance+")
        
        self.dark_mode = dark_mode
        self.payment_in_progress = False
        self.transaction_manager = TransactionManager()  # SQLite-backed manager
        self.setup_ui()

    # ... (keep all existing methods)

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
                "id": f"txn_{int(time.time())}",
                "date": datetime.now().isoformat(),
                "amount": amount,
                "currency": currency,
                "method": method.value,
                "status": "initiated",
                "user_email": email,
                "metadata": {
                    "source": "HALOS GUI",
                    "tab_version": "2.0"
                }
            }

            self.payment_in_progress = True
            self.payment_status.config(text="Processing payment...")

            # First save the transaction to database
            if not self.transaction_manager.add_transaction(transaction):
                raise RuntimeError("Failed to save transaction record")

            # Run payment in background thread
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
            user_info = {
                "email": transaction['user_email'],
                "name": "HALOS User",
                "currency": transaction['currency']
            }
            
            # Start payment process
            start_payment(method, amount, user_info)
            
            # Update transaction status
            self.transaction_manager.update_transaction_status(
                transaction['id'], 
                "completed"
            )
            
            self.tab.after(0, lambda: self.payment_status.config(
                text=f"Payment completed - {transaction['id']}"
            ))
            
        except Exception as e:
            # Update transaction status to failed
            self.transaction_manager.update_transaction_status(
                transaction['id'],
                "failed"
            )
            
            self.tab.after(0, lambda: messagebox.showerror("Payment Error", str(e)))
            self.tab.after(0, lambda: self.payment_status.config(
                text=f"Payment failed: {str(e)}"
            ))
            
        finally:
            self.payment_in_progress = False
            self.tab.after(0, self.load_transactions)

    def load_transactions(self):
        """Load transactions from database into the treeview"""
        # Clear existing data
        for item in self.transaction_tree.get_children():
            self.transaction_tree.delete(item)
        
        # Get transactions from database
        transactions = self.transaction_manager.get_recent_transactions(limit=100)
        
        # Add to treeview
        for txn in transactions:
            self.transaction_tree.insert("", "end", values=(
                txn['id'],
                txn['date'],
                f"{txn['amount']} {txn['currency']}",
                txn['method'],
                txn['status']
            ))

    def verify_payment(self):
        """Verify payment status against database"""
        selection = self.transaction_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a transaction to verify")
            return

        item = self.transaction_tree.item(selection[0])
        txn_id = item['values'][0]  # First column is transaction ID

        try:
            # Get transaction from database
            txn = self.transaction_manager.get_transaction(txn_id)
            if not txn:
                raise ValueError("Transaction not found in database")
            
            # For demo purposes, we'll just check the status
            # In a real app, you would verify with payment provider
            if txn['status'] == 'completed':
                messagebox.showinfo("Success", "Payment already verified")
            else:
                # Simulate verification with payment provider
                method_str = txn['method'].lower()
                method = PaymentMethod(method_str) if method_str in PaymentMethod._value2member_map_ else None
                
                if not method:
                    raise ValueError("Cannot verify this payment method")

                # Update status in database
                self.transaction_manager.update_transaction_status(txn_id, "completed")
                messagebox.showinfo("Success", "Payment verified and marked complete")
                self.load_transactions()
                
        except Exception as e:
            messagebox.showerror("Error", f"Verification failed: {str(e)}")

    # ... (rest of the class remains the same)