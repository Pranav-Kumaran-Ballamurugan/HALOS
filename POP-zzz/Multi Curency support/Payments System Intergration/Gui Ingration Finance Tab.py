# Replace setup_finance_tab() method in HALOSApp class

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
    self.payment_currency.set('USD')
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
    transactions = self.core.finance_tracker.transactions
    
    # Add to treeview
    for t in sorted(transactions, key=lambda x: x.timestamp, reverse=True):
        self.transactions_tree.insert("", tk.END, values=(
            t.timestamp.strftime("%Y-%m-%d %H:%M"),
            f"{t.amount:.2f}",
            t.currency,
            t.method.name,
            t.status.name,
            t.description
        ))