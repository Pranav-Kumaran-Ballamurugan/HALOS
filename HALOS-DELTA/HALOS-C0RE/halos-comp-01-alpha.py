#!/usr/bin/env python3
"""
HALOS V8.1 - Enhanced with Financial Visualization, Persistent Ledger, and Grafana Integration
"""

# [Previous imports remain the same, add these new ones]
import sqlite3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import psutil
from prometheus_client import Summary

# ======================
# ENHANCED FINANCE TRACKER WITH SQLITE
# ======================

class SecureSQLiteLedger:
    """Encrypted SQLite ledger for transaction persistence"""
    
    def __init__(self, db_path: str = 'halos_finance.db', encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self._init_db()
        
    def _init_db(self):
        """Initialize database with encrypted columns"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    description_encrypted TEXT NOT NULL,
                    metadata_encrypted TEXT,
                    transaction_id_encrypted TEXT,
                    audit_log_encrypted TEXT
                )
            """)
            conn.commit()

    def _encrypt_data(self, data: Optional[Any]) -> Optional[str]:
        """Encrypt data for storage"""
        if data is None:
            return None
        return self.cipher_suite.encrypt(json.dumps(data).encode()).decode()

    def _decrypt_data(self, encrypted: Optional[str]) -> Optional[Any]:
        """Decrypt data from storage"""
        if encrypted is None:
            return None
        return json.loads(self.cipher_suite.decrypt(encrypted.encode()).decode())

    def add_transaction(self, transaction: Transaction) -> int:
        """Add a transaction to the ledger"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (
                    amount, currency, method, status, timestamp,
                    description_encrypted, metadata_encrypted,
                    transaction_id_encrypted, audit_log_encrypted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                transaction.amount,
                transaction.currency,
                transaction.method.name,
                transaction.status.name,
                transaction.timestamp.isoformat(),
                self._encrypt_data(transaction.description),
                self._encrypt_data(transaction.metadata),
                self._encrypt_data(transaction.transaction_id),
                self._encrypt_data(transaction.audit_log)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_transactions(self, limit: int = 100) -> List[Transaction]:
        """Retrieve transactions from the ledger"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM transactions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append(Transaction(
                    amount=row['amount'],
                    currency=row['currency'],
                    method=PaymentMethod[row['method']],
                    status=PaymentStatus[row['status']],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    description=self._decrypt_data(row['description_encrypted']),
                    metadata=self._decrypt_data(row['metadata_encrypted']),
                    transaction_id=self._decrypt_data(row['transaction_id_encrypted']),
                    audit_log=self._decrypt_data(row['audit_log_encrypted'])
                ))
            return transactions

    def get_transactions_by_period(self, start: datetime, end: datetime) -> List[Transaction]:
        """Get transactions within a date range"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM transactions
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (start.isoformat(), end.isoformat()))
            
            transactions = []
            for row in cursor.fetchall():
                transactions.append(Transaction(
                    amount=row['amount'],
                    currency=row['currency'],
                    method=PaymentMethod[row['method']],
                    status=PaymentStatus[row['status']],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    description=self._decrypt_data(row['description_encrypted']),
                    metadata=self._decrypt_data(row['metadata_encrypted']),
                    transaction_id=self._decrypt_data(row['transaction_id_encrypted']),
                    audit_log=self._decrypt_data(row['audit_log_encrypted'])
                ))
            return transactions

class FinanceTrackerPlus:
    """Updated FinanceTracker with SQLite persistence and enhanced analytics"""
    
    def __init__(self):
        # Initialize SQLite ledger
        self.ledger = SecureSQLiteLedger(
            db_path='halos_finance.db',
            encryption_key=base64.urlsafe_b64encode(hashlib.sha256(
                os.getenv('ENCRYPTION_SECRET', 'default-secret').encode()
            ).digest())
        )
        
        # Load initial balance from ledger
        self.balance = self._calculate_initial_balance()
        self.analytics_engine = FinanceAnalytics()
        self.payment_processors = {
            PaymentMethod.STRIPE: StripeProcessor(),
            PaymentMethod.UPI: UPIProcessor()
        }
        self._transaction_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.new_event_loop()
        self.security = SecuritySuite()
        self.logger = logging.getLogger("HALOS.FinanceTracker")
        
        # Start async loop in background
        threading.Thread(target=self._start_async_loop, daemon=True).start()

    def _calculate_initial_balance(self) -> float:
        """Calculate balance from existing transactions"""
        transactions = self.ledger.get_transactions(limit=10000)
        return sum(t.amount for t in transactions if t.status == PaymentStatus.COMPLETED)

    async def get_transactions(self, limit: int = 100) -> List[Transaction]:
        """Get transactions from persistent ledger"""
        return self.ledger.get_transactions(limit=limit)

    async def _record_transaction(self, transaction: Transaction):
        """Record transaction to persistent storage"""
        try:
            # Add to SQLite ledger
            self.ledger.add_transaction(transaction)
            
            # Update balance if completed
            if transaction.status == PaymentStatus.COMPLETED:
                self.balance += transaction.amount
                
            # Update analytics
            await self.analytics_engine.update(transaction)
            
        except Exception as e:
            self.logger.error(f"Failed to record transaction: {str(e)}")
            raise

# ======================
# ENHANCED GUI WITH FINANCE VISUALIZATION
# ======================

class HALOSApp(tk.Tk):
    # ... (previous methods remain the same)
    
    def setup_finance_tab(self):
        """Enhanced Finance tab with visualization"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Finance")
        
        finance_notebook = ttk.Notebook(tab)
        finance_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Payment Frame
        payment_frame = ttk.Frame(finance_notebook)
        self.setup_payment_controls(payment_frame)
        finance_notebook.add(payment_frame, text="Payments")
        
        # Analytics Frame
        analytics_frame = ttk.Frame(finance_notebook)
        self.setup_analytics_controls(analytics_frame)
        finance_notebook.add(analytics_frame, text="Analytics")
        
        # Transactions Frame
        transactions_frame = ttk.Frame(finance_notebook)
        self.setup_transactions_table(transactions_frame)
        finance_notebook.add(transactions_frame, text="Transactions")
        
        # Visualization Frame
        viz_frame = ttk.Frame(finance_notebook)
        self.setup_visualization_controls(viz_frame)
        finance_notebook.add(viz_frame, text="Visualizations")

    def setup_visualization_controls(self, parent):
        """Setup spending visualization controls"""
        frame = ttk.LabelFrame(parent, text="Spending Visualizations")
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualization type selector
        viz_type_frame = ttk.Frame(frame)
        viz_type_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(viz_type_frame, text="Chart Type:").pack(side=tk.LEFT)
        self.viz_type = ttk.Combobox(viz_type_frame, 
                                    values=['Pie - By Category', 
                                            'Bar - By Method',
                                            'Line - Daily Spending',
                                            'Bar - Monthly Trends'])
        self.viz_type.set('Pie - By Category')
        self.viz_type.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(viz_type_frame, text="Generate", 
                  command=self.generate_visualization).pack(side=tk.LEFT)
        
        # Time period selector
        period_frame = ttk.Frame(frame)
        period_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(period_frame, text="Time Period:").pack(side=tk.LEFT)
        self.viz_period = ttk.Combobox(period_frame, 
                                      values=['Last 7 Days', 
                                              'Last 30 Days',
                                              'Current Month',
                                              'Current Year',
                                              'All Time'])
        self.viz_period.set('Last 30 Days')
        self.viz_period.pack(side=tk.LEFT, padx=5)
        
        # Matplotlib canvas
        self.viz_figure = Figure(figsize=(8, 4), dpi=100)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_figure, master=frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        self.viz_toolbar = NavigationToolbar2Tk(self.viz_canvas, frame)
        self.viz_toolbar.update()
        self.viz_canvas._tkcanvas.pack(fill=tk.X)

    def generate_visualization(self):
        """Generate and display the selected visualization"""
        try:
            chart_type = self.viz_type.get()
            time_period = self.viz_period.get()
            
            # Get data based on time period
            transactions = self._get_transactions_for_period(time_period)
            
            if not transactions:
                messagebox.showinfo("Info", "No transactions found for selected period")
                return
            
            # Clear previous figure
            self.viz_figure.clf()
            ax = self.viz_figure.add_subplot(111)
            
            if chart_type == 'Pie - By Category':
                self._generate_pie_chart(ax, transactions)
            elif chart_type == 'Bar - By Method':
                self._generate_method_bar_chart(ax, transactions)
            elif chart_type == 'Line - Daily Spending':
                self._generate_daily_line_chart(ax, transactions)
            elif chart_type == 'Bar - Monthly Trends':
                self._generate_monthly_bar_chart(ax, transactions)
                
            self.viz_figure.tight_layout()
            self.viz_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")

    def _get_transactions_for_period(self, period: str) -> List[Transaction]:
        """Get transactions for the specified time period"""
        now = datetime.now()
        
        if period == 'Last 7 Days':
            cutoff = now - timedelta(days=7)
            return self.core.finance_tracker.ledger.get_transactions_by_period(cutoff, now)
        elif period == 'Last 30 Days':
            cutoff = now - timedelta(days=30)
            return self.core.finance_tracker.ledger.get_transactions_by_period(cutoff, now)
        elif period == 'Current Month':
            start = datetime(now.year, now.month, 1)
            return self.core.finance_tracker.ledger.get_transactions_by_period(start, now)
        elif period == 'Current Year':
            start = datetime(now.year, 1, 1)
            return self.core.finance_tracker.ledger.get_transactions_by_period(start, now)
        else:  # All Time
            return self.core.finance_tracker.ledger.get_transactions(limit=1000)

    def _generate_pie_chart(self, ax, transactions: List[Transaction]):
        """Generate pie chart by spending category"""
        categories = defaultdict(float)
        for t in transactions:
            category = self.core.finance_tracker.analytics_engine._extract_category(t.description)
            categories[category] += t.amount
        
        labels = list(categories.keys())
        sizes = list(categories.values())
        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Spending by Category')

    def _generate_method_bar_chart(self, ax, transactions: List[Transaction]):
        """Generate bar chart by payment method"""
        methods = defaultdict(float)
        for t in transactions:
            methods[t.method.name] += t.amount
        
        methods = dict(sorted(methods.items(), key=lambda item: item[1], reverse=True))
        
        ax.bar(methods.keys(), methods.values())
        ax.set_title('Spending by Payment Method')
        ax.set_ylabel('Amount')
        ax.tick_params(axis='x', rotation=45)

    def _generate_daily_line_chart(self, ax, transactions: List[Transaction]):
        """Generate line chart of daily spending"""
        daily = defaultdict(float)
        for t in transactions:
            date = t.timestamp.date()
            daily[date] += t.amount
        
        dates = sorted(daily.keys())
        amounts = [daily[date] for date in dates]
        
        ax.plot(dates, amounts, marker='o')
        ax.set_title('Daily Spending')
        ax.set_ylabel('Amount')
        ax.tick_params(axis='x', rotation=45)

    def _generate_monthly_bar_chart(self, ax, transactions: List[Transaction]):
        """Generate bar chart of monthly trends"""
        monthly = defaultdict(float)
        for t in transactions:
            month_year = f"{t.timestamp.year}-{t.timestamp.month:02d}"
            monthly[month_year] += t.amount
        
        months = sorted(monthly.keys())
        amounts = [monthly[month] for month in months]
        
        ax.bar(months, amounts)
        ax.set_title('Monthly Spending Trends')
        ax.set_ylabel('Amount')
        ax.tick_params(axis='x', rotation=45)

# ======================
# ENHANCED METRICS FOR GRAFANA
# ======================

class HALOSCore:
    # ... (previous methods remain the same)
    
    def _start_metrics_server(self):
        """Start Prometheus metrics server with additional metrics"""
        try:
            # Define additional metrics
            self.MEMORY_USAGE = Gauge('halos_memory_usage_bytes', 'Memory usage in bytes')
            self.THREAD_COUNT = Gauge('halos_thread_count', 'Number of active threads')
            self.TASK_QUEUE_SIZE = Gauge('halos_task_queue_size', 'Number of pending tasks')
            self.TASK_DURATION = Histogram(
                'halos_task_duration_seconds',
                'Task processing duration',
                ['task_type']
            )
            self.LLM_RESPONSE_TIME = Summary(
                'halos_llm_response_time_seconds',
                'LLM response time summary',
                ['provider']
            )
            self.DB_OPERATION_TIME = Histogram(
                'halos_db_operation_duration_seconds',
                'Database operation duration',
                ['operation']
            )
            
            # Start server
            start_http_server(8000)
            self.logger.info("Metrics server started on port 8000")
            
            # Start metrics reporter
            threading.Thread(target=self._report_system_metrics, daemon=True).start()
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {str(e)}")

    def _report_system_metrics(self):
        """Continuously report system metrics"""
        while True:
            try:
                # Report memory usage
                self.MEMORY_USAGE.set(psutil.Process().memory_info().rss)
                
                # Report thread count
                self.THREAD_COUNT.set(threading.active_count())
                
                # Report queue sizes
                self.TASK_QUEUE_SIZE.set(self.task_queue.qsize() if hasattr(self, 'task_queue') else 0)
                
                time.sleep(5)
            except Exception as e:
                self.logger.warning(f"Metrics reporting error: {str(e)}")
                time.sleep(30)

# ======================
# GRAFANA DASHBOARD CONFIGURATION
# ======================

def setup_grafana_dashboard():
    """Generate Grafana dashboard configuration"""
    dashboard = {
        "title": "HALOS System Metrics",
        "description": "Comprehensive monitoring of HALOS system performance",
        "panels": [
            {
                "title": "System Requests",
                "type": "graph",
                "targets": [
                    {"expr": "rate(halos_requests_total[1m])", "legendFormat": "Requests/s"},
                    {"expr": "rate(halos_errors_total[1m])", "legendFormat": "Errors/s"}
                ]
            },
            {
                "title": "Resource Usage",
                "type": "gauge",
                "targets": [
                    {"expr": "halos_memory_usage_bytes / (1024^2)", "legendFormat": "Memory (MB)"},
                    {"expr": "halos_thread_count", "legendFormat": "Threads"}
                ]
            },
            {
                "title": "Payment Processing",
                "type": "stat",
                "targets": [
                    {"expr": "sum(halos_payments_total)", "legendFormat": "Total Payments"}
                ]
            }
        ]
    }
    
    # Save dashboard config
    with open('halos_grafana_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)

# ======================
# MAIN APPLICATION
# ======================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('halos.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("HALOS")
    
    try:
        # Generate Grafana dashboard config
        setup_grafana_dashboard()
        
        # Initialize core system
        logger.info("Starting HALOS core system")
        core_system = HALOSCore()
        
        # Start GUI
        logger.info("Starting HALOS GUI")
        app = HALOSApp(core_system)
        
        # Start main loop
        logger.info("HALOS system ready")
        app.mainloop()
        
    except Exception as e:
        logger.critical(f"Fatal error during startup: {str(e)}")
        raise