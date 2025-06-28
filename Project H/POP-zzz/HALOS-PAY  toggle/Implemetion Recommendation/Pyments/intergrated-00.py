# enhanced_finance_sqlite.py
import sqlite3
from sqlite3 import Error
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
import threading
from enum import Enum
from dataclasses import dataclass

class TransactionStatus(Enum):
    """Enum for transaction status values"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"

class PaymentMethod(Enum):
    """Enum for payment methods"""
    STRIPE = "stripe"
    UPI = "upi"
    BANK = "bank"
    CRYPTO = "crypto"

@dataclass
class Transaction:
    """Dataclass representing a transaction"""
    id: str
    date: str
    amount: float
    currency: str
    method: str
    status: str
    user_email: str
    metadata: Dict = None
    created_at: str = None

class TransactionManager:
    """Enhanced SQLite transaction manager with thread safety and advanced features"""
    
    def __init__(self, db_path: str = "halos_transactions.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database with proper schema and indexes"""
        with self._lock:
            conn = None
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Main transactions table
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
                
                # Create indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_transactions_email 
                    ON transactions(user_email)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_transactions_date 
                    ON transactions(date)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_transactions_status 
                    ON transactions(status)
                ''')
                
                # Create transaction audit log table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transaction_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transaction_id TEXT NOT NULL,
                        changed_field TEXT NOT NULL,
                        old_value TEXT,
                        new_value TEXT,
                        changed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        changed_by TEXT,
                        FOREIGN KEY(transaction_id) REFERENCES transactions(id)
                    )
                ''')
                
                conn.commit()
            except Error as e:
                raise RuntimeError(f"Database initialization failed: {str(e)}")
            finally:
                if conn:
                    conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with proper settings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        conn.execute("PRAGMA foreign_keys=ON")   # Enable foreign key constraints
        return conn

    def add_transaction(self, transaction: Dict) -> Tuple[bool, str]:
        """
        Add a new transaction with validation
        
        Args:
            transaction: Dictionary containing transaction data
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        required_fields = {
            'id', 'date', 'amount', 'currency', 
            'method', 'status', 'user_email'
        }
        
        if not all(field in transaction for field in required_fields):
            return False, f"Missing required fields: {required_fields}"
        
        try:
            # Validate payment method
            PaymentMethod(transaction['method'].lower())
            
            # Validate status
            TransactionStatus(transaction['status'].lower())
            
            # Prepare metadata
            metadata = transaction.get('metadata', {})
            if not isinstance(metadata, dict):
                return False, "Metadata must be a dictionary"
            
            with self._lock:
                conn = self._get_connection()
                try:
                    cursor = conn.cursor()
                    
                    # Insert transaction
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
                        json.dumps(metadata)
                    ))
                    
                    # Log the creation
                    cursor.execute('''
                        INSERT INTO transaction_audit
                        (transaction_id, changed_field, new_value)
                        VALUES (?, ?, ?)
                    ''', (
                        transaction['id'],
                        'status',
                        transaction['status']
                    ))
                    
                    conn.commit()
                    return True, "Transaction added successfully"
                except sqlite3.IntegrityError:
                    return False, "Transaction ID already exists"
                except Error as e:
                    return False, f"Database error: {str(e)}"
                finally:
                    conn.close()
        except ValueError as e:
            return False, f"Validation error: {str(e)}"

    def update_transaction_status(self, 
                                transaction_id: str, 
                                new_status: str,
                                changed_by: str = "system") -> Tuple[bool, str]:
        """
        Update a transaction's status with audit logging
        
        Args:
            transaction_id: ID of transaction to update
            new_status: New status value
            changed_by: Who is making the change
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            new_status = TransactionStatus(new_status.lower())
        except ValueError:
            return False, f"Invalid status. Must be one of: {[s.value for s in TransactionStatus]}"
        
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Get current status
                cursor.execute('''
                    SELECT status FROM transactions WHERE id = ?
                ''', (transaction_id,))
                result = cursor.fetchone()
                
                if not result:
                    return False, "Transaction not found"
                
                old_status = result['status']
                
                # Update status
                cursor.execute('''
                    UPDATE transactions
                    SET status = ?
                    WHERE id = ?
                ''', (new_status.value, transaction_id))
                
                # Log the change
                cursor.execute('''
                    INSERT INTO transaction_audit
                    (transaction_id, changed_field, old_value, new_value, changed_by)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    transaction_id,
                    'status',
                    old_status,
                    new_status.value,
                    changed_by
                ))
                
                conn.commit()
                return True, "Status updated successfully"
            except Error as e:
                return False, f"Database error: {str(e)}"
            finally:
                conn.close()

    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Retrieve a single transaction by ID"""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM transactions WHERE id = ?
                ''', (transaction_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_transaction(row)
                return None
            finally:
                conn.close()

    def get_transactions_by_user(self, 
                               email: str, 
                               limit: int = 100,
                               status: Optional[str] = None) -> List[Transaction]:
        """
        Get transactions for a specific user
        
        Args:
            email: User email to filter by
            limit: Maximum number of transactions to return
            status: Optional status filter
            
        Returns:
            List of Transaction objects
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM transactions 
                    WHERE user_email = ?
                '''
                params = [email]
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                query += " ORDER BY date DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return [self._row_to_transaction(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def get_recent_transactions(self, 
                              limit: int = 100,
                              status: Optional[str] = None) -> List[Transaction]:
        """
        Get most recent transactions
        
        Args:
            limit: Maximum number of transactions to return
            status: Optional status filter
            
        Returns:
            List of Transaction objects
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM transactions"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status)
                
                query += " ORDER BY date DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return [self._row_to_transaction(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def search_transactions(self,
                          amount_min: Optional[float] = None,
                          amount_max: Optional[float] = None,
                          date_start: Optional[str] = None,
                          date_end: Optional[str] = None,
                          status: Optional[str] = None,
                          method: Optional[str] = None) -> List[Transaction]:
        """
        Search transactions with multiple filters
        
        Args:
            amount_min: Minimum amount
            amount_max: Maximum amount
            date_start: Start date (ISO format)
            date_end: End date (ISO format)
            status: Transaction status
            method: Payment method
            
        Returns:
            List of matching Transaction objects
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                query = "SELECT * FROM transactions WHERE 1=1"
                params = []
                
                if amount_min is not None:
                    query += " AND amount >= ?"
                    params.append(amount_min)
                
                if amount_max is not None:
                    query += " AND amount <= ?"
                    params.append(amount_max)
                
                if date_start:
                    query += " AND date >= ?"
                    params.append(date_start)
                
                if date_end:
                    query += " AND date <= ?"
                    params.append(date_end)
                
                if status:
                    query += " AND status = ?"
                    params.append(status)
                
                if method:
                    query += " AND method = ?"
                    params.append(method)
                
                query += " ORDER BY date DESC"
                
                cursor.execute(query, params)
                return [self._row_to_transaction(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def get_transaction_audit_log(self, 
                                transaction_id: str,
                                limit: int = 10) -> List[Dict]:
        """
        Get audit log entries for a transaction
        
        Args:
            transaction_id: Transaction ID to get logs for
            limit: Maximum number of log entries to return
            
        Returns:
            List of audit log entries as dictionaries
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM transaction_audit
                    WHERE transaction_id = ?
                    ORDER BY changed_at DESC
                    LIMIT ?
                ''', (transaction_id, limit))
                
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def _row_to_transaction(self, row) -> Transaction:
        """Convert database row to Transaction object"""
        metadata = {}
        if row['metadata']:
            try:
                metadata = json.loads(row['metadata'])
            except json.JSONDecodeError:
                metadata = {}
        
        return Transaction(
            id=row['id'],
            date=row['date'],
            amount=row['amount'],
            currency=row['currency'],
            method=row['method'],
            status=row['status'],
            user_email=row['user_email'],
            metadata=metadata,
            created_at=row['created_at']
        )

    def __del__(self):
        """Clean up resources when the manager is destroyed"""
        if hasattr(self, '_conn'):
            self._conn.close()