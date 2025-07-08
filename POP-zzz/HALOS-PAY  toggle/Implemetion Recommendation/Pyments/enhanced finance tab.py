# transaction_manager.py
import sqlite3
from sqlite3 import Error
from datetime import datetime
from typing import List, Dict, Optional
import json

class TransactionManager:
    """SQLite-backed transaction manager with CRUD operations"""
    
    def __init__(self, db_path: str = "transactions.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database with required tables"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
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
                    plan TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_email 
                ON transactions(user_email)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_transactions_date 
                ON transactions(date)
            ''')
            
            conn.commit()
        except Error as e:
            print(f"Database initialization error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def add_transaction(self, transaction: Dict) -> bool:
        """Add a new transaction to the database"""
        required_fields = {'id', 'date', 'amount', 'currency', 'method', 'status', 'user_email'}
        if not all(field in transaction for field in required_fields):
            raise ValueError(f"Transaction missing required fields. Needs: {required_fields}")
        
        # Prepare metadata
        metadata = transaction.get('metadata', {})
        if 'plan' in transaction:
            metadata['plan'] = transaction['plan']
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO transactions 
                (id, date, amount, currency, method, status, user_email, plan, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction['id'],
                transaction['date'],
                transaction['amount'],
                transaction['currency'],
                transaction['method'],
                transaction['status'],
                transaction['user_email'],
                transaction.get('plan'),
                json.dumps(metadata)
            ))
            
            conn.commit()
            return True
        except Error as e:
            print(f"Error adding transaction: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def update_transaction_status(self, transaction_id: str, new_status: str) -> bool:
        """Update a transaction's status"""
        valid_statuses = {'pending', 'completed', 'failed', 'refunded', 'cancelled'}
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {valid_statuses}")
        
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE transactions
                SET status = ?
                WHERE id = ?
            ''', (new_status, transaction_id))
            
            conn.commit()
            return cursor.rowcount > 0
        except Error as e:
            print(f"Error updating transaction: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def get_transaction(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve a single transaction by ID"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Access columns by name
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM transactions
                WHERE id = ?
            ''', (transaction_id,))
            
            row = cursor.fetchone()
            return self._row_to_dict(row) if row else None
        except Error as e:
            print(f"Error fetching transaction: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def get_transactions_by_user(self, email: str, limit: int = 100) -> List[Dict]:
        """Get all transactions for a specific user"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM transactions
                WHERE user_email = ?
                ORDER BY date DESC
                LIMIT ?
            ''', (email, limit))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
        except Error as e:
            print(f"Error fetching user transactions: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def get_recent_transactions(self, limit: int = 50) -> List[Dict]:
        """Get most recent transactions across all users"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM transactions
                ORDER BY date DESC
                LIMIT ?
            ''', (limit,))
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
        except Error as e:
            print(f"Error fetching recent transactions: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def search_transactions(self, 
                          amount_min: Optional[float] = None,
                          amount_max: Optional[float] = None,
                          date_start: Optional[str] = None,
                          date_end: Optional[str] = None,
                          status: Optional[str] = None,
                          method: Optional[str] = None) -> List[Dict]:
        """Search transactions with filters"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
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
            return [self._row_to_dict(row) for row in cursor.fetchall()]
        except Error as e:
            print(f"Error searching transactions: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert SQLite row to dictionary"""
        transaction = dict(row)
        
        # Parse metadata JSON if present
        if 'metadata' in transaction and transaction['metadata']:
            try:
                transaction['metadata'] = json.loads(transaction['metadata'])
            except json.JSONDecodeError:
                transaction['metadata'] = {}
        
        return transaction

    def delete_transaction(self, transaction_id: str) -> bool:
        """Delete a transaction by ID"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM transactions
                WHERE id = ?
            ''', (transaction_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        except Error as e:
            print(f"Error deleting transaction: {e}")
            return False
        finally:
            if conn:
                conn.close()