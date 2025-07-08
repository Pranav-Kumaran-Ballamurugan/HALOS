# transaction_manager.py

import sqlite3
from typing import List, Dict, Optional

class TransactionManager:
    def __init__(self, db_path="transactions.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_transactions_table()

    def create_transactions_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    date TEXT,
                    amount REAL,
                    currency TEXT,
                    method TEXT,
                    status TEXT,
                    user_email TEXT,
                    metadata TEXT
                )
            """)

    def add_transaction(self, txn: Dict) -> bool:
        try:
            with self.conn:
                self.conn.execute("""
                    INSERT INTO transactions (id, date, amount, currency, method, status, user_email, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    txn['id'], txn['date'], txn['amount'], txn['currency'],
                    txn['method'], txn['status'], txn['user_email'],
                    json.dumps(txn.get('metadata', {}))
                ))
            return True
        except sqlite3.IntegrityError:
            return False

    def update_transaction_status(self, txn_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute("""
                UPDATE transactions SET status=? WHERE id=?
            """, (status, txn_id))

    def get_transaction(self, txn_id: str) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM transactions WHERE id=?", (txn_id,))
        row = cur.fetchone()
        return self.row_to_dict(row) if row else None

    def get_recent_transactions(self, limit=100) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM transactions ORDER BY date DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [self.row_to_dict(row) for row in rows]

    def row_to_dict(self, row) -> Dict:
        return {
            "id": row[0], "date": row[1], "amount": row[2], "currency": row[3],
            "method": row[4], "status": row[5], "user_email": row[6],
            "metadata": json.loads(row[7])
        }
