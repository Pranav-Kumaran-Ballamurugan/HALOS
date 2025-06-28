# finance_analytics.py

from typing import List, Dict
from datetime import datetime
import sqlite3

class FinanceAnalytics:
    def __init__(self, db_path="halos_transactions.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_monthly_totals(self) -> List[Dict]:
        """
        Get total payments aggregated by month
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT substr(date, 1, 7) AS month, SUM(amount) AS total_amount, COUNT(*) AS count
                FROM transactions
                WHERE status = 'completed'
                GROUP BY month
                ORDER BY month DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_most_popular_methods(self) -> List[Dict]:
        """
        Get payment methods ranked by usage count
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT method, COUNT(*) AS usage_count, SUM(amount) AS total_amount
                FROM transactions
                WHERE status = 'completed'
                GROUP BY method
                ORDER BY usage_count DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_summary(self) -> Dict:
        """
        Get a quick financial summary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) AS total_transactions, SUM(amount) AS total_revenue
                FROM transactions
                WHERE status = 'completed'
            ''')
            row = cursor.fetchone()
            return dict(row) if row else {"total_transactions": 0, "total_revenue": 0.0}
