# finance_analytics.py
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import sqlite3
import calendar
from collections import defaultdict
import json
from enum import Enum

class TimePeriod(Enum):
    """Enum for different time period types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class FinanceAnalytics:
    """Enhanced financial analytics with comprehensive reporting capabilities"""
    
    def __init__(self, db_path: str = "halos_transactions.db"):
        self.db_path = db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_summary(self, days: int = 30) -> Dict:
        """
        Get a comprehensive financial summary
        
        Args:
            days: Number of days to include in recent activity
            
        Returns:
            Dictionary containing various summary metrics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate date ranges
            today = datetime.now().date()
            start_date = (today - timedelta(days=days)).isoformat()
            
            summary = {
                "time_period": f"Last {days} days",
                "start_date": start_date,
                "end_date": today.isoformat()
            }
            
            # Total transactions and revenue
            cursor.execute('''
                SELECT 
                    COUNT(*) AS total_transactions,
                    SUM(amount) AS total_revenue,
                    AVG(amount) AS avg_transaction
                FROM transactions
                WHERE status = 'completed'
                AND date >= ?
            ''', (start_date,))
            summary.update(dict(cursor.fetchone()))
            
            # Failed transactions
            cursor.execute('''
                SELECT COUNT(*) AS failed_transactions
                FROM transactions
                WHERE status = 'failed'
                AND date >= ?
            ''', (start_date,))
            summary.update(dict(cursor.fetchone()))
            
            # Payment method distribution
            cursor.execute('''
                SELECT 
                    method,
                    COUNT(*) AS count,
                    SUM(amount) AS total_amount
                FROM transactions
                WHERE status = 'completed'
                AND date >= ?
                GROUP BY method
                ORDER BY total_amount DESC
            ''', (start_date,))
            summary["payment_methods"] = [dict(row) for row in cursor.fetchall()]
            
            # Recent growth metrics
            prev_period_start = (today - timedelta(days=days*2)).isoformat()
            cursor.execute('''
                SELECT SUM(amount) AS previous_revenue
                FROM transactions
                WHERE status = 'completed'
                AND date >= ? AND date < ?
            ''', (prev_period_start, start_date))
            prev_revenue = cursor.fetchone()["previous_revenue"] or 0
            current_revenue = summary["total_revenue"] or 0
            summary["revenue_growth_pct"] = (
                ((current_revenue - prev_revenue) / prev_revenue * 100) 
                if prev_revenue else float('inf')
            )
            
            return summary
    
    def get_totals_by_period(self, 
                           period: TimePeriod = TimePeriod.MONTHLY,
                           limit: int = 12) -> List[Dict]:
        """
        Get totals aggregated by time period
        
        Args:
            period: Time period to aggregate by
            limit: Number of periods to return
            
        Returns:
            List of period totals with metrics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Determine the SQL date grouping based on period
            if period == TimePeriod.DAILY:
                date_expr = "substr(date, 1, 10)"
                order_by = "period DESC"
            elif period == TimePeriod.WEEKLY:
                date_expr = "strftime('%Y-%W', date)"
                order_by = "period DESC"
            elif period == TimePeriod.MONTHLY:
                date_expr = "substr(date, 1, 7)"
                order_by = "period DESC"
            elif period == TimePeriod.QUARTERLY:
                date_expr = "strftime('%Y', date) || '-' || ((strftime('%m', date)-1)/3 + 1)"
                order_by = "period DESC"
            else:  # YEARLY
                date_expr = "substr(date, 1, 4)"
                order_by = "period DESC"
            
            cursor.execute(f'''
                SELECT 
                    {date_expr} AS period,
                    SUM(amount) AS total_amount,
                    COUNT(*) AS transaction_count,
                    AVG(amount) AS avg_amount,
                    MIN(amount) AS min_amount,
                    MAX(amount) AS max_amount
                FROM transactions
                WHERE status = 'completed'
                GROUP BY period
                ORDER BY {order_by}
                LIMIT ?
            ''', (limit,))
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # Format periods for better readability
            for result in results:
                if period == TimePeriod.QUARTERLY:
                    year, quarter = result['period'].split('-')
                    result['period_label'] = f"Q{quarter} {year}"
                elif period == TimePeriod.WEEKLY:
                    year, week = result['period'].split('-')
                    result['period_label'] = f"Week {week}, {year}"
                else:
                    result['period_label'] = result['period']
            
            return results
    
    def get_payment_method_analysis(self) -> Dict:
        """
        Get detailed analysis of payment methods
        
        Returns:
            Dictionary with method statistics and trends
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            analysis = {}
            
            # Overall method distribution
            cursor.execute('''
                SELECT 
                    method,
                    COUNT(*) AS total_count,
                    SUM(amount) AS total_amount,
                    AVG(amount) AS avg_amount,
                    100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions) AS percentage
                FROM transactions
                WHERE status = 'completed'
                GROUP BY method
                ORDER BY total_amount DESC
            ''')
            analysis["methods"] = [dict(row) for row in cursor.fetchall()]
            
            # Monthly trends for each method
            cursor.execute('''
                SELECT 
                    substr(date, 1, 7) AS month,
                    method,
                    COUNT(*) AS count,
                    SUM(amount) AS amount
                FROM transactions
                WHERE status = 'completed'
                GROUP BY month, method
                ORDER BY month DESC, amount DESC
            ''')
            
            monthly_data = cursor.fetchall()
            analysis["monthly_trends"] = self._format_method_trends(monthly_data)
            
            # Success rates by method
            cursor.execute('''
                SELECT 
                    method,
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS success_count,
                    100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*) AS success_rate
                FROM transactions
                GROUP BY method
                ORDER BY success_rate DESC
            ''')
            analysis["success_rates"] = [dict(row) for row in cursor.fetchall()]
            
            return analysis
    
    def _format_method_trends(self, rows) -> Dict:
        """Format monthly method trends into a nested structure"""
        trends = defaultdict(dict)
        
        for row in rows:
            month = row['month']
            method = row['method']
            trends[month][method] = {
                'count': row['count'],
                'amount': row['amount']
            }
        
        # Convert to list of months with method data
        return [
            {'month': month, 'methods': methods}
            for month, methods in sorted(trends.items(), reverse=True)
        ]
    
    def get_customer_metrics(self, top_n: int = 10) -> Dict:
        """
        Get metrics about customers and their spending
        
        Args:
            top_n: Number of top customers to return
            
        Returns:
            Dictionary with customer metrics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            metrics = {}
            
            # Top customers by spending
            cursor.execute(f'''
                SELECT 
                    user_email,
                    COUNT(*) AS transaction_count,
                    SUM(amount) AS total_spent,
                    AVG(amount) AS avg_transaction,
                    MAX(date) AS last_purchase
                FROM transactions
                WHERE status = 'completed'
                GROUP BY user_email
                ORDER BY total_spent DESC
                LIMIT ?
            ''', (top_n,))
            metrics["top_customers"] = [dict(row) for row in cursor.fetchall()]
            
            # Customer acquisition over time
            cursor.execute('''
                SELECT 
                    substr(date, 1, 7) AS month,
                    COUNT(DISTINCT user_email) AS new_customers
                FROM transactions
                WHERE user_email NOT IN (
                    SELECT DISTINCT user_email 
                    FROM transactions 
                    WHERE date < substr(?, 1, 7) || '-01'
                )
                AND status = 'completed'
                GROUP BY month
                ORDER BY month DESC
            ''', (metrics["top_customers"][0]["last_purchase"],))
            metrics["customer_acquisition"] = [dict(row) for row in cursor.fetchall()]
            
            return metrics
    
    def get_failed_transactions_analysis(self) -> Dict:
        """
        Analyze failed transactions for patterns
        
        Returns:
            Dictionary with failure analysis
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            analysis = {}
            
            # Failure reasons by method (assuming reasons are in metadata)
            cursor.execute('''
                SELECT 
                    method,
                    COUNT(*) AS failure_count,
                    json_extract(metadata, '$.failure_reason') AS reason
                FROM transactions
                WHERE status = 'failed'
                GROUP BY method, reason
                ORDER BY failure_count DESC
            ''')
            
            failures = cursor.fetchall()
            analysis["failure_reasons"] = self._group_failure_reasons(failures)
            
            # Failure rate over time
            cursor.execute('''
                SELECT 
                    substr(date, 1, 7) AS month,
                    COUNT(*) AS total_transactions,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count,
                    100.0 * SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) / COUNT(*) AS failure_rate
                FROM transactions
                GROUP BY month
                ORDER BY month DESC
            ''')
            analysis["failure_trends"] = [dict(row) for row in cursor.fetchall()]
            
            return analysis
    
    def _group_failure_reasons(self, rows) -> Dict:
        """Group failure reasons by payment method"""
        reasons = defaultdict(list)
        
        for row in rows:
            method = row['method']
            reason = row['reason'] or "Unknown"
            reasons[method].append({
                'reason': reason,
                'count': row['failure_count']
            })
        
        return dict(reasons)
    
    def generate_report(self, report_type: str = "monthly") -> Dict:
        """
        Generate a comprehensive financial report
        
        Args:
            report_type: Type of report ('monthly', 'quarterly', 'yearly')
            
        Returns:
            Complete financial report
        """
        period = {
            'monthly': TimePeriod.MONTHLY,
            'quarterly': TimePeriod.QUARTERLY,
            'yearly': TimePeriod.YEARLY
        }.get(report_type, TimePeriod.MONTHLY)
        
        return {
            'summary': self.get_summary(days=30 if report_type == 'monthly' else 90 if report_type == 'quarterly' else 365),
            'period_totals': self.get_totals_by_period(period),
            'payment_methods': self.get_payment_method_analysis(),
            'customer_metrics': self.get_customer_metrics(),
            'failure_analysis': self.get_failed_transactions_analysis(),
            'generated_at': datetime.now().isoformat(),
            'report_type': report_type
        }