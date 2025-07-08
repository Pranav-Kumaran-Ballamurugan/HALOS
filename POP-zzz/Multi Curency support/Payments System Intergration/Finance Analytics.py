class FinanceAnalytics:
    """Financial analytics and reporting engine"""
    def __init__(self):
        self.data = []
        
    def update(self, transaction: Transaction):
        """Update analytics with new transaction"""
        self.data.append(transaction)
    
    def generate_report(self, period: str) -> Dict:
        """Generate analytics report for the given period"""
        now = datetime.now()
        filtered = []
        
        if period == 'daily':
            filtered = [t for t in self.data if t.timestamp.date() == now.date()]
        elif period == 'weekly':
            start = now - timedelta(days=now.weekday())
            filtered = [t for t in self.data if t.timestamp.date() >= start.date()]
        elif period == 'monthly':
            filtered = [t for t in self.data if t.timestamp.month == now.month]
        else:
            filtered = self.data
            
        return {
            'period': period,
            'total_transactions': len(filtered),
            'total_amount': sum(t.amount for t in filtered),
            'payment_methods': self._aggregate_by_method(filtered),
            'categories': self._aggregate_by_category(filtered)
        }
    
    def _aggregate_by_method(self, transactions):
        """Aggregate transactions by payment method"""
        result = defaultdict(float)
        for t in transactions:
            result[t.method.name] += t.amount
        return dict(result)
    
    def _aggregate_by_category(self, transactions):
        """Aggregate transactions by category (from description)"""
        result = defaultdict(float)
        for t in transactions:
            category = self._extract_category(t.description)
            result[category] += t.amount
        return dict(result)
    
    def _extract_category(self, description: str) -> str:
        """Extract category from description (simplified)"""
        desc = description.lower()
        if 'food' in desc or 'restaurant' in desc:
            return 'Food'
        elif 'rent' in desc:
            return 'Housing'
        elif 'transport' in desc or 'uber' in desc:
            return 'Transportation'
        return 'Other'