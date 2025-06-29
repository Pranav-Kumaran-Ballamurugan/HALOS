# Replace the placeholder FinanceTrackerPlus class with this implementation

class FinanceTrackerPlus:
    """Enhanced financial management with payment processing"""
    def __init__(self):
        self.transactions = []
        self.payment_processors = {
            PaymentMethod.STRIPE: StripeProcessor(),
            PaymentMethod.UPI: UPIProcessor()
        }
        self.analytics_engine = FinanceAnalytics()
        self.balance = 0.0
        
    def process_payment(self, amount: float, currency: str, 
                       method: PaymentMethod, description: str) -> Dict:
        """Process a payment through the selected method"""
        processor = self.payment_processors.get(method)
        if not processor:
            return {'success': False, 'error': 'Payment method not supported'}
            
        try:
            result = processor.process(amount, currency, description)
            transaction = Transaction(
                amount=amount,
                currency=currency,
                method=method,
                status=PaymentStatus.COMPLETED if result['success'] else PaymentStatus.FAILED,
                timestamp=datetime.now(),
                description=description,
                metadata=result.get('metadata'),
                transaction_id=result.get('transaction_id')
            )
            self._record_transaction(transaction)
            return {'success': result['success'], 'transaction': transaction}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _record_transaction(self, transaction: Transaction):
        """Record transaction and update balance"""
        self.transactions.append(transaction)
        if transaction.status == PaymentStatus.COMPLETED:
            self.balance += transaction.amount
        self.analytics_engine.update(transaction)
    
    def analyze_spending(self, period: str = 'monthly') -> Dict:
        """Generate spending analytics for the specified period"""
        return self.analytics_engine.generate_report(period)
    
    def get_balance(self) -> float:
        """Get current balance"""
        return self.balance