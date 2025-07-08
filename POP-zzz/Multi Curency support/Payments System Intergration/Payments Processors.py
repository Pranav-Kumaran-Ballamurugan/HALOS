# Add after FinanceTrackerPlus class

class StripeProcessor:
    """Stripe payment processor with multi-currency support"""
    def __init__(self):
        self.client = stripe.Stripe(os.getenv('STRIPE_API_KEY'))
        
    def process(self, amount: float, currency: str, description: str) -> Dict:
        try:
            payment_intent = self.client.PaymentIntent.create(
                amount=int(amount * 100),  # Convert to cents
                currency=currency.lower(),
                description=description
            )
            return {
                'success': True,
                'transaction_id': payment_intent.id,
                'metadata': {
                    'stripe_object': payment_intent.to_dict()
                }
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class UPIProcessor:
    """UPI payment processor"""
    def __init__(self):
        # Initialize UPI SDK or API client
        self.upi_client = UPIWrapper(os.getenv('UPI_API_KEY'))
        
    def process(self, amount: float, currency: str, description: str) -> Dict:
        try:
            if currency != 'INR':
                return {'success': False, 'error': 'UPI only supports INR'}
                
            response = self.upi_client.initiate_transaction(
                amount=amount,
                note=description
            )
            return {
                'success': response['status'] == 'SUCCESS',
                'transaction_id': response['transaction_id'],
                'metadata': response
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}