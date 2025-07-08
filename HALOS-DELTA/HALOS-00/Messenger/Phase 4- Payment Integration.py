# halos/campaigns/payments.py
import stripe

class PaymentHandler:
    def __init__(self):
        stripe.api_key = os.getenv("STRIPE_KEY")
        
    async def split_payment(self, campaign_id: str, amounts: Dict[str: float]):
        """Handle multi-party payments"""
        transfer_group = f"campaign_{campaign_id}"
        
        # Create charges
        for user_id, amount in amounts.items():
            await stripe.Charge.create(
                amount=int(amount * 100),
                currency="usd",
                source=user_id,
                transfer_group=transfer_group
            )
        
        # Batch transfers
        await stripe.Transfer.create(
            amount=sum(amounts.values()) * 100,
            currency="usd",
            destination=VENUE_WALLET_ID,
            transfer_group=transfer_group
        )