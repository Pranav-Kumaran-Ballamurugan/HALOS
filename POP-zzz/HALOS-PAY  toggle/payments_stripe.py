# payments_stripe.py

import stripe
import os

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

def create_stripe_checkout(amount: float, user_info: dict) -> str:
    """
    Create a Stripe checkout session.

    Args:
        amount: Amount in dollars
        user_info: Dict with customer info (expects at least 'email')

    Returns:
        URL to Stripe checkout session
    """
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": "HALOS Pro License",
                },
                "unit_amount": int(amount * 100),  # in cents
            },
            "quantity": 1,
        }],
        mode="payment",
        customer_email=user_info.get("email"),
        success_url="http://localhost/payment-success",
        cancel_url="http://localhost/payment-cancel",
    )
    return session.url
