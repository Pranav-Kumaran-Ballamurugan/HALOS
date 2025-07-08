# payments_stripe.py
import stripe
import os
from typing import Dict
from pydantic import BaseModel, EmailStr, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UserInfo(BaseModel):
    """Validate and structure user information."""
    email: EmailStr
    name: str = None
    user_id: str = None

class StripeConfig:
    """Configuration for Stripe payments."""
    PRODUCT_NAME = "HALOS Pro License"
    SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", "http://localhost/payment-success")
    CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", "http://localhost/payment-cancel")
    CURRENCY = "usd"

def initialize_stripe():
    """Initialize Stripe with API key."""
    stripe_api_key = os.getenv("STRIPE_SECRET_KEY")
    if not stripe_api_key:
        raise ValueError("STRIPE_SECRET_KEY environment variable not set")
    stripe.api_key = stripe_api_key

initialize_stripe()

def create_stripe_checkout(amount: float, user_info: Dict) -> str:
    """
    Create a Stripe checkout session.

    Args:
        amount: Amount in dollars (will be converted to cents)
        user_info: Dictionary containing:
            - email: Customer email (required)
            - name: Customer name (optional)
            - user_id: Internal user reference (optional)

    Returns:
        URL to Stripe checkout session

    Raises:
        ValueError: If amount is invalid or user info is missing email
        stripe.error.StripeError: For Stripe API failures
    """
    try:
        # Validate inputs
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        validated_info = UserInfo(**user_info)
        
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": StripeConfig.CURRENCY,
                    "product_data": {
                        "name": StripeConfig.PRODUCT_NAME,
                    },
                    "unit_amount": int(amount * 100),  # Convert to cents
                },
                "quantity": 1,
            }],
            mode="payment",
            customer_email=validated_info.email,
            success_url=StripeConfig.SUCCESS_URL,
            cancel_url=StripeConfig.CANCEL_URL,
            metadata={
                "user_id": validated_info.user_id,
                "user_name": validated_info.name
            }
        )
        return session.url

    except stripe.error.StripeError as e:
        # Log the full error for debugging
        print(f"Stripe API error: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error creating Stripe session: {str(e)}")
        raise