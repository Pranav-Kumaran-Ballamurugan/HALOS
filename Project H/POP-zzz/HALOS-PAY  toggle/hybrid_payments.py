# payments.py
import webbrowser
from enum import Enum
from typing import Optional, Tuple
from payments_stripe import create_stripe_checkout
from payments_upi import create_upi_payment_link, verify_upi_payment

class PaymentMethod(Enum):
    STRIPE = "stripe"
    UPI = "upi"

def start_payment(method: PaymentMethod, amount: float, user_info: dict) -> None:
    """
    Initiate a payment process using the specified method.
    
    Args:
        method: Payment method (STRIPE or UPI)
        amount: Payment amount
        user_info: Dictionary containing user information
        
    Raises:
        ValueError: If unsupported payment method is provided
        RuntimeError: If payment initiation fails
    """
    try:
        if method == PaymentMethod.STRIPE:
            url = create_stripe_checkout(amount, user_info)
            webbrowser.open(url)
        elif method == PaymentMethod.UPI:
            link = create_upi_payment_link(amount, user_info)
            webbrowser.open(link)
        else:
            raise ValueError(f"Unsupported payment method: {method}")
    except Exception as e:
        raise RuntimeError(f"Payment initiation failed: {str(e)}")

def verify_payment(method: PaymentMethod, payment_id: str) -> Tuple[bool, Optional[str]]:
    """
    Verify the status of a payment.
    
    Args:
        method: Payment method used
        payment_id: The payment identifier to verify
        
    Returns:
        Tuple of (success_status, optional_error_message)
    """
    try:
        if method == PaymentMethod.UPI:
            return verify_upi_payment(payment_id), None
        elif method == PaymentMethod.STRIPE:
            # Implement Stripe verification logic here
            # Could check webhook events or use Stripe API
            raise NotImplementedError("Stripe verification not implemented")
        else:
            raise ValueError(f"Unsupported payment method: {method}")
    except Exception as e:
        return False, str(e)
