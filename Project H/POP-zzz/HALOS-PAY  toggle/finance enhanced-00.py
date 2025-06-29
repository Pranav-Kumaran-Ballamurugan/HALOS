# payments.py
import webbrowser
from enum import Enum
from typing import Optional, Tuple, Dict
from decimal import Decimal
import stripe
from payments_stripe import create_stripe_checkout
from payments_upi import create_upi_payment_link, verify_upi_payment

class PaymentMethod(Enum):
    STRIPE = "stripe"
    UPI = "upi"

class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

def start_payment(
    method: PaymentMethod, 
    amount: Decimal, 
    currency: str,
    user_info: Dict,
    metadata: Optional[Dict] = None
) -> Tuple[str, PaymentStatus]:
    """
    Initiate a payment process using the specified method.
    
    Args:
        method: Payment method (STRIPE or UPI)
        amount: Payment amount as Decimal for precise arithmetic
        currency: 3-letter ISO currency code
        user_info: Dictionary containing user information
        metadata: Additional payment metadata
        
    Returns:
        Tuple of (payment_reference_id, initial_status)
        
    Raises:
        ValueError: If unsupported payment method or invalid amount
        RuntimeError: If payment initiation fails
    """
    try:
        # Validate amount
        if amount <= Decimal('0'):
            raise ValueError("Amount must be positive")
        
        if method == PaymentMethod.STRIPE:
            # Convert to cents/cents equivalent for Stripe
            amount_int = int(amount * Decimal('100'))
            payment_id = create_stripe_checkout(
                amount=amount_int,
                currency=currency,
                user_info=user_info,
                metadata=metadata
            )
            return payment_id, PaymentStatus.PENDING
            
        elif method == PaymentMethod.UPI:
            payment_url, payment_id = create_upi_payment_link(
                amount=float(amount),
                user_info=user_info,
                notes=metadata
            )
            webbrowser.open(payment_url)
            return payment_id, PaymentStatus.PENDING
            
        else:
            raise ValueError(f"Unsupported payment method: {method}")
            
    except Exception as e:
        raise RuntimeError(f"Payment initiation failed: {str(e)}")

def verify_payment(
    method: PaymentMethod, 
    payment_id: str,
    retry_count: int = 3
) -> Tuple[PaymentStatus, Optional[Dict]]:
    """
    Verify the status of a payment with retry logic.
    
    Args:
        method: Payment method used
        payment_id: The payment identifier to verify
        retry_count: Number of verification attempts
        
    Returns:
        Tuple of (payment_status, full_response_data)
    """
    last_error = None
    
    for attempt in range(retry_count):
        try:
            if method == PaymentMethod.UPI:
                is_verified, details = verify_upi_payment(payment_id)
                if is_verified:
                    return PaymentStatus.COMPLETED, details
                return PaymentStatus.PENDING, details
                    
            elif method == PaymentMethod.STRIPE:
                # Retrieve payment intent from Stripe
                payment_intent = stripe.PaymentIntent.retrieve(payment_id)
                
                if payment_intent.status == 'succeeded':
                    return PaymentStatus.COMPLETED, payment_intent.to_dict()
                elif payment_intent.status in ['processing', 'requires_action']:
                    return PaymentStatus.PENDING, payment_intent.to_dict()
                else:
                    return PaymentStatus.FAILED, payment_intent.to_dict()
                    
            else:
                raise ValueError(f"Unsupported payment method: {method}")
                
        except Exception as e:
            last_error = e
            if attempt < retry_count - 1:
                time.sleep(1)  # Wait before retrying
            continue
            
    raise RuntimeError(
        f"Payment verification failed after {retry_count} attempts: {str(last_error)}"
    )

def handle_webhook(
    method: PaymentMethod,
    payload: bytes,
    signature: str,
    secret: str
) -> Tuple[PaymentStatus, Optional[Dict]]:
    """
    Handle payment webhook notifications.
    
    Args:
        method: Payment method
        payload: Raw webhook payload
        signature: Signature header for verification
        secret: Webhook secret for verification
        
    Returns:
        Tuple of (payment_status, webhook_data)
    """
    try:
        if method == PaymentMethod.STRIPE:
            event = stripe.Webhook.construct_event(
                payload, signature, secret
            )
            
            if event.type == 'payment_intent.succeeded':
                data = event['data']['object']
                return PaymentStatus.COMPLETED, data
            elif event.type == 'payment_intent.payment_failed':
                data = event['data']['object']
                return PaymentStatus.FAILED, data
            else:
                return PaymentStatus.PENDING, event.to_dict()
                
        elif method == PaymentMethod.UPI:
            # UPI webhook handling would be implemented here
            raise NotImplementedError("UPI webhook handling not implemented")
            
        else:
            raise ValueError(f"Unsupported payment method: {method}")
            
    except Exception as e:
        raise RuntimeError(f"Webhook processing failed: {str(e)}")