# Updated Stripe configuration
class StripeConfig:
    """Enhanced configuration with multi-currency support"""
    PRODUCT_NAME = os.getenv("PRODUCT_NAME", "HALOS Pro License")
    DEFAULT_CURRENCY = os.getenv("DEFAULT_CURRENCY", "usd")
    SUPPORTED_CURRENCIES = {
        "usd": {"symbol": "$", "minimum_amount": 0.50},
        "eur": {"symbol": "€", "minimum_amount": 0.45},
        "gbp": {"symbol": "£", "minimum_amount": 0.40},
        "inr": {"symbol": "₹", "minimum_amount": 40},
        "jpy": {"symbol": "¥", "minimum_amount": 50}  # JPY has no minor unit
    }

    @classmethod
    def validate_currency(cls, currency: str, amount: float) -> tuple:
        """Validate currency and amount against business rules"""
        currency = currency.lower()
        if currency not in cls.SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {currency}")
        
        min_amount = cls.SUPPORTED_CURRENCIES[currency]["minimum_amount"]
        if amount < min_amount:
            raise ValueError(
                f"Amount {amount} is below minimum {min_amount} for {currency}"
            )
        
        # Handle zero-decimal currencies
        is_zero_decimal = currency in ["jpy"]
        amount_in_units = int(amount * 100) if not is_zero_decimal else int(amount)
        
        return currency, amount_in_units

# Updated checkout function
def create_stripe_checkout(
    amount: float,
    user_info: Dict,
    currency: str = None,
    metadata: Dict = None
) -> str:
    """
    Create a checkout session with multi-currency support
    
    Args:
        amount: Payment amount
        user_info: Customer information
        currency: 3-letter ISO currency code
        metadata: Additional payment metadata
        
    Returns:
        Checkout session URL
    """
    try:
        # Validate and normalize currency
        currency = currency or StripeConfig.DEFAULT_CURRENCY
        validated_currency, amount_units = StripeConfig.validate_currency(
            currency, amount
        )
        
        # Create line items with currency-specific formatting
        line_item = {
            "price_data": {
                "currency": validated_currency,
                "product_data": {"name": StripeConfig.PRODUCT_NAME},
                "unit_amount": amount_units,
            },
            "quantity": 1,
        }
        
        # Merge additional metadata
        full_metadata = {
            "user_id": user_info.get("user_id"),
            "user_name": user_info.get("name"),
            "currency": validated_currency,
            **(metadata or {})
        }
        
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[line_item],
            mode="payment",
            customer_email=user_info["email"],
            success_url=StripeConfig.get_urls()["success"],
            cancel_url=StripeConfig.get_urls()["cancel"],
            metadata=full_metadata,
            automatic_tax={
                "enabled": True  # Enable Stripe's automatic tax calculation
            }
        )
        
        logger.info(
            "Checkout session created",
            extra={
                "event": "checkout_created",
                "amount": amount,
                "currency": validated_currency,
                "session_id": session.id,
                "user_email": user_info["email"]
            }
        )
        
        return session.url

    except ValueError as e:
        logger.warning(
            "Currency validation failed",
            extra={"error": str(e), "currency": currency, "amount": amount}
        )
        raise
    except stripe.error.StripeError as e:
        logger.error(
            "Stripe API error",
            exc_info=True,
            extra={
                "error_type": e.__class__.__name__,
                "http_status": e.http_status,
                "code": e.code
            }
        )
        raise