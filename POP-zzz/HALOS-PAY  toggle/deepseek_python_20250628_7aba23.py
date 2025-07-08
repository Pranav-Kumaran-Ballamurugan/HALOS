from fastapi import HTTPException  # or your framework's equivalent

def verify_webhook_signature(payload: bytes, sig_header: str) -> dict:
    """Verify Stripe webhook signature"""
    try:
        return stripe.Webhook.construct_event(
            payload,
            sig_header,
            os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=401, detail="Invalid signature")