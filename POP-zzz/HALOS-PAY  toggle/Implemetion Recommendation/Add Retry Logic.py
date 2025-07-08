from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(stripe.error.APIConnectionError)
)
def create_stripe_checkout_with_retry(amount: float, user_info: Dict) -> str:
    """Wrapper with retry logic for transient errors"""
    return create_stripe_checkout(amount, user_info)