@pytest.mark.integration
def test_checkout_flow():
    """Test complete checkout flow"""
    with mock.patch.dict(os.environ, {"STRIPE_SECRET_KEY": "sk_test_..."}):
        url = create_stripe_checkout(10.99, {"email": "test@example.com"})
        assert url.startswith("https://checkout.stripe.com")