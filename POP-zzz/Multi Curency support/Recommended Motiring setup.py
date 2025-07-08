# Add to your logging configuration
if os.getenv("SENTRY_DSN"):
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    
    sentry_logging = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR
    )
    
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        integrations=[sentry_logging],
        traces_sample_rate=1.0,
        environment=os.getenv("ENV", "development")
    )