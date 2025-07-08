import logging
import json
from datetime import datetime
from typing import Dict, Any

# Configure structured logging
class StructuredFormatter(logging.Formatter):
    """Log formatter that outputs JSON"""
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "context": getattr(record, "context", {}),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# Logger configuration
def configure_logging():
    logger = logging.getLogger("payment_processor")
    logger.setLevel(logging.INFO)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    logger.addHandler(console_handler)
    
    # File/Syslog handler for production
    if os.getenv("ENV") == "production":
        file_handler = logging.FileHandler("/var/log/payments.log")
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)
    
    return logger

logger = configure_logging()

# Example usage in payment functions
def process_payment_webhook(payload: Dict[str, Any]):
    """Example webhook handler with detailed logging"""
    try:
        event = verify_webhook_signature(payload)
        
        logger.info(
            "Webhook received",
            extra={
                "event": "webhook_received",
                "type": event["type"],
                "webhook_id": event["id"]
            }
        )
        
        if event["type"] == "payment_intent.succeeded":
            payment = event["data"]["object"]
            logger.info(
                "Payment succeeded",
                extra={
                    "event": "payment_success",
                    "payment_id": payment["id"],
                    "amount": payment["amount"],
                    "currency": payment["currency"],
                    "customer": payment.get("customer")
                }
            )
            # Process payment...
            
    except Exception as e:
        logger.error(
            "Webhook processing failed",
            exc_info=True,
            extra={
                "event": "webhook_failure",
                "payload": sanitize_payload(payload)
            }
        )
        raise

def sanitize_payload(payload: Dict) -> Dict:
    """Remove sensitive data before logging"""
    sanitized = payload.copy()
    for sensitive_field in ["card", "cvc", "number", "exp_month", "exp_year"]:
        if sensitive_field in sanitized:
            sanitized[sensitive_field] = "***REDACTED***"
    return sanitized