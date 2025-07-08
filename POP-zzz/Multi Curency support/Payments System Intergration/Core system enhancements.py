# Add to Core System Components section (before HALOSCore class)

class PaymentMethod(Enum):
    """Payment method options"""
    STRIPE = auto()
    UPI = auto()
    PAYPAL = auto()
    BANK_TRANSFER = auto()

class PaymentStatus(Enum):
    """Payment status tracking"""
    PENDING = auto()
    COMPLETED = auto()
    FAILED = auto()
    REFUNDED = auto()

@dataclass
class Transaction:
    """Enhanced transaction data structure"""
    amount: float
    currency: str
    method: PaymentMethod
    status: PaymentStatus
    timestamp: datetime
    description: str
    metadata: Optional[Dict] = None
    transaction_id: Optional[str] = None