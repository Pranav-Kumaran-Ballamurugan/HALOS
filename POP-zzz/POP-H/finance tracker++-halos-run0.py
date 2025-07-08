class FinanceTrackerPlus:
    def __init__(self):
        self.plaid_client = PlaidClient()
        self.receipt_processor = ReceiptProcessor()
        self.budget_predictor = BudgetPredictor()

    def sync_accounts(self):
        accounts = self.plaid_client.get_accounts()
        transactions = self.plaid_client.get_transactions()
        return {
            "accounts": accounts,
            "transactions": transactions,
            "budget_analysis": self.budget_predictor.analyze(transactions)
        }

    def process_receipt(self, image_path: str):
        extracted = self.receipt_processor.scan(image_path)
        return {
            "items": extracted['items'],
            "total": extracted['total'],
            "category_suggestion": self._categorize(extracted)
        }