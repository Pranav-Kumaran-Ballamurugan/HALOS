# Initialize the manager
tm = TransactionManager()

# Add a new transaction
txn = {
    "id": "txn_12345",
    "date": datetime.now().isoformat(),
    "amount": 49.99,
    "currency": "USD",
    "method": "stripe",
    "status": "pending",
    "user_email": "user@example.com",
    "metadata": {"product": "HALOS Pro", "invoice_id": "INV-1001"}
}

success, message = tm.add_transaction(txn)
if not success:
    print(f"Error: {message}")

# Update status
success, message = tm.update_transaction_status("txn_12345", "completed")
if not success:
    print(f"Error: {message}")

# Get recent transactions
recent = tm.get_recent_transactions(limit=10)
for t in recent:
    print(f"{t.id}: {t.amount} {t.currency} ({t.status})")

# Search transactions
results = tm.search_transactions(
    amount_min=10,
    date_start="2023-01-01",
    status="completed"
)

# Get audit log
audit_log = tm.get_transaction_audit_log("txn_12345")
for entry in audit_log:
    print(f"{entry['changed_at']}: {entry['changed_field']} changed from {entry['old_value']} to {entry['new_value']}")