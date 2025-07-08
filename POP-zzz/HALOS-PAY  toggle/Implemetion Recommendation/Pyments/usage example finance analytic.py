# Initialize analytics
analytics = FinanceAnalytics()

# Get a quick summary
summary = analytics.get_summary(days=30)
print(f"Total revenue last 30 days: ${summary['total_revenue']:,.2f}")

# Get quarterly totals
quarterly = analytics.get_totals_by_period(TimePeriod.QUARTERLY)
for q in quarterly:
    print(f"{q['period_label']}: ${q['total_amount']:,.2f}")

# Generate a full monthly report
report = analytics.generate_report("monthly")
print(json.dumps(report, indent=2))

# Analyze payment methods
methods = analytics.get_payment_method_analysis()
for method in methods['methods']:
    print(f"{method['method']}: {method['percentage']:.1f}% of transactions")

# Get customer insights
customers = analytics.get_customer_metrics(top_n=5)
print("Top customers:")
for cust in customers['top_customers']:
    print(f"- {cust['user_email']}: ${cust['total_spent']:,.2f}")