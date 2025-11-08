"""
Example demonstrating Argentum cost alerts and export capabilities.

This example shows how to:
1. Set up cost alerts with Slack webhooks and email notifications
2. Export cost data to various formats (CSV, PDF, JSON)
3. Generate shareable dashboard URLs
"""

from argentum import CostAlerts, CostExporter, CostTracker

def main():
    """Demonstrate cost alerts and export functionality."""
    
    # Initialize cost tracker (in real usage, this would track actual costs)
    cost_tracker = CostTracker()
    
    print("🚨 Setting up Cost Alerts...")
    
    # Initialize cost alerts system
    alerts = CostAlerts(cost_tracker)
    
    # Add Slack webhook alert at 80% budget
    slack_rule = alerts.add_slack_webhook(
        webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
        threshold=0.8,
        channel="#ai-costs",
        username="Argentum Bot",
        icon_emoji=":money_with_wings:"
    )
    print(f"✅ Added Slack alert: {slack_rule}")
    
    # Add email alert at 100% budget
    email_rule = alerts.add_email(
        email="finance@company.com",
        threshold=1.0,
        subject="🚨 AI Budget Exceeded!"
    )
    print(f"✅ Added email alert: {email_rule}")
    
    # Add webhook for Discord at $500 absolute threshold  
    discord_rule = alerts.add_webhook(
        url="https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK",
        threshold=500,
        threshold_type="absolute",
        message="💰 AI costs hit $500 threshold!"
    )
    print(f"✅ Added Discord webhook: {discord_rule}")
    
    # Simulate cost checking (would be automatic in real usage)
    print("\n📊 Checking cost thresholds...")
    triggered_alerts = alerts.check_thresholds(
        current_cost=850,  # $850 spent
        budget=1000,       # $1000 budget
        agent_id="production_system"
    )
    
    if triggered_alerts:
        print(f"🔥 Triggered {len(triggered_alerts)} alerts!")
        for alert in triggered_alerts:
            print(f"   - {alert.rule_name}: {alert.message[:50]}...")
    else:
        print("✅ No alerts triggered")
    
    # Show alert rules
    print("\n📋 Current alert rules:")
    rules = alerts.list_rules()
    for name, config in rules.items():
        status = "enabled" if config["enabled"] else "disabled"
        print(f"   - {name}: {config['threshold']} ({config['threshold_type']}) - {status}")
    
    print("\n📊 Setting up Cost Export...")
    
    # Initialize cost exporter
    exporter = CostExporter(cost_tracker)
    
    # Export to CSV
    csv_file = exporter.export_csv("cost_report.csv")
    print(f"✅ Exported CSV: {csv_file}")
    
    # Export to JSON for API integration
    json_file = exporter.export_json("cost_data.json", pretty=True)
    print(f"✅ Exported JSON: {json_file}")
    
    # Generate PDF executive report
    try:
        pdf_file = exporter.export_pdf_report("executive_summary.pdf", report_type="executive")
        print(f"✅ Generated PDF report: {pdf_file}")
    except ImportError:
        print("⚠️  PDF export requires reportlab: pip install reportlab")
    
    # Export Excel with charts
    try:
        excel_file = exporter.export_excel("detailed_report.xlsx", include_charts=True)
        print(f"✅ Generated Excel report: {excel_file}")
    except ImportError:
        print("⚠️  Excel export requires pandas and openpyxl: pip install pandas openpyxl")
    
    # Generate shareable dashboard
    print("\n🌐 Creating shareable dashboard...")
    from argentum.cost_export import DashboardConfig
    
    dashboard_config = DashboardConfig(
        title="AI Cost Dashboard - November 2024",
        time_range_days=30,
        refresh_interval_minutes=15,
        expiry_hours=24
    )
    
    dashboard_url = exporter.generate_dashboard_url(dashboard_config)
    print(f"📱 Dashboard URL: {dashboard_url}")
    print("   (Share this with stakeholders for real-time cost visibility)")
    
    # Export format for Google Sheets
    print("\n📈 Google Sheets integration data:")
    sheets_data = exporter.export_google_sheets_format()
    print(f"   Prepared {len(sheets_data)} rows for Google Sheets API")
    print(f"   Headers: {sheets_data[0]}")
    print(f"   Sample row: {sheets_data[1] if len(sheets_data) > 1 else 'No data'}")
    
    # Demonstrate alert history
    print("\n📜 Alert History:")
    alert_history = alerts.get_alert_history(limit=10)
    if alert_history:
        for alert in alert_history[-3:]:  # Show last 3 alerts
            print(f"   - {alert.triggered_at.strftime('%Y-%m-%d %H:%M'%)}: {alert.rule_name} (${alert.current_cost:.2f})")
    else:
        print("   No alerts in history")

if __name__ == "__main__":
    main()