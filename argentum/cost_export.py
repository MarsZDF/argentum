"""
Cost export and reporting for Argentum.

This module provides CSV/Excel export, PDF report generation, and shareable
dashboard capabilities for cost data analysis and executive reporting.

Examples:
    >>> from argentum import CostExporter
    >>> 
    >>> exporter = CostExporter(cost_tracker)
    >>> exporter.export_csv("costs_november.csv")
    >>> exporter.export_pdf_report("executive_summary.pdf")
    >>> dashboard_url = exporter.generate_dashboard_url()
"""

import base64
import csv
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Optional dependencies for enhanced reporting
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.widgetbase import Widget
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Try to import cost tracker
try:
    from .cost_optimization.cost_tracker import CostTracker

    COST_TRACKING_AVAILABLE = True
except ImportError:
    COST_TRACKING_AVAILABLE = False


class ExportSecurityError(Exception):
    """Security validation error for exports."""

    pass


def _validate_file_path(filepath: str) -> str:
    """
    Validate and sanitize file paths to prevent directory traversal.

    Args:
        filepath: File path to validate

    Returns:
        Sanitized file path

    Raises:
        ExportSecurityError: If path is potentially unsafe
    """
    # Resolve path and check for traversal attempts
    try:
        resolved_path = os.path.abspath(filepath)
        working_dir = os.getcwd()

        # Ensure the file is within or below the current working directory
        if not resolved_path.startswith(working_dir):
            raise ExportSecurityError("File path outside working directory not allowed")

    except (OSError, ValueError) as e:
        raise ExportSecurityError(f"Invalid file path: {e}")

    # Check for dangerous path components
    dangerous_patterns = ["..", "~", "$", "`", "|", ";", "&"]
    for pattern in dangerous_patterns:
        if pattern in filepath:
            raise ExportSecurityError(f"Dangerous path component detected: {pattern}")

    # Validate file extension
    allowed_extensions = [".csv", ".xlsx", ".xls", ".pdf", ".json", ".txt"]
    if not any(filepath.lower().endswith(ext) for ext in allowed_extensions):
        raise ExportSecurityError("File extension not allowed for export")

    # Limit filename length
    filename = os.path.basename(filepath)
    if len(filename) > 255:  # Most filesystems limit
        raise ExportSecurityError("Filename too long")

    return resolved_path


def _sanitize_dashboard_config(config: "DashboardConfig") -> "DashboardConfig":
    """
    Sanitize dashboard configuration for security.

    Args:
        config: Dashboard configuration to sanitize

    Returns:
        Sanitized configuration
    """
    # Sanitize title
    if hasattr(config, "title"):
        config.title = re.sub(r'[<>"\']', "", config.title)[:100]

    # Validate time ranges
    if hasattr(config, "time_range_days"):
        config.time_range_days = min(max(1, config.time_range_days), 365)  # 1-365 days

    if hasattr(config, "refresh_interval_minutes"):
        config.refresh_interval_minutes = min(
            max(1, config.refresh_interval_minutes), 1440
        )  # 1 min to 24 hours

    if hasattr(config, "expiry_hours"):
        if config.expiry_hours:
            config.expiry_hours = min(max(1, config.expiry_hours), 168)  # 1 hour to 1 week

    return config


@dataclass
class ExportConfig:
    """Configuration for cost exports."""

    include_timestamps: bool = True
    include_agent_breakdown: bool = True
    include_operation_breakdown: bool = True
    date_format: str = "%Y-%m-%d %H:%M:%S"
    currency_symbol: str = "$"
    group_by_day: bool = False


@dataclass
class DashboardConfig:
    """Configuration for shareable dashboards."""

    title: str = "Argentum Cost Dashboard"
    time_range_days: int = 30
    refresh_interval_minutes: int = 15
    include_predictions: bool = False
    public_access: bool = False
    expiry_hours: Optional[int] = 24


class CostExporter:
    """
    Export and reporting system for Argentum cost data.

    Supports CSV, Excel, PDF reports, and shareable dashboard generation
    with flexible formatting and business intelligence tool integration.
    """

    def __init__(self, cost_tracker: Optional[Any] = None):
        """
        Initialize cost exporter.

        Args:
            cost_tracker: Cost tracker instance to export data from
        """
        self._cost_tracker = cost_tracker
        self._dashboard_cache: Dict[str, Dict[str, Any]] = {}

    def export_csv(
        self, filepath: str, config: ExportConfig = None, date_range: Optional[tuple] = None
    ) -> str:
        """
        Export cost data to CSV format.

        Args:
            filepath: Output file path
            config: Export configuration
            date_range: Optional (start_date, end_date) tuple

        Returns:
            Path to created file

        Examples:
            >>> exporter.export_csv("costs.csv")
            >>> exporter.export_csv("costs.csv", date_range=(start, end))
        """
        # Security validation
        filepath = _validate_file_path(filepath)

        if config is None:
            config = ExportConfig()

        cost_data = self._get_cost_data(date_range)

        # Limit data export size
        if len(cost_data) > 50000:  # Limit to 50K records
            raise ExportSecurityError("Dataset too large for export (max 50,000 records)")

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            headers = ["timestamp", "agent_id", "operation", "cost", "tokens_used"]
            if config.include_agent_breakdown:
                headers.extend(["model", "efficiency_score"])
            writer.writerow(headers)

            # Write data rows
            for event in cost_data:
                row = [
                    event.get("timestamp", "").strftime(config.date_format)
                    if config.include_timestamps
                    else "",
                    event.get("agent_id", ""),
                    event.get("operation", ""),
                    f"{config.currency_symbol}{event.get('cost', 0):.4f}",
                    event.get("tokens_used", 0),
                ]

                if config.include_agent_breakdown:
                    row.extend([event.get("model", ""), event.get("efficiency_score", 0.0)])

                writer.writerow(row)

        return filepath

    def export_excel(
        self, filepath: str, config: ExportConfig = None, include_charts: bool = True
    ) -> str:
        """
        Export cost data to Excel format with charts and summaries.

        Args:
            filepath: Output file path (.xlsx)
            config: Export configuration
            include_charts: Include cost trend charts

        Returns:
            Path to created file
        """
        # Security validation
        filepath = _validate_file_path(filepath)

        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas required for Excel export. Install with: pip install pandas openpyxl"
            )

        if config is None:
            config = ExportConfig()

        cost_data = self._get_cost_data()

        # Convert to DataFrame
        df = pd.DataFrame(cost_data)

        # Create Excel writer
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name="Cost Data", index=False)

            # Summary sheet
            summary_data = self._generate_summary_data(cost_data)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Agent breakdown sheet
            if config.include_agent_breakdown:
                agent_breakdown = self._generate_agent_breakdown(cost_data)
                agent_df = pd.DataFrame(agent_breakdown)
                agent_df.to_excel(writer, sheet_name="By Agent", index=False)

        return filepath

    def export_pdf_report(
        self, filepath: str, config: ExportConfig = None, report_type: str = "executive"
    ) -> str:
        """
        Generate PDF executive report with charts and insights.

        Args:
            filepath: Output file path (.pdf)
            config: Export configuration
            report_type: "executive", "detailed", or "technical"

        Returns:
            Path to created file

        Examples:
            >>> exporter.export_pdf_report("exec_summary.pdf", report_type="executive")
            >>> exporter.export_pdf_report("technical.pdf", report_type="technical")
        """
        # Security validation
        filepath = _validate_file_path(filepath)

        if not PDF_AVAILABLE:
            raise ImportError(
                "reportlab required for PDF export. Install with: pip install reportlab"
            )

        if config is None:
            config = ExportConfig()

        cost_data = self._get_cost_data()

        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
        )
        story.append(Paragraph(f"Argentum Cost Report", title_style))
        story.append(Spacer(1, 12))

        # Executive summary
        if report_type == "executive":
            story.extend(self._generate_executive_summary(cost_data, styles))
        elif report_type == "technical":
            story.extend(self._generate_technical_report(cost_data, styles))
        else:
            story.extend(self._generate_detailed_report(cost_data, styles))

        # Build PDF
        doc.build(story)
        return filepath

    def generate_dashboard_url(self, config: DashboardConfig = None) -> str:
        """
        Generate shareable dashboard URL.

        Args:
            config: Dashboard configuration

        Returns:
            Shareable dashboard URL

        Examples:
            >>> url = exporter.generate_dashboard_url()
            >>> # Share with stakeholders: https://dash.argentum.ai/abc123
        """
        if config is None:
            config = DashboardConfig()

        # Security validation
        config = _sanitize_dashboard_config(config)

        # Generate unique dashboard ID
        dashboard_id = str(uuid.uuid4())[:12]

        # Prepare dashboard data
        cost_data = self._get_cost_data()
        dashboard_data = {
            "config": asdict(config),
            "data": cost_data,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=config.expiry_hours)
            if config.expiry_hours
            else None,
        }

        # Cache dashboard data
        self._dashboard_cache[dashboard_id] = dashboard_data

        # Generate URL (in production, this would be a real hosted dashboard)
        base_url = (
            "https://dash.argentum.ai"
            if config.public_access
            else "https://internal-dash.argentum.ai"
        )
        return f"{base_url}/{dashboard_id}"

    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve dashboard data by ID.

        Args:
            dashboard_id: Dashboard identifier

        Returns:
            Dashboard data or None if not found/expired
        """
        if dashboard_id not in self._dashboard_cache:
            return None

        dashboard_data = self._dashboard_cache[dashboard_id]

        # Check expiry
        if dashboard_data.get("expires_at"):
            if datetime.now() > dashboard_data["expires_at"]:
                del self._dashboard_cache[dashboard_id]
                return None

        return dashboard_data

    def export_json(
        self, filepath: str, date_range: Optional[tuple] = None, pretty: bool = True
    ) -> str:
        """
        Export cost data to JSON format for API integration.

        Args:
            filepath: Output file path
            date_range: Optional date range filter
            pretty: Pretty-print JSON output

        Returns:
            Path to created file
        """
        # Security validation
        filepath = _validate_file_path(filepath)

        cost_data = self._get_cost_data(date_range)

        # Limit data export size
        if len(cost_data) > 10000:  # Smaller limit for JSON
            raise ExportSecurityError("Dataset too large for JSON export (max 10,000 records)")

        # Convert datetime objects to ISO format
        serializable_data = []
        for event in cost_data:
            event_copy = event.copy()
            if "timestamp" in event_copy and hasattr(event_copy["timestamp"], "isoformat"):
                event_copy["timestamp"] = event_copy["timestamp"].isoformat()
            serializable_data.append(event_copy)

        with open(filepath, "w", encoding="utf-8") as jsonfile:
            if pretty:
                json.dump(serializable_data, jsonfile, indent=2, ensure_ascii=False)
            else:
                json.dump(serializable_data, jsonfile, ensure_ascii=False)

        return filepath

    def export_google_sheets_format(self) -> List[List[str]]:
        """
        Export data in format suitable for Google Sheets API.

        Returns:
            List of rows (each row is a list of cell values)
        """
        cost_data = self._get_cost_data()

        # Headers
        rows = [["Timestamp", "Agent ID", "Operation", "Cost", "Tokens", "Model", "Efficiency"]]

        # Data rows
        for event in cost_data:
            rows.append(
                [
                    event.get("timestamp", "").isoformat()
                    if hasattr(event.get("timestamp", ""), "isoformat")
                    else str(event.get("timestamp", "")),
                    str(event.get("agent_id", "")),
                    str(event.get("operation", "")),
                    f"${event.get('cost', 0):.4f}",
                    str(event.get("tokens_used", 0)),
                    str(event.get("model", "")),
                    f"{event.get('efficiency_score', 0):.3f}",
                ]
            )

        return rows

    def _get_cost_data(self, date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Get cost data from tracker with optional date filtering."""
        if not self._cost_tracker:
            # Return mock data for demonstration
            return self._generate_mock_data()

        # In a real implementation, this would fetch from the cost tracker
        # For now, return mock data
        return self._generate_mock_data()

    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock cost data for demonstration."""
        now = datetime.now()
        mock_data = []

        agents = ["researcher", "writer", "analyzer", "validator"]
        operations = ["search", "generate", "analyze", "validate"]
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"]

        for i in range(100):
            timestamp = now - timedelta(hours=i // 4)
            mock_data.append(
                {
                    "timestamp": timestamp,
                    "agent_id": agents[i % len(agents)],
                    "operation": operations[i % len(operations)],
                    "cost": round((i + 1) * 0.002, 4),
                    "tokens_used": (i + 1) * 150,
                    "model": models[i % len(models)],
                    "efficiency_score": min(1.0, 0.3 + (i % 10) * 0.07),
                }
            )

        return mock_data

    def _generate_summary_data(self, cost_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate summary statistics from cost data."""
        total_cost = sum(event.get("cost", 0) for event in cost_data)
        total_tokens = sum(event.get("tokens_used", 0) for event in cost_data)

        return [
            {"Metric": "Total Cost", "Value": f"${total_cost:.2f}"},
            {"Metric": "Total Tokens", "Value": f"{total_tokens:,}"},
            {
                "Metric": "Average Cost per Token",
                "Value": f"${total_cost/max(total_tokens, 1):.6f}",
            },
            {"Metric": "Number of Operations", "Value": str(len(cost_data))},
            {
                "Metric": "Most Expensive Operation",
                "Value": f"${max((e.get('cost', 0) for e in cost_data), default=0):.4f}",
            },
        ]

    def _generate_agent_breakdown(self, cost_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate per-agent cost breakdown."""
        agent_costs = {}
        for event in cost_data:
            agent_id = event.get("agent_id", "unknown")
            if agent_id not in agent_costs:
                agent_costs[agent_id] = {"cost": 0, "tokens": 0, "operations": 0}

            agent_costs[agent_id]["cost"] += event.get("cost", 0)
            agent_costs[agent_id]["tokens"] += event.get("tokens_used", 0)
            agent_costs[agent_id]["operations"] += 1

        breakdown = []
        for agent_id, data in agent_costs.items():
            breakdown.append(
                {
                    "Agent": agent_id,
                    "Total Cost": f"${data['cost']:.2f}",
                    "Total Tokens": f"{data['tokens']:,}",
                    "Operations": data["operations"],
                    "Avg Cost per Operation": f"${data['cost']/max(data['operations'], 1):.4f}",
                }
            )

        return sorted(breakdown, key=lambda x: float(x["Total Cost"][1:]), reverse=True)

    def _generate_executive_summary(self, cost_data: List[Dict[str, Any]], styles) -> List:
        """Generate executive summary content for PDF."""
        story = []

        # Summary paragraph
        total_cost = sum(event.get("cost", 0) for event in cost_data)
        story.append(Paragraph(f"<b>Executive Summary</b>", styles["Heading2"]))
        story.append(
            Paragraph(
                f"This report covers AI agent costs totaling <b>${total_cost:.2f}</b> across "
                f"{len(cost_data)} operations. Key insights and cost optimization opportunities are highlighted below.",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

        # Key metrics table
        summary_data = self._generate_summary_data(cost_data)
        table_data = [["Metric", "Value"]]
        table_data.extend([[item["Metric"], item["Value"]] for item in summary_data])

        table = Table(table_data)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        story.append(table)
        return story

    def _generate_technical_report(self, cost_data: List[Dict[str, Any]], styles) -> List:
        """Generate technical report content for PDF."""
        story = []

        story.append(Paragraph("<b>Technical Cost Analysis</b>", styles["Heading2"]))
        story.append(
            Paragraph(
                "Detailed breakdown of cost drivers, efficiency metrics, and optimization opportunities.",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

        # Agent breakdown
        agent_breakdown = self._generate_agent_breakdown(cost_data)
        if agent_breakdown:
            story.append(Paragraph("<b>Cost by Agent</b>", styles["Heading3"]))

            table_data = [["Agent", "Total Cost", "Operations", "Avg Cost/Op"]]
            for agent_data in agent_breakdown[:10]:  # Top 10
                table_data.append(
                    [
                        agent_data["Agent"],
                        agent_data["Total Cost"],
                        str(agent_data["Operations"]),
                        agent_data["Avg Cost per Operation"],
                    ]
                )

            table = Table(table_data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(table)

        return story

    def _generate_detailed_report(self, cost_data: List[Dict[str, Any]], styles) -> List:
        """Generate detailed report content for PDF."""
        story = []

        story.append(Paragraph("<b>Detailed Cost Report</b>", styles["Heading2"]))

        # Include both summary and technical sections
        story.extend(self._generate_executive_summary(cost_data, styles))
        story.append(Spacer(1, 24))
        story.extend(self._generate_technical_report(cost_data, styles))

        return story
