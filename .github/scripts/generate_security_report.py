#!/usr/bin/env python3
"""
Security report generation script for MFGArchon security pipeline.

Renders security-dashboard.html and security-summary.md from security-findings.json, which
`security_findings.py collect` builds from the five scan jobs' findings files.
"""

import json
from pathlib import Path


class SecurityReportGenerator:
    """Generate the dashboard and the markdown summary from security-findings.json."""

    def __init__(self, findings_path: str = "security-findings.json"):
        self.summary = json.loads(Path(findings_path).read_text(encoding="utf-8"))["summary"]

    def generate_html_dashboard(self) -> str:
        """Generate HTML security dashboard."""
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MFGArchon Security Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .status-badge {{ padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; }}
        .status-pass {{ background: #28a745; }}
        .status-low {{ background: #ffc107; color: black; }}
        .status-medium {{ background: #fd7e14; }}
        .status-high {{ background: #dc3545; }}
        .status-critical {{ background: #6f42c1; }}
        .status-unknown {{ background: #6c757d; }}
        .status-incomplete {{ background: #343a40; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .scan-results {{ margin: 20px 0; }}
        .scan-section {{ margin: 20px 0; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 MFGArchon Security Dashboard</h1>
            <span class="status-badge status-{self.summary["overall_status"].lower()}">{
            self.summary["overall_status"]
        }</span>
            <p>Generated: {self.summary["timestamp"]}</p>
        </div>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Issues</h3>
                <p style="font-size: 2em; margin: 0;">{self.summary["total_issues"]}</p>
            </div>
            <div class="summary-card">
                <h3>Critical Issues</h3>
                <p style="font-size: 2em; margin: 0; color: #6f42c1;">{self.summary["critical_issues"]}</p>
            </div>
            <div class="summary-card">
                <h3>High Issues</h3>
                <p style="font-size: 2em; margin: 0; color: #dc3545;">{self.summary["high_issues"]}</p>
            </div>
            <div class="summary-card">
                <h3>Medium Issues</h3>
                <p style="font-size: 2em; margin: 0; color: #fd7e14;">{self.summary["medium_issues"]}</p>
            </div>
            <div class="summary-card">
                <h3>Low Issues</h3>
                <p style="font-size: 2em; margin: 0;">{self.summary["low_issues"]}</p>
            </div>
            <div class="summary-card">
                <h3>No Severity Given</h3>
                <p style="font-size: 2em; margin: 0; color: #6c757d;">{self.summary["unknown_issues"]}</p>
            </div>
        </div>

        <div class="scan-results">
            <h2>Scan Results Summary</h2>
            <div class="scan-section">
                <h3>✅ Completed Scans</h3>
                <ul>
                    {"".join(f"<li>{scan}</li>" for scan in self.summary["scans_completed"])}
                </ul>
            </div>

            {
            f'''<div class="scan-section">
                <h3>❌ Failed Scans</h3>
                <ul>
                    {"".join(f"<li>{scan}</li>" for scan in self.summary["scans_failed"])}
                </ul>
            </div>'''
            if self.summary["scans_failed"]
            else ""
        }
        </div>
    </div>
</body>
</html>
        """
        return html_template

    def generate_markdown_summary(self) -> str:
        """Generate markdown security summary."""
        lines = []
        lines.append("# MFGArchon Security Summary")
        lines.append("")
        lines.append(f"**Generated**: {self.summary['timestamp']}")
        lines.append(f"**Overall Status**: {self.summary['overall_status']}")
        lines.append("")

        lines.append("## Summary")
        lines.append(f"- **Total Issues**: {self.summary['total_issues']}")
        lines.append(f"- **Critical**: {self.summary['critical_issues']}")
        lines.append(f"- **High**: {self.summary['high_issues']}")
        lines.append(f"- **Medium**: {self.summary['medium_issues']}")
        lines.append(f"- **Low**: {self.summary['low_issues']}")
        lines.append(f"- **No severity given by the tool**: {self.summary['unknown_issues']}")
        lines.append("")

        lines.append("## Scan Status")
        lines.append(f"- **Completed**: {', '.join(self.summary['scans_completed'])}")
        if self.summary["scans_failed"]:
            lines.append(f"- **Failed**: {', '.join(self.summary['scans_failed'])}")
        lines.append("")

        lines.append("## Tools")
        lines.append("| job | tool | status | findings | scanned | could not scan |")
        lines.append("|---|---|---|---|---|---|")
        for row in self.summary["tools"]:
            cells = [
                row["job"],
                row["tool"],
                row["status"],
                row.get("findings", ""),
                row.get("scanned", ""),
                row.get("errors", ""),
            ]
            lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
        lines.append("")

        return "\n".join(lines)

    def save_reports(self):
        """Save the dashboard and the markdown summary."""
        with open("security-dashboard.html", "w") as f:
            f.write(self.generate_html_dashboard())

        with open("security-summary.md", "w") as f:
            f.write(self.generate_markdown_summary())

        print("📊 Security reports generated:")
        print("  - security-dashboard.html")
        print("  - security-summary.md")

        print(f"\n🔒 Security Status: {self.summary['overall_status']}")
        print(f"📊 Total Issues: {self.summary['total_issues']}")

        if self.summary["critical_issues"] > 0 or self.summary["high_issues"] > 0:
            print(
                f"⚠️ Found {self.summary['critical_issues']} critical and {self.summary['high_issues']} high severity issues!"
            )


def main():
    """Main entry point."""
    print("📊 Generating comprehensive security report...")
    SecurityReportGenerator().save_reports()


if __name__ == "__main__":
    main()
