#!/usr/bin/env python3
"""
Custom secrets scanning script for MFGArchon security pipeline.
Scans for hardcoded secrets, API keys, and sensitive information.

It reads every file git tracks that holds no NUL byte in its first 8000 bytes, whatever its name or suffix,
as UTF-8 with any byte that does not decode dropped, as the scan always did, and its reports name the file,
line and kind of each finding, never the matched text.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import ClassVar


class CustomSecretsScanner:
    """Custom scanner for secrets and sensitive information."""

    # Patterns for different types of secrets
    SECRET_PATTERNS: ClassVar[dict[str, list[str]]] = {
        "api_key": [
            r"(?i)api[_-]?key['\"\s]*[:=]['\"\s]*[a-zA-Z0-9]{20,}",
            r"(?i)apikey['\"\s]*[:=]['\"\s]*[a-zA-Z0-9]{20,}",
        ],
        "secret_key": [
            r"(?i)secret[_-]?key['\"\s]*[:=]['\"\s]*[a-zA-Z0-9]{20,}",
            r"(?i)secretkey['\"\s]*[:=]['\"\s]*[a-zA-Z0-9]{20,}",
        ],
        "password": [
            r"(?i)password['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
            r"(?i)passwd['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
            r"(?i)pwd['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
        ],
        "token": [
            r"(?i)token['\"\s]*[:=]['\"\s]*[a-zA-Z0-9._\-]{20,}",
            r"(?i)access[_-]?token['\"\s]*[:=]['\"\s]*[a-zA-Z0-9._\-]{20,}",
            r"(?i)auth[_-]?token['\"\s]*[:=]['\"\s]*[a-zA-Z0-9._\-]{20,}",
        ],
        "private_key": [
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        ],
        "database_url": [
            r"(?i)(?:postgres|mysql|mongodb)://[^\s\"']+",
            r"(?i)database[_-]?url['\"\s]*[:=]['\"\s]*[a-zA-Z0-9+.:/\-_@]+",
        ],
        "aws_key": [
            r"AKIA[0-9A-Z]{16}",
            r"(?i)aws[_-]?access[_-]?key[_-]?id['\"\s]*[:=]['\"\s]*[A-Z0-9]{20}",
            r"(?i)aws[_-]?secret[_-]?access[_-]?key['\"\s]*[:=]['\"\s]*[A-Za-z0-9/+=]{40}",
        ],
        "github_token": [
            r"ghp_[a-zA-Z0-9]{36}",
            r"gho_[a-zA-Z0-9]{36}",
            r"ghu_[a-zA-Z0-9]{36}",
            r"ghs_[a-zA-Z0-9]{36}",
            r"ghr_[a-zA-Z0-9]{36}",
        ],
        "jwt_token": [
            r"eyJ[a-zA-Z0-9_\-]*\.eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*",
        ],
        "email_credentials": [
            r"(?i)smtp[_-]?password['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
            r"(?i)email[_-]?password['\"\s]*[:=]['\"\s]*[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]{8,}",
        ],
        # A key, token or secret name, then = or :, then a quoted run of 32 or more alphanumerics. Either quote
        # may be backslash-escaped, as it is inside a JSON string. The whitelist reads the value alone.
        "credential_assignment": [
            r"""(?i)(?:key|token|secret)[\w.-]{0,64}\\?["']?\s*[:=]\s*\\?["'](?P<value>[A-Za-z0-9]{32,})""",
        ],
    }

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.findings = []
        self.false_positives = set()
        self.load_whitelist()

    def load_whitelist(self):
        """Load whitelist of known false positives."""
        # Common false positives in scientific computing
        self.false_positives = {
            # Mathematical constants and examples
            "example_key",
            "demo_password",
            "test_token",
            "your_api_key",
            "insert_key_here",
            "replace_with_key",
            "dummy_password",
            "fake_secret",
            "placeholder",
            # Documentation examples
            "sk-test",
            "pk_test",
            "demo_",
            "example_",
            # Common variable names
            "api_key_var",
            "secret_key_var",
            "token_var",
            # An expression assigned to a name ending in "token" (mfgarchon/utils/progress.py)
            "_progress_context",
        }

    def is_whitelisted(self, match: str) -> bool:
        """Check if a match is a known false positive."""
        match_lower = match.lower()

        # Check explicit whitelist
        for fp in self.false_positives:
            if fp in match_lower:
                return True

        # Check if it's a template or example
        return bool(any(marker in match_lower for marker in ["example", "demo", "test", "placeholder", "dummy"]))

    def scan_file(self, file_path: Path, content: str) -> list[dict]:
        """Scan one file's text for secrets."""
        findings = []

        for secret_type, patterns in self.SECRET_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    matched_text = match.group(0)

                    # Skip if whitelisted
                    if self.is_whitelisted(match.groupdict().get("value") or matched_text):
                        continue

                    # Find line number
                    line_num = content[: match.start()].count("\n") + 1

                    # The reports are uploaded, so a finding records where it is and what kind, not the text.
                    finding = {
                        "file": str(file_path.relative_to(self.root_dir)),
                        "line": line_num,
                        "type": secret_type,
                        "pattern": pattern,
                        "match": "<not recorded>",
                        "severity": self.get_severity(secret_type),
                    }

                    findings.append(finding)

        return findings

    def get_severity(self, secret_type: str) -> str:
        """Get severity level for secret type."""
        high_severity = {"private_key", "aws_key", "database_url", "github_token", "credential_assignment"}
        medium_severity = {"api_key", "secret_key", "token", "jwt_token"}

        if secret_type in high_severity:
            return "HIGH"
        elif secret_type in medium_severity:
            return "MEDIUM"
        else:
            return "LOW"

    def scan_directory(self) -> list[dict]:
        """Scan every file git tracks under root_dir that holds no NUL byte in its first 8000 bytes."""
        print(f"🔍 Scanning {self.root_dir} for secrets...")

        listed = subprocess.run(
            ["git", "-C", str(self.root_dir), "ls-files", "-z"], capture_output=True, check=True
        ).stdout
        names = [os.fsdecode(name) for name in listed.split(b"\0") if name]
        scanned, binary, not_regular = 0, 0, 0
        for name in names:
            file_path = self.root_dir / name
            if not file_path.is_file():
                not_regular += 1
                continue
            raw = file_path.read_bytes()
            if b"\0" in raw[:8000]:
                binary += 1
                continue
            scanned += 1
            self.findings.extend(self.scan_file(file_path, raw.decode("utf-8", errors="ignore")))

        print(
            f"Tracked files: {len(names)}; scanned as text: {scanned}; "
            f"binary (a NUL in the first 8000 bytes): {binary}; not a regular file: {not_regular}"
        )
        return self.findings

    def generate_report(self) -> dict:
        """Generate comprehensive secrets scan report."""
        # Group findings by severity
        by_severity = {"HIGH": [], "MEDIUM": [], "LOW": []}
        by_type = {}

        for finding in self.findings:
            severity = finding["severity"]
            secret_type = finding["type"]

            by_severity[severity].append(finding)

            if secret_type not in by_type:
                by_type[secret_type] = []
            by_type[secret_type].append(finding)

        report = {
            "summary": {
                "total_findings": len(self.findings),
                "high_severity": len(by_severity["HIGH"]),
                "medium_severity": len(by_severity["MEDIUM"]),
                "low_severity": len(by_severity["LOW"]),
                "files_scanned": len({f["file"] for f in self.findings}),
                "secret_types_found": list(by_type.keys()),
            },
            "findings_by_severity": by_severity,
            "findings_by_type": by_type,
            "all_findings": self.findings,
        }

        return report

    def save_results(self):
        """Save scan results to files."""
        report = self.generate_report()

        # Save JSON report
        with open("custom-secrets-report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Generate markdown summary
        markdown_report = self.generate_markdown_report(report)
        with open("secrets-summary.md", "w") as f:
            f.write(markdown_report)

        print("Custom secrets scan completed.")
        print(f"Total findings: {report['summary']['total_findings']}")

        if report["summary"]["high_severity"] > 0:
            print(f"❌ Found {report['summary']['high_severity']} HIGH severity secrets!")
            sys.exit(1)
        elif report["summary"]["medium_severity"] > 0:
            print(f"⚠️ Found {report['summary']['medium_severity']} MEDIUM severity secrets.")

        print("✅ No critical secrets found.")

    def generate_markdown_report(self, report: dict) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# Custom Secrets Scan Report")
        lines.append("")

        summary = report["summary"]
        lines.append("## Summary")
        lines.append(f"- **Total Findings**: {summary['total_findings']}")
        lines.append(f"- **High Severity**: {summary['high_severity']}")
        lines.append(f"- **Medium Severity**: {summary['medium_severity']}")
        lines.append(f"- **Low Severity**: {summary['low_severity']}")
        lines.append(f"- **Files Scanned**: {summary['files_scanned']}")
        lines.append("")

        if summary["high_severity"] > 0:
            lines.append("## ❌ High Severity Findings")
            for finding in report["findings_by_severity"]["HIGH"]:
                lines.append(f"- **{finding['file']}:{finding['line']}** - {finding['type']}")
                lines.append(f"  - Match: `{finding['match']}`")
            lines.append("")

        if summary["medium_severity"] > 0:
            lines.append("## ⚠️ Medium Severity Findings")
            for finding in report["findings_by_severity"]["MEDIUM"]:
                lines.append(f"- **{finding['file']}:{finding['line']}** - {finding['type']}")
                lines.append(f"  - Match: `{finding['match']}`")
            lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point."""
    scanner = CustomSecretsScanner()
    scanner.scan_directory()
    scanner.save_results()


if __name__ == "__main__":
    main()
