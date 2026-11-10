#!/usr/bin/env python3
"""
Critical security issues checker for MFGArchon CI/CD pipeline.

This script reads security-findings.json, which `security_findings.py collect` builds from the five
scan jobs' findings files, and determines if there are any critical issues that should fail the CI/CD
pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Any


class CriticalIssuesChecker:
    """Check for critical security issues across all scan results."""

    def __init__(self, findings_path: str = "security-findings.json"):
        self.jobs = json.loads(Path(findings_path).read_text(encoding="utf-8"))["jobs"]
        self.critical_issues = []
        self.blocking_issues = []
        self.warnings = []

        # Define what constitutes critical/blocking issues
        # Note: Relaxed thresholds for research/educational projects
        self.critical_criteria = {
            "dependency": {
                "critical_vulns": 2,  # Allow some critical vulns for research dependencies
                "high_vulns": 10,  # Max 10 high severity vulnerabilities
            },
            "static_analysis": {
                "high_confidence_high_severity": 2,  # Allow some high confidence + high severity
                "total_high_confidence": 15,  # Max 15 high confidence issues
            },
            "secrets": {
                "total_secrets": 5,  # Max 5 potential secrets (scientific code may have examples)
            },
            "container": {
                "critical_vulns": 10,  # Allow more container vulns for research (base image limitations)
                "high_vulns": 20,  # Max 20 high severity container vulns (research tolerance)
            },
            "license": {
                "non_compliant": 3,  # Allow some non-compliant licenses for research tools
            },
        }

    def check_all_results(self) -> dict[str, Any]:
        """Check all security scan results for critical issues."""
        print("🔍 Checking for critical security issues...")

        # Check each type of scan result
        self._check_dependency_results()
        self._check_static_analysis_results()
        self._check_secrets_results()
        self._check_container_results()
        self._check_license_results()
        self._check_custom_security_results()

        # Generate summary
        summary = {
            "has_critical_issues": len(self.critical_issues) > 0,
            "has_blocking_issues": len(self.blocking_issues) > 0,
            "critical_issues_count": len(self.critical_issues),
            "blocking_issues_count": len(self.blocking_issues),
            "warnings_count": len(self.warnings),
            "critical_issues": self.critical_issues,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
            "recommendation": self._get_recommendation(),
        }

        # Save results
        with open("critical-issues-report.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _findings(self, job: str, label: str, tools: tuple[str, ...]) -> list[dict[str, Any]] | None:
        """The job's findings from `tools`.

        A job whose findings file was not read gives None; one with a tool whose report was not read gives what was
        read. Either is a blocking issue: a scan that did not run is not a scan that found nothing.
        """
        data = self.jobs[job]
        if data["status"] in ("missing", "unreadable"):
            self.blocking_issues.append({"type": job, "severity": "high", "message": f"{label} results not found"})
            return None
        unread = sorted(tool for tool in tools if data["tools"].get(tool, {}).get("status") != "read")
        if unread:
            self.blocking_issues.append(
                {"type": job, "severity": "high", "message": f"{label} results incomplete: {', '.join(unread)}"}
            )
        return [f for f in data["findings"] if f["tool"] in tools]

    def _check_dependency_results(self):
        """Check dependency scan results for critical issues."""
        findings = self._findings("dependency", "Dependency scan", ("safety", "pip-audit"))
        if findings is None:
            return

        critical = [f for f in findings if f["severity"] == "critical"]
        high = [f for f in findings if f["severity"] == "high"]

        if len(critical) > self.critical_criteria["dependency"]["critical_vulns"]:
            self.critical_issues.append(
                {
                    "type": "dependency",
                    "severity": "critical",
                    "message": f"Found {len(critical)} critical dependency vulnerabilities",
                    "details": critical[:3],  # Top 3
                }
            )

        if len(high) > self.critical_criteria["dependency"]["high_vulns"]:
            self.blocking_issues.append(
                {
                    "type": "dependency",
                    "severity": "high",
                    "message": f"Found {len(high)} high severity vulnerabilities (max allowed: {self.critical_criteria['dependency']['high_vulns']})",
                    "details": high[:5],  # Top 5
                }
            )

        # Check for specific high-risk packages
        high_risk_packages = ["pickle", "yaml", "requests"]
        for finding in findings:
            package = finding.get("package", "").lower()
            if any(risk_pkg in package for risk_pkg in high_risk_packages):
                if finding["severity"] in ["critical", "high"]:
                    self.critical_issues.append(
                        {
                            "type": "dependency",
                            "severity": "critical",
                            "message": f"High-risk package {package} has {finding['severity']} vulnerability",
                            "details": [finding],
                        }
                    )

    def _check_static_analysis_results(self):
        """Check static analysis results for critical issues."""
        findings = self._findings("static", "Static analysis", ("bandit", "semgrep"))
        if findings is None:
            return

        high_confidence_issues = [f for f in findings if f.get("confidence") == "high"]
        high_conf_high_sev = [f for f in high_confidence_issues if f["severity"] in ["high", "critical"]]

        if len(high_conf_high_sev) > self.critical_criteria["static_analysis"]["high_confidence_high_severity"]:
            self.critical_issues.append(
                {
                    "type": "static_analysis",
                    "severity": "critical",
                    "message": f"Found {len(high_conf_high_sev)} high confidence + high severity static analysis issues",
                    "details": high_conf_high_sev,
                }
            )

        if len(high_confidence_issues) > self.critical_criteria["static_analysis"]["total_high_confidence"]:
            self.blocking_issues.append(
                {
                    "type": "static_analysis",
                    "severity": "high",
                    "message": f"Found {len(high_confidence_issues)} high confidence issues (max allowed: {self.critical_criteria['static_analysis']['total_high_confidence']})",
                    "details": high_confidence_issues[:5],  # Top 5
                }
            )

        # Check for specific dangerous patterns
        dangerous_patterns = ["sql_injection", "command_injection", "hardcoded_password", "weak_crypto"]
        for finding in findings:
            rule = finding.get("rule", "").lower()
            if any(pattern in rule for pattern in dangerous_patterns):
                self.critical_issues.append(
                    {
                        "type": "static_analysis",
                        "severity": "critical",
                        "message": f"Dangerous security pattern detected: {rule}",
                        "details": [finding],
                    }
                )

    def _check_secrets_results(self):
        """Check secrets scan results for critical issues."""
        findings = self._findings("secrets", "Secrets scan", ("detect-secrets", "trufflehog"))
        if findings is None:
            return

        # Check total secrets count
        if len(findings) > self.critical_criteria["secrets"]["total_secrets"]:
            self.blocking_issues.append(
                {
                    "type": "secrets",
                    "severity": "high",
                    "message": f"Found {len(findings)} potential secrets (max allowed: {self.critical_criteria['secrets']['total_secrets']})",
                    "details": findings[:3],  # Top 3
                }
            )

        # Check for specific secret types
        high_risk_secret_types = ["private_key", "api_key", "password", "token"]
        for finding in findings:
            rule = finding.get("rule", "").lower()
            if any(risk_type in rule for risk_type in high_risk_secret_types):
                self.critical_issues.append(
                    {
                        "type": "secrets",
                        "severity": "critical",
                        "message": f"High-risk secret type detected: {rule}",
                        "details": [finding],
                    }
                )

    def _check_container_results(self):
        """Check container scan results for critical issues."""
        findings = self._findings("container", "Container scan", ("trivy", "hadolint"))
        if findings is None:
            return

        vulnerabilities = [f for f in findings if f["tool"] == "trivy" and "advisory" in f]
        critical = [f for f in vulnerabilities if f["severity"] == "critical"]
        high = [f for f in vulnerabilities if f["severity"] == "high"]

        if len(critical) > self.critical_criteria["container"]["critical_vulns"]:
            self.critical_issues.append(
                {
                    "type": "container",
                    "severity": "critical",
                    "message": f"Found {len(critical)} critical container vulnerabilities",
                    "details": critical[:3],
                }
            )

        if len(high) > self.critical_criteria["container"]["high_vulns"]:
            self.blocking_issues.append(
                {
                    "type": "container",
                    "severity": "high",
                    "message": f"Found {len(high)} high severity container vulnerabilities (max allowed: {self.critical_criteria['container']['high_vulns']})",
                    "details": high[:3],
                }
            )

        # Check for base image issues, over trivy's vulnerabilities and hadolint, as the deleted parser read them
        dockerfile_issues = [f for f in findings if f["tool"] == "hadolint"]
        base_image_issues = [f for f in vulnerabilities + dockerfile_issues if "base" in f.get("file", "").lower()]
        if len(base_image_issues) > 5:
            self.blocking_issues.append(
                {
                    "type": "container",
                    "severity": "high",
                    "message": f"Base image has {len(base_image_issues)} security issues",
                    "details": base_image_issues[:3],
                }
            )

    def _check_license_results(self):
        """Check license compliance results for critical issues."""
        findings = self._findings("license", "License compliance", ("license_compliance_check",))
        if findings is None:
            return

        # Check for non-compliant licenses
        non_compliant_packages = [f for f in findings if f["status"] in ["review_required", "prohibited"]]
        if len(non_compliant_packages) > self.critical_criteria["license"]["non_compliant"]:
            # Check if these are production dependencies (not dev/test dependencies)
            production_non_compliant = []
            for package in non_compliant_packages:
                pkg_name = package.get("package", "").lower()
                # Skip development/testing packages
                if not any(dev_keyword in pkg_name for dev_keyword in ["test", "dev", "mock", "debug"]):
                    production_non_compliant.append(package)

            if production_non_compliant:
                self.critical_issues.append(
                    {
                        "type": "license",
                        "severity": "critical",
                        "message": f"Found {len(production_non_compliant)} non-compliant licenses in production dependencies",
                        "details": production_non_compliant,
                    }
                )

        # Check for unknown licenses in critical packages
        unknown_licenses = [f for f in findings if f["status"] == "unknown"]
        critical_packages = ["numpy", "scipy", "matplotlib"]  # Core scientific packages

        for package in unknown_licenses:
            pkg_name = package.get("package", "").lower()
            if any(critical_pkg in pkg_name for critical_pkg in critical_packages):
                self.warnings.append(f"Unknown license for critical package: {pkg_name}")

    def _check_custom_security_results(self):
        """Check custom security scan results for critical issues."""
        findings = self._findings("static", "Custom security", ("custom_security_checks",))
        if findings is None:
            return

        # Check for high severity custom issues
        critical_findings = [f for f in findings if f["severity"] == "critical"]
        high_findings = [f for f in findings if f["severity"] == "high"]

        if critical_findings:
            self.critical_issues.append(
                {
                    "type": "custom_security",
                    "severity": "critical",
                    "message": f"Found {len(critical_findings)} critical custom security issues",
                    "details": critical_findings,
                }
            )

        if len(high_findings) > 5:  # Increased threshold for research projects
            self.blocking_issues.append(
                {
                    "type": "custom_security",
                    "severity": "high",
                    "message": f"Found {len(high_findings)} high severity custom security issues",
                    "details": high_findings[:3],
                }
            )

        # Check for specific dangerous patterns
        dangerous_types = ["subprocess_security", "pickle_security", "plugin_security"]
        for finding in findings:
            finding_type = finding.get("rule", "")
            if finding_type in dangerous_types and finding["severity"] in ["high", "critical"]:
                self.critical_issues.append(
                    {
                        "type": "custom_security",
                        "severity": "critical",
                        "message": f"Dangerous security pattern: {finding_type}",
                        "details": [finding],
                    }
                )

    def _get_recommendation(self) -> str:
        """Get recommendation based on findings."""
        if self.critical_issues:
            return "BLOCK_DEPLOYMENT - Critical security issues found that must be resolved before deployment"
        elif self.blocking_issues:
            return "BLOCK_MERGE - High severity issues found that should be resolved before merging"
        elif self.warnings:
            return "PROCEED_WITH_CAUTION - Some security warnings found, review recommended"
        else:
            return "PROCEED - No critical security issues found"

    def print_summary(self, summary: dict[str, Any]):
        """Print summary of critical issues check."""
        print("\n" + "=" * 60)
        print("🔒 CRITICAL SECURITY ISSUES ANALYSIS")
        print("=" * 60)

        print(f"Critical Issues: {summary['critical_issues_count']}")
        print(f"Blocking Issues: {summary['blocking_issues_count']}")
        print(f"Warnings: {summary['warnings_count']}")
        print(f"Recommendation: {summary['recommendation']}")

        if summary["critical_issues"]:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in summary["critical_issues"]:
                print(f"  - {issue['type']}: {issue['message']}")

        if summary["blocking_issues"]:
            print("\n⚠️ BLOCKING ISSUES:")
            for issue in summary["blocking_issues"]:
                print(f"  - {issue['type']}: {issue['message']}")

        if summary["warnings"]:
            print("\n💡 WARNINGS:")
            for warning in summary["warnings"]:
                print(f"  - {warning}")

        print("\n" + "=" * 60)


def main():
    """Main function for critical issues check."""
    checker = CriticalIssuesChecker()
    summary = checker.check_all_results()

    checker.print_summary(summary)

    # Determine exit code based on findings
    if summary["has_critical_issues"]:
        print("\n❌ CRITICAL ISSUES FOUND - Failing pipeline")
        return 2  # Critical failure
    elif summary["has_blocking_issues"]:
        print("\n⚠️ BLOCKING ISSUES FOUND - Consider addressing before merge")
        return 1  # Warning failure
    else:
        print("\n✅ NO CRITICAL SECURITY ISSUES FOUND")
        return 0  # Success


if __name__ == "__main__":
    sys.exit(main())
