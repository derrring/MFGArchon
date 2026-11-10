#!/usr/bin/env python3
"""
Container security configuration checking script for MFGArchon.
Checks Docker configuration for security best practices.

Its reports name the file and line of each finding, never the line's text.
"""

import json
import sys
from pathlib import Path


class ContainerSecurityChecker:
    """Check container security configuration."""

    def __init__(self):
        self.findings = []
        self.dockerfile_path = Path("Dockerfile")
        self.compose_files = ["docker-compose.yml", "docker-compose.yaml"]

    def check_dockerfile_security(self) -> list[dict]:
        """Check Dockerfile for security issues."""
        findings = []

        if not self.dockerfile_path.exists():
            # This is not necessarily an error for a Python package
            findings.append(
                {
                    "type": "info",
                    "severity": "low",
                    "issue": "No Dockerfile found",
                    "description": "No Dockerfile present for container security analysis",
                    "recommendation": "Consider adding Dockerfile for containerized deployment",
                }
            )
            return findings

        try:
            with open(self.dockerfile_path) as f:
                dockerfile_content = f.read()

            lines = dockerfile_content.splitlines()

            # Check for common security issues
            findings.extend(self.check_dockerfile_lines(lines))

        except Exception as e:
            findings.append(
                {
                    "type": "error",
                    "severity": "medium",
                    "issue": f"Failed to read Dockerfile: {e}",
                    "description": "Could not analyze Dockerfile for security issues",
                    "recommendation": "Ensure Dockerfile is readable and properly formatted",
                }
            )

        return findings

    def check_dockerfile_lines(self, lines: list[str]) -> list[dict]:
        """Check individual Dockerfile lines for security issues."""
        findings = []

        has_user = False
        runs_as_root = True
        has_health_check = False

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Check for latest tag usage
            if line.startswith("FROM") and ":latest" in line:
                findings.append(
                    {
                        "type": "dockerfile",
                        "severity": "medium",
                        "line": line_num,
                        "issue": "Using latest tag",
                        "file": str(self.dockerfile_path),
                        "description": f"Line {line_num}",
                        "recommendation": "Use specific version tags instead of 'latest'",
                    }
                )

            # Check for USER instruction
            if line.startswith("USER"):
                has_user = True
                user = line.split()[1] if len(line.split()) > 1 else ""
                if user not in ["root", "0"]:
                    runs_as_root = False

            # Check for HEALTHCHECK
            if line.startswith("HEALTHCHECK"):
                has_health_check = True

            # Check for ADD vs COPY
            if line.startswith("ADD") and not line.startswith("ADD --"):
                findings.append(
                    {
                        "type": "dockerfile",
                        "severity": "low",
                        "line": line_num,
                        "issue": "Using ADD instead of COPY",
                        "file": str(self.dockerfile_path),
                        "description": f"Line {line_num}",
                        "recommendation": "Use COPY instead of ADD unless auto-extraction is needed",
                    }
                )

            # Check for privileged operations
            if "sudo" in line.lower() or "su -" in line:
                findings.append(
                    {
                        "type": "dockerfile",
                        "severity": "high",
                        "line": line_num,
                        "issue": "Privileged operations in RUN",
                        "file": str(self.dockerfile_path),
                        "description": f"Line {line_num}",
                        "recommendation": "Avoid sudo/su operations in containers",
                    }
                )

            # Check for secrets in environment variables
            if line.startswith("ENV") and any(
                secret in line.upper() for secret in ["PASSWORD", "SECRET", "KEY", "TOKEN"]
            ):
                findings.append(
                    {
                        "type": "dockerfile",
                        "severity": "high",
                        "line": line_num,
                        "issue": "Potential secret in ENV",
                        "file": str(self.dockerfile_path),
                        "description": f"Line {line_num}",
                        "recommendation": "Use Docker secrets or runtime environment variables",
                    }
                )

        # Check for missing USER instruction
        if not has_user:
            findings.append(
                {
                    "type": "dockerfile",
                    "severity": "medium",
                    "issue": "No USER instruction",
                    "description": "Container runs as root by default",
                    "recommendation": "Add USER instruction to run as non-root user",
                }
            )
        elif runs_as_root:
            findings.append(
                {
                    "type": "dockerfile",
                    "severity": "high",
                    "issue": "Running as root user",
                    "description": "Container explicitly runs as root",
                    "recommendation": "Create and use non-root user",
                }
            )

        # Check for missing HEALTHCHECK
        if not has_health_check:
            findings.append(
                {
                    "type": "dockerfile",
                    "severity": "low",
                    "issue": "No HEALTHCHECK instruction",
                    "description": "Container has no health check defined",
                    "recommendation": "Add HEALTHCHECK instruction for better monitoring",
                }
            )

        return findings

    def check_docker_compose_security(self) -> list[dict]:
        """Check docker-compose files for security issues."""
        findings = []

        for compose_file in self.compose_files:
            if Path(compose_file).exists():
                findings.extend(self.check_compose_file(compose_file))

        if not any(Path(f).exists() for f in self.compose_files):
            findings.append(
                {
                    "type": "info",
                    "severity": "low",
                    "issue": "No docker-compose files found",
                    "description": "No docker-compose configuration for analysis",
                    "recommendation": "Consider using docker-compose for complex deployments",
                }
            )

        return findings

    def check_compose_file(self, compose_file: str) -> list[dict]:
        """Check individual docker-compose file."""
        findings = []

        try:
            # Simple text-based analysis since we may not have PyYAML
            with open(compose_file) as f:
                content = f.read()

            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()

                # Check for privileged mode
                if "privileged:" in line and "true" in line:
                    findings.append(
                        {
                            "type": "docker-compose",
                            "severity": "high",
                            "line": line_num,
                            "issue": "Privileged mode enabled",
                            "file": compose_file,
                            "description": f"Line {line_num} in {compose_file}",
                            "recommendation": "Avoid privileged mode unless absolutely necessary",
                        }
                    )

                # Check for host network mode
                if "network_mode:" in line and "host" in line:
                    findings.append(
                        {
                            "type": "docker-compose",
                            "severity": "medium",
                            "line": line_num,
                            "issue": "Host network mode",
                            "file": compose_file,
                            "description": f"Line {line_num} in {compose_file}",
                            "recommendation": "Use bridge network mode for better isolation",
                        }
                    )

                # Check for volume mounts
                if "/:" in line and ("rw" in line or line.endswith(":")):
                    findings.append(
                        {
                            "type": "docker-compose",
                            "severity": "medium",
                            "line": line_num,
                            "issue": "Potentially dangerous volume mount",
                            "file": compose_file,
                            "description": f"Line {line_num} in {compose_file}",
                            "recommendation": "Limit volume mounts and use read-only when possible",
                        }
                    )

        except Exception as e:
            findings.append(
                {
                    "type": "error",
                    "severity": "medium",
                    "issue": f"Failed to read {compose_file}: {e}",
                    "description": "Could not analyze docker-compose file",
                    "recommendation": "Ensure file is readable and properly formatted",
                }
            )

        return findings

    def check_container_runtime_security(self) -> list[dict]:
        """Check container runtime security recommendations."""
        findings = []

        # These are general recommendations
        recommendations = [
            {
                "type": "runtime",
                "severity": "medium",
                "issue": "Container image scanning",
                "description": "Ensure container images are regularly scanned for vulnerabilities",
                "recommendation": "Use Trivy, Clair, or similar tools for image scanning",
            },
            {
                "type": "runtime",
                "severity": "low",
                "issue": "Resource limits",
                "description": "Set appropriate CPU and memory limits",
                "recommendation": "Use --memory and --cpus flags or set limits in docker-compose",
            },
            {
                "type": "runtime",
                "severity": "medium",
                "issue": "Secrets management",
                "description": "Use proper secrets management for sensitive data",
                "recommendation": "Use Docker secrets, Kubernetes secrets, or external secret managers",
            },
            {
                "type": "runtime",
                "severity": "low",
                "issue": "Network security",
                "description": "Use custom networks for container isolation",
                "recommendation": "Create custom Docker networks instead of using default bridge",
            },
        ]

        findings.extend(recommendations)
        return findings

    def generate_report(self) -> dict:
        """Generate comprehensive container security report."""
        print("🔍 Checking container security configuration...")

        all_findings = []

        # Check Dockerfile
        dockerfile_findings = self.check_dockerfile_security()
        all_findings.extend(dockerfile_findings)

        # Check docker-compose files
        compose_findings = self.check_docker_compose_security()
        all_findings.extend(compose_findings)

        # Check runtime recommendations
        runtime_findings = self.check_container_runtime_security()
        all_findings.extend(runtime_findings)

        # Categorize by severity
        by_severity = {"high": [], "medium": [], "low": [], "info": []}
        for finding in all_findings:
            severity = finding.get("severity", "low")
            by_severity[severity].append(finding)

        report = {
            "summary": {
                "total_findings": len(all_findings),
                "high_severity": len(by_severity["high"]),
                "medium_severity": len(by_severity["medium"]),
                "low_severity": len(by_severity["low"]),
                "info_findings": len(by_severity["info"]),
                "dockerfile_exists": self.dockerfile_path.exists(),
                "compose_files_found": [f for f in self.compose_files if Path(f).exists()],
            },
            "findings_by_severity": by_severity,
            "all_findings": all_findings,
        }

        return report

    def save_results(self):
        """Save container security results."""
        report = self.generate_report()

        # Save JSON report
        with open("container-security-report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Generate markdown summary
        markdown_report = self.generate_markdown_report(report)
        with open("container-summary.md", "w") as f:
            f.write(markdown_report)

        print("Container security check completed.")
        print(f"Total findings: {report['summary']['total_findings']}")

        if report["summary"]["high_severity"] > 0:
            print(f"❌ Found {report['summary']['high_severity']} HIGH severity issues!")
            sys.exit(1)
        elif report["summary"]["medium_severity"] > 0:
            print(f"⚠️ Found {report['summary']['medium_severity']} MEDIUM severity issues.")

        print("✅ Container security check completed.")

    def generate_markdown_report(self, report: dict) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# Container Security Check Report")
        lines.append("")

        summary = report["summary"]
        lines.append("## Summary")
        lines.append(f"- **Total Findings**: {summary['total_findings']}")
        lines.append(f"- **High Severity**: {summary['high_severity']}")
        lines.append(f"- **Medium Severity**: {summary['medium_severity']}")
        lines.append(f"- **Low Severity**: {summary['low_severity']}")
        lines.append(f"- **Dockerfile Present**: {summary['dockerfile_exists']}")
        lines.append(
            f"- **Compose Files**: {', '.join(summary['compose_files_found']) if summary['compose_files_found'] else 'None'}"
        )
        lines.append("")

        if summary["high_severity"] > 0:
            lines.append("## ❌ High Severity Issues")
            for finding in report["findings_by_severity"]["high"]:
                lines.append(f"- **{finding['issue']}** ({finding['type']})")
                lines.append(f"  - {finding['description']}")
                lines.append(f"  - Recommendation: {finding['recommendation']}")
            lines.append("")

        if summary["medium_severity"] > 0:
            lines.append("## ⚠️ Medium Severity Issues")
            for finding in report["findings_by_severity"]["medium"]:
                lines.append(f"- **{finding['issue']}** ({finding['type']})")
                lines.append(f"  - {finding['description']}")
                lines.append(f"  - Recommendation: {finding['recommendation']}")
            lines.append("")

        return "\n".join(lines)


def main():
    """Main entry point."""
    checker = ContainerSecurityChecker()
    checker.save_results()


if __name__ == "__main__":
    main()
