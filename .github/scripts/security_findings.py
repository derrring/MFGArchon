#!/usr/bin/env python3
"""The security workflow's one reader of raw scan reports, and the one writer of what its scan jobs upload (#2919).

Each scan job in `.github/workflows/security.yml` runs its tools, then runs this script on their reports:

    security_findings.py <job> <tool>=<report> ...

It writes `<job>-findings.json`, whose findings hold only the keys in FIELDS, and the job uploads that file alone. A
tool's matched text, code lines, messages, descriptions and every other field of its report stay on the runner. It
prints each report's status and the number of findings it read, never a value read from a report. It exits 1 when a
report is missing, cannot be read, or says its tool did not finish, after writing the file.

The summary job downloads the five per-job files into one directory, which their unique names allow, and runs

    security_findings.py collect <download directory>

which reads `<job>-findings.json` for each job, keeps the same keys, and writes `security-findings.json`. A job counts as
completed only when its file was read and so was each of its tools' reports; otherwise the overall status is INCOMPLETE
and it exits 1, after writing the file. `generate_security_report.py` and `check_critical_issues.py` read that file.
"""

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

#: The keys a finding can hold, and the type of each value. Nothing else in a report reaches a findings file.
FIELDS = {
    "tool": str,
    "rule": str,
    "file": str,
    "line": int,
    "commit": str,
    "severity": str,
    "confidence": str,
    "package": str,
    "version": str,
    "advisory": str,
    "license": str,
    "status": str,
}

SEVERITIES = ("critical", "high", "medium", "low", "unknown")

#: Each tool's severity word, lower-cased, to one of SEVERITIES; a tool that gives none counts as unknown.
_SEVERITY = {
    "critical": "critical",
    "high": "high",
    "error": "high",
    "medium": "medium",
    "moderate": "medium",
    "warning": "medium",
    "low": "low",
    "info": "low",
    "style": "low",
}

#: license_compliance_check.py's categories that are findings, with a severity each. "approved" is not a finding.
_LICENSE_SEVERITY = {"prohibited": "high", "review_required": "medium", "unknown": "low"}

#: The jobs whose counts set the overall status, as generate_security_report.py counted them.
STATUS_JOBS = ("dependency", "static", "secrets")


def _severity(word):
    return _SEVERITY.get(word.lower(), "unknown") if isinstance(word, str) else "unknown"


def _finding(**fields):
    finding = {key: value for key, value in fields.items() if value is not None}
    finding.setdefault("severity", "unknown")
    for key, value in finding.items():
        if not isinstance(value, FIELDS[key]) or isinstance(value, bool):
            raise TypeError(f"{key} is a {type(value).__name__}")
    if finding["severity"] not in SEVERITIES:
        raise ValueError("severity outside SEVERITIES")
    return finding


#: The keys of a tool's status in a findings file, and the type of each value.
_STATUS_FIELDS = {"status": str, "findings": int, "error": str}


def _tool_status(job, tool, status):
    if tool not in JOBS[job]:
        raise KeyError("tool")
    for key, value in status.items():
        if not isinstance(value, _STATUS_FIELDS[key]) or isinstance(value, bool):
            raise TypeError(f"{key} is a {type(value).__name__}")
    return status


def _safety(report):
    for vuln in report["vulnerabilities"]:
        cvssv3 = (vuln.get("severity") or {}).get("cvssv3") or {}
        yield {
            "package": vuln["package_name"],
            "version": vuln["analyzed_version"],
            "advisory": vuln["vulnerability_id"],
            "severity": _severity(cvssv3.get("base_severity")),
        }


def _pip_audit(report):
    for dependency in report["dependencies"]:
        for vuln in dependency.get("vulns", []):
            yield {"package": dependency["name"], "version": dependency["version"], "advisory": vuln["id"]}


def _bandit(report):
    for issue in report["results"]:
        yield {
            "rule": issue["test_id"],
            "file": issue["filename"],
            "line": issue["line_number"],
            "severity": _severity(issue["issue_severity"]),
            "confidence": issue["issue_confidence"].lower(),
        }


def _semgrep(report):
    for result in report["results"]:
        extra = result["extra"]
        confidence = extra.get("metadata", {}).get("confidence")
        yield {
            "rule": result["check_id"],
            "file": result["path"],
            "line": result["start"]["line"],
            "severity": _severity(extra.get("severity")),
            "confidence": confidence.lower() if isinstance(confidence, str) else None,
        }


def _custom_security_checks(report):
    for finding in report["findings"]:
        yield {
            "rule": finding["type"],
            "file": finding["file"],
            "line": finding.get("line"),
            "severity": _severity(finding["severity"]),
        }


def _detect_secrets(report):
    for file, secrets in report["results"].items():
        for secret in secrets:
            yield {"rule": secret["type"], "file": file, "line": secret["line_number"]}


def _trufflehog(report):
    for result in report:
        # Every Git field is omitted from the JSON when empty: a match in a binary file has no line.
        git = result["SourceMetadata"]["Data"]["Git"]
        yield {
            "rule": result["DetectorName"],
            "file": git.get("file"),
            "line": git.get("line"),
            "commit": git.get("commit"),
        }


def _custom_secrets_scan(report):
    for finding in report["all_findings"]:
        yield {
            "rule": finding["type"],
            "file": finding["file"],
            "line": finding["line"],
            "severity": _severity(finding["severity"]),
        }


def _trivy(report):
    for result in report.get("Results", []):
        for vuln in result.get("Vulnerabilities") or []:
            # These fields are omitted from the JSON when empty.
            yield {
                "package": vuln.get("PkgName"),
                "version": vuln.get("InstalledVersion"),
                "advisory": vuln.get("VulnerabilityID"),
                "severity": _severity(vuln.get("Severity")),
            }
        for secret in result.get("Secrets") or []:
            yield {
                "rule": secret["RuleID"],
                "file": result["Target"],
                "line": secret["StartLine"],
                "severity": _severity(secret["Severity"]),
            }


class _ToolFailedError(Exception):
    """A report that says its own tool did not finish. The message is ours, never the report's."""


def _hadolint(report):
    # DL1000 is hadolint's report of a Dockerfile it could not parse: nothing was linted.
    if any(issue["code"] == "DL1000" for issue in report):
        raise _ToolFailedError("DL1000")
    for issue in report:
        yield {
            "rule": issue["code"],
            "file": issue["file"],
            "line": issue["line"],
            "severity": _severity(issue["level"]),
        }


def _container_security_check(report):
    for finding in report["all_findings"]:
        yield {
            "rule": finding["issue"],
            "file": finding.get("file"),
            "line": finding.get("line"),
            "severity": _severity(finding["severity"]),
        }


def _license_compliance_check(report):
    for status, severity in _LICENSE_SEVERITY.items():
        for package in report[status]:
            yield {
                "package": package["name"],
                "version": package["version"],
                "license": package["license"],
                "status": status,
                "severity": severity,
            }


#: Each job's tools, with the reader of each report.
JOBS = {
    "dependency": {"safety": _safety, "pip-audit": _pip_audit},
    "static": {"bandit": _bandit, "semgrep": _semgrep, "custom_security_checks": _custom_security_checks},
    "secrets": {
        "detect-secrets": _detect_secrets,
        "trufflehog": _trufflehog,
        "custom_secrets_scan": _custom_secrets_scan,
    },
    "container": {"trivy": _trivy, "hadolint": _hadolint, "container_security_check": _container_security_check},
    "license": {"license_compliance_check": _license_compliance_check},
}

#: Reports with one JSON document per line.
_JSON_LINES = {"trufflehog"}


def _load(tool, path):
    text = path.read_text(encoding="utf-8")
    if tool in _JSON_LINES:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _counts(findings):
    counted = Counter(finding["severity"] for finding in findings)
    return {severity: counted[severity] for severity in SEVERITIES}


def findings_for(job, reports):
    """Read each tool's report for `job` and keep FIELDS. `reports` maps every tool of the job to its report path."""
    tools = JOBS[job]
    if set(reports) != set(tools):
        raise SystemExit(f"{job} takes exactly these tools: {', '.join(tools)}; given: {', '.join(reports)}")
    status, findings = {}, []
    for tool, read in tools.items():
        path = Path(reports[tool])
        if not path.is_file():
            status[tool] = {"status": "missing"}
            continue
        try:
            read_now = [_finding(tool=tool, **fields) for fields in read(_load(tool, path))]
        except _ToolFailedError as failure:
            status[tool] = {"status": "failed", "error": str(failure)}
            continue
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            # The class only: a message can quote the report.
            status[tool] = {"status": "unreadable", "error": type(error).__name__}
            continue
        status[tool] = {"status": "read", "findings": len(read_now)}
        findings.extend(read_now)
    return {"job": job, "tools": status, "counts": _counts(findings), "findings": findings}


def collect(root):
    """Read the five `<job>-findings.json` in `root`, keep FIELDS, and add the summary."""
    jobs = {}
    for job in JOBS:
        path = Path(root) / f"{job}-findings.json"
        if not path.is_file():
            jobs[job] = {"status": "missing"}
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            findings = [_finding(**finding) for finding in data["findings"]]
            tools = {tool: _tool_status(job, tool, status) for tool, status in data["tools"].items()}
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            jobs[job] = {"status": "unreadable", "error": type(error).__name__}
            continue
        complete = set(tools) == set(JOBS[job]) and all(status["status"] == "read" for status in tools.values())
        jobs[job] = {
            "status": "read" if complete else "incomplete",
            "tools": tools,
            "counts": _counts(findings),
            "findings": findings,
        }

    totals = Counter()
    for job in STATUS_JOBS:
        totals.update(jobs[job].get("counts", {}))
    failed = [job for job, data in jobs.items() if data["status"] != "read"]
    # A finding its tool gives no severity for ranks below high and above medium.
    ranked = ("critical", "high", "unknown", "medium", "low")
    overall = "INCOMPLETE" if failed else next((s.upper() for s in ranked if totals[s]), "PASS")
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_issues": sum(totals.values()),
        **{f"{severity}_issues": totals[severity] for severity in SEVERITIES},
        "scans_completed": [job for job in jobs if job not in failed],
        "scans_failed": failed,
        "overall_status": overall,
    }
    return {"summary": summary, "jobs": jobs}


def main(argv):
    if len(argv) == 2 and argv[0] == "collect":
        result = collect(argv[1])
        Path("security-findings.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        for job, data in result["jobs"].items():
            print(f"{job}: {data['status']}" + (f", {data['counts']}" if "counts" in data else ""))
        print(f"overall status: {result['summary']['overall_status']}")
        return 1 if result["summary"]["scans_failed"] else 0
    if not argv or argv[0] not in JOBS or not all("=" in arg for arg in argv[1:]):
        raise SystemExit(__doc__)
    job = argv[0]
    result = findings_for(job, dict(arg.split("=", 1) for arg in argv[1:]))
    Path(f"{job}-findings.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for tool, status in result["tools"].items():
        print(f"{tool}: " + ", ".join(f"{key} {value}" for key, value in status.items()))
    print(f"{job}: {result['counts']}")
    return 1 if any(status["status"] != "read" for status in result["tools"].values()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
