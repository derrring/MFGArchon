#!/usr/bin/env python3
"""The security workflow's one reader of raw scan reports, and the one writer of what its scan jobs upload (#2919).

Each scan job in `.github/workflows/security.yml` runs its tools, then runs this script on their reports:

    security_findings.py <job> <tool>=<report> ... [trufflehog.log=<log>]

From TruffleHog's log it reads only the two integers its final "finished scanning" line carries. It writes `<job>-findings.json`, whose findings hold only the keys in FIELDS, and the job uploads that file alone. A
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
_STATUS_FIELDS = {"status": str, "findings": int, "error": str, "scanned": int, "errors": int}


#: The statuses a tool can have in a findings file.
_STATUSES = ("read", "missing", "unreadable", "failed")


def _tool_status(job, tool, status):
    if tool not in JOBS[job]:
        raise KeyError("tool")
    if status.get("status") not in _STATUSES:
        raise ValueError("status")
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
    # pkg/report/writer.go: every report trivy writes carries SchemaVersion 2; Results is omitted when empty.
    if not isinstance(report, dict) or "SchemaVersion" not in report:
        raise TypeError("not a trivy report")
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
    # hadolint exits 1 on findings and on failure alike, so the report's shape must say it finished: a JSON list.
    if not isinstance(report, list):
        raise TypeError("not a hadolint report")
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


def _bandit_counts(report):
    # bandit/core/manager.py: metrics.begin(name) runs before a file is parsed, and a file it cannot open, parse or
    # scan goes to `skipped`; bandit/formatters/json.py writes `skipped` as `errors` and the metrics, keyed by file
    # with a `_totals` entry, as `metrics`.
    failed = {error["filename"] for error in report["errors"]}
    return len((set(report["metrics"]) - {"_totals"}) | failed), len(failed)


def _semgrep_counts(report):
    # semgrep_output_v1.atd: `paths.scanned` lists every target, and a `cli_error` may carry the `path` it is about.
    return len(report["paths"]["scanned"]), len({error["path"] for error in report["errors"] if error.get("path")})


def _pip_audit_counts(report):
    # pip_audit/_format/json.py: a dependency it could not audit is written with a `skip_reason`.
    return len(report["dependencies"]), sum("skip_reason" in dependency for dependency in report["dependencies"])


def _safety_counts(report):
    # safety/formatters/json.py: `scanned_packages`; the report records no package it failed on.
    return len(report["scanned_packages"]), None


def _container_security_check_counts(report):
    summary = report["summary"]
    errors = sum(finding.get("type") == "error" for finding in report["all_findings"])
    return int(summary["dockerfile_exists"]) + len(summary["compose_files_found"]), errors


def _trivy_counts(report):
    # pkg/scan/ospkg/scan.go returns a result for a detected OS whether or not it has vulnerabilities, the language
    # scanner one per application, and pkg/report/json.go keeps every result that names a target.
    return sum(result.get("Class") in ("os-pkgs", "lang-pkgs") for result in report.get("Results", [])), None


def _custom_secrets_scan_counts(report):
    # It reads every tracked file; one it cannot read stops the scan, which then writes no report.
    return report["summary"]["text_files_scanned"], None


def _custom_security_checks_counts(report):
    return report["summary"]["files_scanned"], report["summary"]["files_unreadable"]


def _trufflehog_log_counts(lines):
    # main.go: with --json the log is JSON on stderr, and a scan that finishes logs "finished scanning" with the
    # chunks and bytes it scanned. Only those two integers are read from the log.
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("msg") == "finished scanning":
            chunks, scanned_bytes = entry["chunks"], entry["bytes"]
            if not isinstance(chunks, int) or not isinstance(scanned_bytes, int):
                raise TypeError("counts")
            return (chunks if scanned_bytes else 0), None
    return 0, None


def _license_compliance_check_counts(report):
    # It records no package it failed on; a pip-licenses failure leaves the total at 0.
    return report["summary"]["total_packages"], None


#: For each tool whose report says so, (how many files or packages it scanned, how many of them it could not); None
#: where the report does not say. The other tools' reports give neither.
_COUNTS = {
    "bandit": _bandit_counts,
    "semgrep": _semgrep_counts,
    "pip-audit": _pip_audit_counts,
    "safety": _safety_counts,
    "container_security_check": _container_security_check_counts,
    "license_compliance_check": _license_compliance_check_counts,
    "trivy": _trivy_counts,
    "custom_secrets_scan": _custom_secrets_scan_counts,
    "custom_security_checks": _custom_security_checks_counts,
}

#: Tools whose counts come from their log, given as `<tool>.log=<path>`, not from their report.
_LOG_COUNTS = {"trufflehog": _trufflehog_log_counts}

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
    logs = {f"{tool}.log" for tool in tools if tool in _LOG_COUNTS}
    if set(reports) != set(tools) | logs:
        expected = ", ".join([*tools, *sorted(logs)])
        raise SystemExit(f"{job} takes exactly these inputs: {expected}; given: {', '.join(reports)}")
    status, findings = {}, []
    for tool, read in tools.items():
        path = Path(reports[tool])
        if not path.is_file():
            status[tool] = {"status": "missing"}
            continue
        try:
            report = _load(tool, path)
            read_now = [_finding(tool=tool, **fields) for fields in read(report)]
            counts = {}
            if tool in _COUNTS or tool in _LOG_COUNTS:
                if tool in _LOG_COUNTS:
                    log = Path(reports[f"{tool}.log"])
                    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.is_file() else []
                    scanned, errors = _LOG_COUNTS[tool](lines)
                else:
                    scanned, errors = _COUNTS[tool](report)
                counts = {"scanned": scanned, **({"errors": errors} if errors is not None else {})}
                # A tool that scanned nothing, or could scan none of what it found, did not run a scan.
                if scanned == 0 or (errors is not None and errors >= scanned):
                    status[tool] = {"status": "failed", "error": "nothing scanned", **counts}
                    continue
        except _ToolFailedError as failure:
            status[tool] = {"status": "failed", "error": str(failure)}
            continue
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            # The class only: a message can quote the report.
            status[tool] = {"status": "unreadable", "error": type(error).__name__}
            continue
        status[tool] = {"status": "read", "findings": len(read_now), **counts}
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
            tools = {tool: tools.get(tool, {"status": "missing"}) for tool in JOBS[job]}
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            jobs[job] = {"status": "unreadable", "error": type(error).__name__}
            continue
        complete = all(status["status"] == "read" for status in tools.values())
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
        # Each tool's status beside its findings, with what it could not scan.
        "tools": [
            {"job": job, "tool": tool, **status}
            for job, data in jobs.items()
            for tool, status in data.get("tools", {}).items()
        ],
    }
    return {"summary": summary, "jobs": jobs}


def main(argv):
    if len(argv) == 2 and argv[0] == "collect":
        result = collect(argv[1])
        Path("security-findings.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        for job, data in result["jobs"].items():
            print(f"{job}: {data['status']}" + (f", {data['counts']}" if "counts" in data else ""))
            for tool, status in data.get("tools", {}).items():
                print(f"  {tool}: " + ", ".join(f"{key} {value}" for key, value in status.items()))
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
