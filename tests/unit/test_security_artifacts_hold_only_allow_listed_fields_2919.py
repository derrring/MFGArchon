"""What the security workflow lets leave the runner (#2919).

`.github/workflows/security.yml` is read as data; no scanner runs here.

- The workflow census: each scan job uploads one `<job>-findings.json` and the summary job its three reports; every
  scanner's stdout and stderr go to a file under `$RUNNER_TEMP`; every tool that can call out on its own runs with
  that turned off; no other workflow starts a scanner.
- The fault injection: a marker is planted in every field of each tool's report that the allow-list does not carry.
  The workflow's own `run:` steps that call our scripts run per job, the uploaded files move into one directory per
  artifact, as download-artifact lays them out, and the summary job's steps run there. The marker must reach no
  uploaded file and no line our scripts print, and every planted finding must arrive.
- Our scanners' own reports name a finding's file and line, never the text there.

What the fault injection rests on, and does not test: download-artifact's layout is taken from its documentation (a
directory named after each artifact when no name is given), and the tools' report shapes from their source.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "security.yml"
SCRIPTS = REPO / ".github" / "scripts"
OUR_SCRIPTS = ("security_findings.py", "generate_security_report.py", "check_critical_issues.py")
OUR_SCANNERS = (
    "custom_security_checks.py",
    "custom_secrets_scan.py",
    "container_security_check.py",
    "license_compliance_check.py",
)
SCAN_JOBS = {
    "dependency-scanning": "dependency",
    "static-code-analysis": "static",
    "secrets-scanning": "secrets",
    "container-security": "container",
    "license-compliance": "license",
}
SUMMARY_OUTPUTS = {"security-dashboard.html", "security-summary.md", "security-findings.json"}

#: Each tool's command word, and the flags that turn off what it would otherwise send or fetch on its own.
TOOLS = {
    "safety": ("--disable-optional-telemetry", "--disable-audit-and-monitor"),
    "pip-audit": (),
    "bandit": (),
    "semgrep": ("--metrics=off",),
    "detect-secrets": ("--no-verify",),
    "trufflehog": ("--no-verification", "--no-update"),
    "trivy": ("--disable-telemetry", "--skip-version-check"),
    "./hadolint": (),
}


def _jobs():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def _runs(job):
    return [step["run"] for step in job.get("steps", []) if "run" in step]


def _uploads(job):
    """(artifact name, the paths its upload step names) for each upload step of the job."""
    return [
        (step["with"]["name"], [p.strip() for p in step["with"]["path"].splitlines() if p.strip()])
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]


def _scanner_lines(job):
    """Each line of a run step that starts a scanner, with the tool or script it starts."""
    for run in _runs(job):
        for line in run.splitlines():
            words = line.split()
            if not words:
                continue
            if words[0] in TOOLS:
                yield words[0], line
            elif words[0] == "python" and len(words) > 1 and Path(words[1]).name in OUR_SCANNERS:
                yield Path(words[1]).name, line


def test_the_workflow_uploads_only_our_allow_listed_files_and_logs_no_scanner_output():
    jobs = _jobs()
    assert set(SCAN_JOBS) | {"security-summary"} == set(jobs)

    uploads = {name: _uploads(job) for name, job in jobs.items()}
    wrong = {
        name: uploads[name]
        for name, key in SCAN_JOBS.items()
        if [paths for _, paths in uploads[name]] != [[f"{key}-findings.json"]]
    }
    assert not wrong
    assert {path for _, paths in uploads["security-summary"] for path in paths} == SUMMARY_OUTPUTS

    lines = [(tool, line) for job in jobs.values() for tool, line in _scanner_lines(job)]
    # The census reached every tool and every scanner of ours: each starts on some line.
    assert {tool for tool, _ in lines} == set(TOOLS) | set(OUR_SCANNERS)
    logged = [
        line
        for _, line in lines
        if '> "$RUNNER_TEMP/' not in line or not ("2>&1" in line or '2> "$RUNNER_TEMP/' in line)
    ]
    assert not logged
    calls_out = [line for tool, line in lines if not all(flag in line for flag in TOOLS.get(tool, ()))]
    assert not calls_out
    assert not [line for _, line in lines if "--config=auto" in line or "--config auto" in line]

    # No other workflow starts a scanner, so this census covers every scanner the repository runs.
    elsewhere = [
        f"{other.name}: {line.strip()}"
        for other in sorted(WORKFLOW.parent.glob("*.y*ml"))
        if other != WORKFLOW
        for job in (yaml.safe_load(other.read_text(encoding="utf-8")).get("jobs") or {}).values()
        for _, line in _scanner_lines(job)
    ]
    assert elsewhere == []


MARKER = f"planted_{uuid.uuid4().hex}"


def _planted_reports(m):
    """Each tool's report, in the shape the tool writes, with `m` in every field the allow-list does not carry."""
    return {
        "safety": {
            "report_meta": {"note": m},
            "vulnerabilities": [
                {
                    "vulnerability_id": "70612",
                    "package_name": "jinja2",
                    "analyzed_version": "3.1.2",
                    "advisory": m,
                    "more_info_url": m,
                    "severity": {"source": m, "cvssv2": None, "cvssv3": {"base_severity": "HIGH", "vector": m}},
                }
            ],
        },
        "pip-audit": {
            "dependencies": [
                {
                    "name": "requests",
                    "version": "2.0.0",
                    "vulns": [{"id": "PYSEC-2023-74", "fix_versions": [m], "aliases": [m], "description": m}],
                },
                {"name": "mfgarchon", "skip_reason": m},
            ],
            "fixes": [],
        },
        "bandit": {
            "errors": [{"reason": m}],
            "results": [
                {
                    "code": f"12 api_key = '{m}'\n",
                    "filename": "mfgarchon/a.py",
                    "issue_confidence": "HIGH",
                    "issue_severity": "LOW",
                    "issue_text": m,
                    "line_number": 12,
                    "line_range": [12],
                    "more_info": m,
                    "test_id": "B105",
                    "test_name": m,
                }
            ],
        },
        "semgrep": {
            "errors": [{"message": m}],
            "results": [
                {
                    "check_id": "python.lang.security.audit.eval-detected",
                    "path": "mfgarchon/b.py",
                    "start": {"line": 3, "col": 1},
                    "end": {"line": 3, "col": 9},
                    "extra": {
                        "lines": m,
                        "message": m,
                        "severity": "WARNING",
                        "fingerprint": m,
                        "metadata": {"confidence": "MEDIUM", "references": [m]},
                    },
                }
            ],
        },
        "custom_security_checks": {
            "findings": [
                {
                    "type": "config_security",
                    "severity": "high",
                    "file": "mfgarchon/config.py",
                    "line": 7,
                    "pattern": m,
                    "message": m,
                    "code_snippet": m,
                }
            ],
            "recommendations": [m],
        },
        "detect-secrets": {
            "version": "1.4.0",
            "results": {
                "mfgarchon/c.py": [
                    {
                        "type": "Secret Keyword",
                        "filename": "mfgarchon/c.py",
                        "hashed_secret": m,
                        "is_verified": False,
                        "line_number": 5,
                    }
                ]
            },
        },
        "trufflehog": [
            {
                "SourceMetadata": {
                    "Data": {
                        "Git": {
                            "commit": "a" * 40,
                            "file": "old/creds.txt",
                            "email": m,
                            "repository": m,
                            "timestamp": m,
                            "line": 9,
                        }
                    }
                },
                "SourceName": m,
                "DetectorType": 2,
                "DetectorName": "AWS",
                "Verified": False,
                "Raw": m,
                "RawV2": m,
                "Redacted": m,
                "ExtraData": {"account": m},
            }
        ],
        "custom_secrets_scan": {
            "summary": {"secret_types_found": ["private_key"]},
            "all_findings": [{"file": "deploy.sh", "line": 2, "type": "private_key", "severity": "HIGH", "match": m}],
        },
        "trivy": {
            "ArtifactName": m,
            "Results": [
                {
                    "Target": "app/settings.py",
                    "Class": "secret",
                    "Secrets": [
                        {
                            "RuleID": "aws-access-key-id",
                            "Category": "AWS",
                            "Severity": "CRITICAL",
                            "Title": m,
                            "StartLine": 4,
                            "EndLine": 4,
                            "Code": {"Lines": [{"Number": 4, "Content": m}]},
                            "Match": m,
                        }
                    ],
                },
                {
                    "Target": "Python",
                    "Class": "lang-pkgs",
                    "Vulnerabilities": [
                        {
                            "VulnerabilityID": "CVE-2024-0001",
                            "PkgName": "urllib3",
                            "InstalledVersion": "1.26.0",
                            "FixedVersion": m,
                            "Severity": "HIGH",
                            "Title": m,
                            "Description": m,
                            "References": [m],
                        }
                    ],
                },
            ],
        },
        "hadolint": [
            {"code": "DL3008", "column": 1, "file": "Dockerfile", "level": "warning", "line": 6, "message": m}
        ],
        "container_security_check": {
            "all_findings": [
                {
                    "type": "dockerfile",
                    "severity": "high",
                    "line": 8,
                    "file": "Dockerfile",
                    "issue": "Potential secret in ENV",
                    "description": m,
                    "recommendation": m,
                },
                {
                    "type": "runtime",
                    "severity": "medium",
                    "issue": "Container image scanning",
                    "description": m,
                    "recommendation": m,
                },
            ]
        },
        "license_compliance_check": {
            "approved": [{"name": "numpy", "version": "2.0", "license": "BSD License", "category": "approved"}],
            "review_required": [
                {"name": "gplpkg", "version": "1.0", "license": "GPL-3.0", "category": "review_required"}
            ],
            "prohibited": [],
            "unknown": [{"name": "mystery", "version": "0.1", "license": "UNKNOWN", "category": "unknown"}],
            "summary": {"note": m},
        },
    }


#: Per job, each planted finding as (tool, file or package, line or version, severity), as it must arrive.
ARRIVES = {
    "dependency": {("safety", "jinja2", "3.1.2", "high"), ("pip-audit", "requests", "2.0.0", "unknown")},
    "static": {
        ("bandit", "mfgarchon/a.py", 12, "low"),
        ("semgrep", "mfgarchon/b.py", 3, "medium"),
        ("custom_security_checks", "mfgarchon/config.py", 7, "high"),
    },
    "secrets": {
        ("detect-secrets", "mfgarchon/c.py", 5, "unknown"),
        ("trufflehog", "old/creds.txt", 9, "unknown"),
        ("custom_secrets_scan", "deploy.sh", 2, "high"),
    },
    "container": {
        ("trivy", "app/settings.py", 4, "critical"),
        ("trivy", "urllib3", "1.26.0", "high"),
        ("hadolint", "Dockerfile", 6, "medium"),
        ("container_security_check", "Dockerfile", 8, "high"),
        ("container_security_check", None, None, "medium"),
    },
    "license": {
        ("license_compliance_check", "gplpkg", "1.0", "medium"),
        ("license_compliance_check", "mystery", "0.1", "low"),
    },
}


def _arrived(finding):
    if "package" in finding:
        return finding["tool"], finding["package"], finding["version"], finding["severity"]
    return finding["tool"], finding.get("file"), finding.get("line"), finding["severity"]


def _run(run, cwd, env):
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", run],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _ours(job):
    return [run for run in _runs(job) if any(script in run for script in OUR_SCRIPTS)]


def _workspace(path):
    path.mkdir(parents=True)
    (path / ".github").symlink_to(REPO / ".github")
    return path


def test_a_planted_marker_reaches_no_upload_and_no_log_line_and_every_finding_arrives(tmp_path):
    jobs = _jobs()
    runner_temp = tmp_path / "runner_temp"
    runner_temp.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    env = {**os.environ, "RUNNER_TEMP": str(runner_temp), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    reports = _planted_reports(MARKER)
    download = _workspace(tmp_path / "download")
    printed, uploaded = [], []

    for name, job_key in SCAN_JOBS.items():
        workspace = _workspace(tmp_path / name)
        (boundary,) = [run for run in _ours(jobs[name]) if f"security_findings.py {job_key} " in run]
        # Plant each report where the boundary step reads it.
        for arg in boundary.split()[3:]:
            tool, path = arg.split("=", 1)
            path = Path(path.strip('"').replace("$RUNNER_TEMP", str(runner_temp)))
            target = path if path.is_absolute() else workspace / path
            report = reports[tool]
            text = "\n".join(json.dumps(line) for line in report) if tool == "trufflehog" else json.dumps(report)
            target.write_text(text, encoding="utf-8")
        result = _run(boundary, workspace, env)
        assert result.returncode == 0, result.stdout + result.stderr
        printed.append(result.stdout + result.stderr)

        ((artifact, paths),) = _uploads(jobs[name])
        for path in paths:
            (download / artifact).mkdir(exist_ok=True)
            shutil.copy(workspace / path, download / artifact / path)
            uploaded.append(download / artifact / path)

        findings = json.loads((workspace / f"{job_key}-findings.json").read_text(encoding="utf-8"))
        assert {tool: status["status"] for tool, status in findings["tools"].items()} == dict.fromkeys(
            {arrived[0] for arrived in ARRIVES[job_key]}, "read"
        )
        assert {_arrived(f) for f in findings["findings"]} == ARRIVES[job_key]

    summary_runs = _ours(jobs["security-summary"])
    assert len(summary_runs) == 3
    for run in summary_runs:
        result = _run(run, download, env)
        printed.append(result.stdout + result.stderr)
        if "check_critical_issues.py" not in run:
            assert result.returncode == 0, result.stdout + result.stderr
    assert "results not found" not in printed[-1]
    uploaded += [download / path for _, paths in _uploads(jobs["security-summary"]) for path in paths]

    collected = json.loads((download / "security-findings.json").read_text(encoding="utf-8"))
    assert collected["summary"]["scans_completed"] == list(ARRIVES)
    for job_key, arrives in ARRIVES.items():
        assert {_arrived(f) for f in collected["jobs"][job_key]["findings"]} == arrives
    assert collected["summary"]["overall_status"] == "HIGH"

    assert [path.name for path in uploaded if MARKER in path.read_text(encoding="utf-8")] == []
    assert [text for text in printed if MARKER in text] == []


def _load(script):
    spec = importlib.util.spec_from_file_location(Path(script).stem, SCRIPTS / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_our_scanners_report_the_file_and_line_not_the_text(tmp_path):
    container = _load("container_security_check.py").ContainerSecurityChecker()
    dockerfile = container.check_dockerfile_lines(["FROM python:3.12", f"ENV API_TOKEN={MARKER}", "USER app"])
    assert {(f["issue"], f.get("file"), f.get("line")) for f in dockerfile} >= {
        ("Potential secret in ENV", "Dockerfile", 2)
    }

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "service_config.py").write_text(f'\n\napi_key = "{MARKER}"\n', encoding="utf-8")
    checks = _load("custom_security_checks.py").CustomSecurityChecker()
    checks.package_root = tmp_path / "pkg"
    checks.check_configuration_security()
    assert {(f["type"], f["line"]) for f in checks.findings} == {("config_security", 3)}

    assert MARKER not in json.dumps(dockerfile) + json.dumps(checks.findings)
