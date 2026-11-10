"""What the security workflow lets leave the runner, and what its summary reads (#2919).

`.github/workflows/security.yml` is read as data. No real scanner runs here: each third-party tool is a stub that
writes a planted report where the workflow's flags tell it to, and exits with the code the tool gives when it has
found something.

- The workflow census:
  - each scan job uploads one `<job>-findings.json`, the five names distinct, and the summary job its three reports;
  - every scanner line sends stdout and stderr to a file under `$RUNNER_TEMP`, and no other line reads that directory;
  - every tool that can call out on its own runs with that turned off;
  - no job can be skipped, the steps that must run after a failure carry `if: always()`, the actions are a fixed set,
    and no other workflow starts a scanner.
- The fault injection: a marker is planted in every string field of each tool's report that the allow-list does not
  carry. Each scan job's own scanner steps run with the stubs; our scanners' reports are planted under the names they
  write, and the job's boundary step runs. The uploaded files merge into one directory, as the download step lays
  them out, and the summary job's steps run there. The marker must reach no uploaded file and no line our scripts
  print, and every planted finding must arrive.
- A tool that never ran, or that stopped on a fatal error but still wrote a well-formed report: its job and the
  summary read INCOMPLETE and the check blocks. A summary
  over an empty download does the same.
- Our scanners' own reports name a finding's file and line, never the text there.

What this rests on, and does not test: download-artifact's layout, from its `src/download-artifact.ts` at v8; and the
tools' report shapes and exit codes, from their source. A dispatch run of the workflow tests both.
"""

import fnmatch
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
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
ACTIONS = {
    "actions/checkout",
    "actions/setup-python",
    "actions/cache",
    "actions/upload-artifact",
    "actions/download-artifact",
    "docker/setup-buildx-action",
    "actions/github-script",
}

#: Each tool's command word, and the flags that turn off what it would otherwise send or fetch on its own.
TOOLS = {
    "safety": ("--disable-optional-telemetry", "--disable-audit-and-monitor"),
    "pip-audit": (),
    "bandit": (),
    "semgrep": ("--metrics=off", "--disable-version-check"),
    "detect-secrets": ("--no-verify",),
    "trufflehog": ("--no-verification", "--no-update"),
    "trivy": ("--disable-telemetry", "--skip-version-check"),
    "./hadolint": (),
}

#: Modules a `python -m` line could start a scanner with.
SCANNER_MODULES = {"bandit", "safety", "pip_audit", "semgrep", "detect_secrets"}

#: Each tool's exit code when it finishes having found something, as its source sets it.
FINDING_EXIT = {
    "safety": 64,  # safety/constants.py: EXIT_CODE_VULNERABILITIES_FOUND
    "pip-audit": 1,  # pip_audit/_cli.py: sys.exit(1) when a vulnerability is found
    "bandit": 1,  # bandit/cli/main.py: sys.exit(1) when results_count > 0
    "semgrep": 0,  # 1 only with --error
    "detect-secrets": 0,  # detect_secrets/main.py: main returns 0 after a scan
    "trufflehog": 0,  # main.go: 183 only with --fail
    "trivy": 0,  # --exit-code defaults to 0
    "./hadolint": 1,  # app/Main.hs: exitFailure on a rule at or above --failure-threshold, default info
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
    """Each line of a run step that starts a scanner, with the tool, module or script it starts."""
    for run in _runs(job):
        for line in run.splitlines():
            words = line.split()
            if words[:1] == ["if"]:
                words = words[1:]
            if not words:
                continue
            if words[0] in TOOLS:
                yield words[0], line
            elif words[0] == "python" and len(words) > 2 and words[1] == "-m" and words[2] in SCANNER_MODULES:
                yield f"python -m {words[2]}", line
            elif words[0] == "python" and len(words) > 1 and Path(words[1]).name in OUR_SCANNERS:
                yield Path(words[1]).name, line


def _calls_ours(run):
    """Whether a run step starts one of our summary-side scripts, by command and not by mention."""
    return any(
        line.split()[:2] == ["python", f".github/scripts/{script}"]
        for line in run.splitlines()
        for script in OUR_SCRIPTS
    )


def _boundary(job, job_key):
    (run,) = [run for run in _runs(job) if f"security_findings.py {job_key} " in run]
    return run


def _boundary_inputs(run):
    """The boundary step's `tool=path` arguments."""
    return {tool: path.strip('"') for tool, path in (arg.split("=", 1) for arg in run.split()[3:])}


def test_the_workflow_uploads_only_our_allow_listed_files_and_logs_no_scanner_output():
    text = WORKFLOW.read_text(encoding="utf-8")
    jobs = _jobs()
    assert set(SCAN_JOBS) | {"security-summary"} == set(jobs)

    uploads = {name: _uploads(job) for name, job in jobs.items()}
    wrong = {
        name: uploads[name]
        for name, key in SCAN_JOBS.items()
        if [paths for _, paths in uploads[name]] != [[f"{key}-findings.json"]]
    }
    assert not wrong
    assert len({paths[0] for name in SCAN_JOBS for _, paths in uploads[name]}) == len(SCAN_JOBS)
    assert {path for _, paths in uploads["security-summary"] for path in paths} == SUMMARY_OUTPUTS

    lines = [(tool, line) for job in jobs.values() for tool, line in _scanner_lines(job)]
    # The census reached every tool and every scanner of ours, and nothing else: each starts on some line.
    assert {tool for tool, _ in lines} == set(TOOLS) | set(OUR_SCANNERS)
    logged = [
        line
        for _, line in lines
        if '> "$RUNNER_TEMP/' not in line or not ("2>&1" in line or '2> "$RUNNER_TEMP/' in line)
    ]
    assert not logged
    piped = [line for _, line in lines if any(word in line.split() for word in ("|", "cat", "tee", "jq"))]
    assert not piped
    calls_out = [line for tool, line in lines if not all(flag in line for flag in TOOLS.get(tool, ()))]
    assert not calls_out
    assert not [line for _, line in lines if "--config=auto" in line or "--config auto" in line]
    assert "GITHUB_STEP_SUMMARY" not in text

    # $RUNNER_TEMP is written by the scanner lines and read only by the boundary.
    scanner_lines = {line for _, line in lines}
    reads = [
        line.strip()
        for job in jobs.values()
        for run in _runs(job)
        for line in run.splitlines()
        if "$RUNNER_TEMP" in line
        and line not in scanner_lines
        and "security_findings.py " not in line
        and not line.strip().startswith("cp .secrets.baseline ")
    ]
    assert reads == []

    # A boundary input in the workspace is the file its scanner writes under that name.
    for name, key in SCAN_JOBS.items():
        for tool, path in _boundary_inputs(_boundary(jobs[name], key)).items():
            if not path.startswith("$RUNNER_TEMP"):
                assert f'open("{path}", "w")' in (SCRIPTS / f"{tool}.py").read_text(encoding="utf-8"), (tool, path)

    # Every job runs on every trigger; what must run after a failure says so.
    assert {name: job.get("if") for name, job in jobs.items()} == {
        **dict.fromkeys(SCAN_JOBS),
        "security-summary": "always()",
    }
    after_failure = [
        step
        for job in jobs.values()
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact@") or _calls_ours(step.get("run", ""))
    ]
    # Per scan job its boundary and its upload; in the summary, collect, generate, upload and check.
    assert len(after_failure) == 2 * len(SCAN_JOBS) + 4
    assert [step["name"] for step in after_failure if step.get("if") != "always()"] == []
    assert {str(step["uses"]).split("@")[0] for job in jobs.values() for step in job["steps"] if "uses" in step} <= (
        ACTIONS
    )

    (download,) = [
        step
        for step in jobs["security-summary"]["steps"]
        if str(step.get("uses", "")).startswith("actions/download-artifact@")
    ]
    assert download["with"].get("merge-multiple") is True
    names = [name for job in SCAN_JOBS for name, _ in uploads[job]]
    assert [name for name in names if not fnmatch.fnmatch(name, download["with"]["pattern"])] == []

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
    """Each tool's report, in the shape the tool writes, with `m` in every string field the allow-list does not carry."""
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
            "version": m,
            "results": {
                "mfgarchon/c.py": [
                    {
                        "type": "Secret Keyword",
                        "filename": m,
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
            },
            # A match in a binary file: TruffleHog omits an empty field, and this one has no line.
            {
                "SourceMetadata": {"Data": {"Git": {"commit": "b" * 40, "file": "docs/logo.png", "email": m}}},
                "DetectorName": "PrivateKey",
                "Raw": m,
            },
        ],
        "custom_secrets_scan": {
            "summary": {"secret_types_found": [m]},
            "all_findings": [{"file": "deploy.sh", "line": 2, "type": "private_key", "severity": "HIGH", "match": m}],
        },
        "trivy": {
            "ArtifactName": m,
            "Results": [
                {
                    "Target": "app/settings.py",
                    "Class": m,
                    "Secrets": [
                        {
                            "RuleID": "aws-access-key-id",
                            "Category": m,
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
                    "Target": m,
                    "Class": m,
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
                        },
                        # Trivy omits an empty field, and this one has no installed version.
                        {"VulnerabilityID": "CVE-2024-0002", "PkgName": "libfoo", "Severity": "MEDIUM", "Title": m},
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
                    "type": m,
                    "severity": "high",
                    "line": 8,
                    "file": "Dockerfile",
                    "issue": "Potential secret in ENV",
                    "description": m,
                    "recommendation": m,
                },
                {
                    "type": m,
                    "severity": "medium",
                    "issue": "Container image scanning",
                    "description": m,
                    "recommendation": m,
                },
            ]
        },
        "license_compliance_check": {
            "approved": [{"name": m, "version": m, "license": m, "category": m}],
            "review_required": [{"name": "gplpkg", "version": "1.0", "license": "GPL-3.0", "category": m}],
            "prohibited": [],
            "unknown": [{"name": "mystery", "version": "0.1", "license": "UNKNOWN", "category": m}],
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
        ("trufflehog", "docs/logo.png", None, "unknown"),
        ("custom_secrets_scan", "deploy.sh", 2, "high"),
    },
    "container": {
        ("trivy", "app/settings.py", 4, "critical"),
        ("trivy", "urllib3", "1.26.0", "high"),
        ("trivy", "libfoo", None, "medium"),
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
        return finding["tool"], finding["package"], finding.get("version"), finding["severity"]
    return finding["tool"], finding.get("file"), finding.get("line"), finding["severity"]


_STUB = """#!{python}
import sys
args, out = sys.argv[1:], None
for i, arg in enumerate(args):
    for flag in ("-o", "--output", "--save-json", "--baseline"):
        if arg == flag and i + 1 < len(args):
            out = args[i + 1]
        elif arg.startswith(flag + "="):
            out = arg.split("=", 1)[1]
text = open({report!r}, encoding="utf-8").read()
if out:
    open(out, "w", encoding="utf-8").write(text)
else:
    sys.stdout.write(text)
sys.exit({code})
"""


class _Run:
    """The workflow's steps, run on a scratch runner with stub tools and planted reports."""

    def __init__(self, root):
        self.root = root
        self.jobs = _jobs()
        self.reports = _planted_reports(MARKER)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(sys.executable)
        self.path = os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"])
        self.printed = []

    def _env(self, workspace):
        """Each job its own runner temp directory, as on GitHub, and its own stubs ahead of the system path."""
        runner_temp = workspace / ".runner_temp"
        runner_temp.mkdir(exist_ok=True)
        path = f"{workspace / '.stubs'}{os.pathsep}{self.path}"
        return {"RUNNER_TEMP": str(runner_temp), "PATH": path, "HOME": str(self.root), "LC_ALL": "C.UTF-8"}

    def _workspace(self, path):
        path.mkdir(parents=True)
        (path / ".github").symlink_to(REPO / ".github")
        shutil.copy(REPO / ".secrets.baseline", path / ".secrets.baseline")
        return path

    def _bash(self, run, cwd, env):
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", run],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return result

    def _report_file(self, tool):
        path = self.root / "planted" / tool
        path.parent.mkdir(exist_ok=True)
        report = self.reports[tool]
        text = "\n".join(json.dumps(line) for line in report) if tool == "trufflehog" else json.dumps(report)
        path.write_text(text, encoding="utf-8")
        return path

    def scan_job(self, name, absent=(), fatal=None):
        """`fatal`, a (tool, report text, exit code): that tool's stub writes the text and exits with the code."""
        """Run a scan job's scanner steps and boundary step; return (workspace, the boundary's result)."""
        job, key = self.jobs[name], SCAN_JOBS[name]
        label = "-".join(absent) or (f"fatal-{fatal[0]}" if fatal else "all")
        workspace = self._workspace(self.root / "jobs" / f"{name}-{label}".replace("./", ""))
        stubs = workspace / ".stubs"
        stubs.mkdir()
        env = self._env(workspace)
        for step in job["steps"]:
            tools = [tool for tool, _ in _scanner_lines({"steps": [step]}) if tool in TOOLS]
            if not tools:
                continue
            for tool in tools:
                if tool in absent:
                    assert shutil.which(tool.lstrip("./"), path=env["PATH"]) is None
                    continue
                stub = workspace / "hadolint" if tool == "./hadolint" else stubs / tool
                report, code = self._report_file(tool.lstrip("./")), FINDING_EXIT[tool]
                if fatal and fatal[0] == tool:
                    report, code = self.root / "planted" / f"fatal-{tool.lstrip('./')}", fatal[2]
                    report.write_text(fatal[1], encoding="utf-8")
                stub.write_text(_STUB.format(python=sys.executable, report=str(report), code=code))
                stub.chmod(0o755)
            result = self._bash(step["run"], workspace, env)
            assert result.returncode == 0, (step["name"], result.stdout + result.stderr)
        boundary = _boundary(job, key)
        # Our scanners' reports, under the names they write.
        for tool, path in _boundary_inputs(boundary).items():
            if not path.startswith("$RUNNER_TEMP"):
                (workspace / path).write_text(json.dumps(self.reports[tool]), encoding="utf-8")
        result = self._bash(boundary, workspace, env)
        self.printed.append(result.stdout + result.stderr)
        return workspace, result

    def summary(self, uploaded):
        """Merge `uploaded` as the download step does, run the summary job's steps; return (workspace, results)."""
        steps = self.jobs["security-summary"]["steps"]
        (download,) = [s for s in steps if str(s.get("uses", "")).startswith("actions/download-artifact@")]
        assert download["with"].get("merge-multiple") is True
        workspace = self._workspace(self.root / f"summary-{len(self.printed)}")
        target = workspace / download["with"]["path"]
        target.mkdir()
        for artifact, path in uploaded:
            if fnmatch.fnmatch(artifact, download["with"]["pattern"]):
                assert not (target / path.name).exists(), path.name
                shutil.copy(path, target / path.name)
        results = []
        for run in [run for run in _runs({"steps": steps}) if _calls_ours(run)]:
            result = self._bash(run, workspace, self._env(workspace))
            self.printed.append(result.stdout + result.stderr)
            results.append(result)
        return workspace, results

    def uploads(self, name, workspace):
        ((artifact, paths),) = _uploads(self.jobs[name])
        return [(artifact, workspace / path) for path in paths]


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    run = _Run(tmp_path_factory.mktemp("security"))
    done = {name: run.scan_job(name) for name in SCAN_JOBS}
    return run, done


def test_a_planted_marker_reaches_no_upload_and_no_log_line_and_every_finding_arrives(complete):
    run, done = complete
    uploaded = []
    for name, (workspace, result) in done.items():
        assert result.returncode == 0, result.stdout + result.stderr
        uploaded += run.uploads(name, workspace)
        findings = json.loads((workspace / f"{SCAN_JOBS[name]}-findings.json").read_text(encoding="utf-8"))
        assert {tool: status["status"] for tool, status in findings["tools"].items()} == dict.fromkeys(
            {arrived[0] for arrived in ARRIVES[SCAN_JOBS[name]]}, "read"
        )
        assert {_arrived(f) for f in findings["findings"]} == ARRIVES[SCAN_JOBS[name]]

    workspace, results = run.summary(uploaded)
    assert [r.returncode for r in results] == [0, 0, 0], [r.stdout + r.stderr for r in results]
    assert "results not found" not in results[-1].stdout
    assert "incomplete" not in results[-1].stdout
    collected = json.loads((workspace / "security-findings.json").read_text(encoding="utf-8"))
    assert collected["summary"]["scans_completed"] == list(ARRIVES)
    for job_key, arrives in ARRIVES.items():
        assert {_arrived(f) for f in collected["jobs"][job_key]["findings"]} == arrives
    assert collected["summary"]["overall_status"] == "HIGH"

    sent = [path for _, path in uploaded] + [workspace / path for path in sorted(SUMMARY_OUTPUTS)]
    assert [path.name for path in sent if MARKER in path.read_text(encoding="utf-8")] == []
    assert [text for text in run.printed if MARKER in text] == []


@pytest.mark.parametrize("tool", list(TOOLS))
def test_a_tool_that_never_ran_leaves_the_run_incomplete(complete, tool):
    run, done = complete
    (name,) = [name for name in SCAN_JOBS if any(t == tool for t, _ in _scanner_lines(run.jobs[name]))]
    workspace, result = run.scan_job(name, absent=(tool,))
    key, tool_key = SCAN_JOBS[name], tool.lstrip("./")
    findings = json.loads((workspace / f"{key}-findings.json").read_text(encoding="utf-8"))
    assert findings["tools"][tool_key]["status"] != "read"
    assert result.returncode == 1

    uploaded = [item for other, (ws, _) in done.items() if other != name for item in run.uploads(other, ws)]
    summary, results = run.summary(uploaded + run.uploads(name, workspace))
    collected = json.loads((summary / "security-findings.json").read_text(encoding="utf-8"))
    assert (collected["summary"]["overall_status"], collected["summary"]["scans_failed"]) == ("INCOMPLETE", [key])
    assert results[0].returncode == 1
    assert results[-1].returncode != 0
    assert f"results incomplete: {tool_key}" in results[-1].stdout


#: A tool that stops on a fatal error, with the report it still leaves and its exit code, as its source has them.
FATAL = {
    # cli/src/semgrep/output.py saves the report before _final_raise; error.py: FATAL_EXIT_CODE = 2.
    "semgrep": (json.dumps({"results": [], "errors": [{"code": 2, "level": "error", "type": "SemgrepError"}]}), 2),
    # src/Hadolint/Formatter/Json.hs renders a Dockerfile it cannot parse as DL1000; app/Main.hs exits 1.
    "./hadolint": (
        json.dumps([{"code": "DL1000", "column": 1, "file": "Dockerfile", "level": "error", "line": 1, "message": ""}]),
        1,
    ),
    # The control: bandit's -o is argparse.FileType("w"), created empty before a fatal exit 2, which the boundary
    # already could not read.
    "bandit": ("", 2),
}


@pytest.mark.parametrize("tool", list(FATAL))
def test_a_tool_that_stops_on_a_fatal_error_leaves_the_run_incomplete(complete, tool):
    run, done = complete
    (name,) = [name for name in SCAN_JOBS if any(t == tool for t, _ in _scanner_lines(run.jobs[name]))]
    text, code = FATAL[tool]
    workspace, result = run.scan_job(name, fatal=(tool, text, code))
    key, tool_key = SCAN_JOBS[name], tool.lstrip("./")
    findings = json.loads((workspace / f"{key}-findings.json").read_text(encoding="utf-8"))
    assert findings["tools"][tool_key]["status"] != "read"
    assert result.returncode == 1

    uploaded = [item for other, (ws, _) in done.items() if other != name for item in run.uploads(other, ws)]
    summary, results = run.summary(uploaded + run.uploads(name, workspace))
    collected = json.loads((summary / "security-findings.json").read_text(encoding="utf-8"))
    assert (collected["summary"]["overall_status"], collected["summary"]["scans_failed"]) == ("INCOMPLETE", [key])
    assert f"results incomplete: {tool_key}" in results[-1].stdout


def test_a_summary_over_an_empty_download_is_incomplete(complete):
    run, _ = complete
    summary, results = run.summary([])
    collected = json.loads((summary / "security-findings.json").read_text(encoding="utf-8"))
    assert collected["summary"]["overall_status"] == "INCOMPLETE"
    assert collected["summary"]["scans_failed"] == list(ARRIVES)
    assert results[0].returncode == 1
    assert results[-1].returncode != 0
    assert results[-1].stdout.count("results not found") == 6


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
    (tmp_path / "Dockerfile").write_bytes(b"FROM python:3.12\nENV NAME=caf\xe9\n")
    container.dockerfile_path = tmp_path / "Dockerfile"
    unread = container.check_dockerfile_security()
    assert [f["issue"] for f in unread] == ["Failed to read Dockerfile: UnicodeDecodeError"]

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "service_config.py").write_text(f'\n\napi_key = "{MARKER}"\n', encoding="utf-8")
    checks = _load("custom_security_checks.py").CustomSecurityChecker()
    checks.package_root = tmp_path / "pkg"
    checks.check_configuration_security()
    assert {(f["type"], f["line"]) for f in checks.findings} == {("config_security", 3)}

    assert MARKER not in json.dumps(dockerfile) + json.dumps(checks.findings)


def test_the_base_image_arm_reads_trivy_vulnerabilities_and_hadolint_only(tmp_path, monkeypatch):
    """The deleted parser fed this arm trivy's vulnerabilities and hadolint; trivy's secrets, which name a file, are
    not base-image issues however their path reads."""
    secrets = [
        {"tool": "trivy", "rule": "generic-secret", "file": f"usr/lib/pkg{i}/database.py", "line": 1, "severity": "low"}
        for i in range(6)
    ]
    jobs = {
        job: {"status": "read", "tools": {}, "findings": secrets if job == "container" else []}
        for job in ("dependency", "static", "secrets", "container", "license")
    }
    for job, data in jobs.items():
        data["tools"] = {tool: {"status": "read", "findings": 0} for tool in _load("security_findings.py").JOBS[job]}
    (tmp_path / "security-findings.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    summary = _load("check_critical_issues.py").CriticalIssuesChecker().check_all_results()
    assert summary["blocking_issues"] == []
    assert summary["critical_issues"] == []
