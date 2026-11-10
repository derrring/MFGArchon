"""The custom secrets scan, `.github/scripts/custom_secrets_scan.py`, on throwaway repositories (#2918): what it must
find, and what a clean tree gives.

The values are drawn from a fixed seed and taken from no real file. Each seeded row pins one part of #2918's change,
and goes red when that part alone is undone:
- a suffix no list names: the scan reads every tracked file, not a suffix list;
- a comment marker elsewhere in the file: no match is skipped because the file holds one;
- a value inside a JSON string: either quote may be backslash-escaped;
- a byte that is not UTF-8: the file is still read, and a byte inside a value does not hide it;
- a NUL past the first 8000 bytes: the file is text, by git's own rule;
- an identifier holding a whitelisted word: the whitelist reads the value, not the name.
The reports record no seeded value.

A private-key header is found with or without a word between `BEGIN` and `PRIVATE`, the PKCS#8 form having none
(#2919). The headers are put together at runtime, and no tracked file holds one.
"""

import importlib.util
import json
import random
import re
import string
import subprocess
from pathlib import Path

import pytest

SCANNER = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "custom_secrets_scan.py"
_RNG = random.Random(2918)
_WHITELISTED = ("example", "demo", "test", "placeholder", "dummy", "fake", "sk-test", "pk_test", "_var")


def _value(n=40):
    while True:
        value = "".join(_RNG.choice(string.ascii_letters + string.digits) for _ in range(n))
        if not any(word in value.lower() for word in _WHITELISTED):
            return value


VALUES = {name: _value() for name in ("suffix", "marker", "json", "encoding", "late NUL", "identifier")}

#: row -> (file, its bytes). Each must be flagged as credential_assignment.
SEEDED = {
    "a suffix no list names": ("notes.bak-local", f'client_secret = "{VALUES["suffix"]}"\n'.encode()),
    "a comment marker elsewhere in the file": (
        "settings.py",
        f'# settings for the service\nclient_secret = "{VALUES["marker"]}"\n'.encode(),
    ),
    "a value inside a JSON string": ("env.json", f'{{"env": "CLIENT_SECRET=\\"{VALUES["json"]}\\""}}\n'.encode()),
    "a byte that is not UTF-8": (
        "deploy.sh",
        b"caf\xe9\n"
        + f'export CLIENT_SECRET="{VALUES["encoding"][:20]}'.encode()
        + b"\xe9"
        + f'{VALUES["encoding"][20:]}"\n'.encode(),
    ),
    "a NUL past the first 8000 bytes": (
        "run.cfg",
        f'client_secret = "{VALUES["late NUL"]}"\n'.encode() + b"x" * 8100 + b"\0\n",
    ),
    "an identifier holding a whitelisted word": ("conf.toml", f'SECRET_LATEST = "{VALUES["identifier"]}"\n'.encode()),
}


def _repo(root, files):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    subprocess.run(["git", "-C", str(root), "add", "-f", "--", *files], check=True)
    return root


def _scan(root):
    spec = importlib.util.spec_from_file_location("custom_secrets_scan", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scanner = module.CustomSecretsScanner(str(root))
    scanner.scan_directory()
    return scanner


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    return _scan(_repo(tmp_path_factory.mktemp("seeded"), dict(SEEDED.values())))


@pytest.mark.parametrize("row", list(SEEDED))
def test_a_seeded_value_is_found(seeded, row):
    name, _ = SEEDED[row]
    kinds = {(f["type"], f["severity"]) for f in seeded.findings if f["file"] == name}
    assert ("credential_assignment", "HIGH") in kinds


def test_the_reports_record_no_seeded_value(seeded):
    report = seeded.generate_report()
    text = json.dumps(report) + seeded.generate_markdown_report(report)
    assert seeded.findings
    assert not [row for row, value in zip(SEEDED, VALUES.values(), strict=True) if value in text]


def test_a_clean_tree_comes_back_clean(tmp_path, capsys):
    files = {
        "README.md": b"# A project\n\nSee https://example.org and // a comment.\n",
        "tool.py": b"session_token = make_session().token\n",
        "short.toml": f'client_secret = "{_value(31)}"\n'.encode(),
        "notes.bak-local": b"plain notes\n",
        "image.bin": b"\x89PNG\x00\x01\x02" * 4,
    }
    scanner = _scan(_repo(tmp_path, files))
    assert scanner.findings == []
    assert "scanned as text: 4; binary (a NUL in the first 8000 bytes): 1" in capsys.readouterr().out


def _key_header(*kind):
    """A private-key header, put together here so that no tracked file holds one (#2919)."""
    return " ".join(["-----BEGIN", *kind, "PRIVATE", "KEY-----"])


@pytest.mark.parametrize("kind", [(), ("RSA",), ("ENCRYPTED",)], ids=["PKCS#8", "RSA", "encrypted PKCS#8"])
def test_a_private_key_header_is_found(tmp_path, kind):
    rng = random.Random(2919)
    body = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(64))
    header = _key_header(*kind)
    pem = f"{header}\n{body}\n{header.replace('BEGIN', 'END')}\n"
    scanner = _scan(_repo(tmp_path, {"id.pem": pem.encode()}))
    assert [(f["type"], f["severity"], f["line"]) for f in scanner.findings if f["type"] == "private_key"] == [
        ("private_key", "HIGH", 1)
    ]


def test_no_tracked_file_holds_a_private_key_header():
    spec = importlib.util.spec_from_file_location("custom_secrets_scan", SCANNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (pattern,) = module.CustomSecretsScanner.SECRET_PATTERNS["private_key"]
    repo = SCANNER.parents[2]
    listed = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True, check=True).stdout
    names = [name.decode() for name in listed.split(b"\0") if name]
    assert "tests/unit/test_custom_secrets_scan.py" in names
    held = [
        name
        for name in names
        if (repo / name).is_file() and re.search(pattern, (repo / name).read_bytes().decode("utf-8", errors="ignore"))
    ]
    assert held == []
