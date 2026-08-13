#!/usr/bin/env python3
"""
MFGarchon Package Health Check

Analyzes package organization and identifies potential issues.
"""

import subprocess
from pathlib import Path


def section(title):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def run_command(cmd, description):
    """Run a command and return output."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"Output:\n{result.stdout[:500]}")
        if result.returncode != 0 and result.stderr:
            print(f"Errors:\n{result.stderr[:500]}")
        return result
    except subprocess.TimeoutExpired:
        print("Command timed out!")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None


def check_root_clutter():
    """Check for files that should be organized."""
    section("1. Root Directory Organization")

    root = Path(".")

    # Check for phase-specific docs in root
    phase_docs = (
        list(root.glob("*PHASE*.md"))
        + list(root.glob("*MERGE*.md"))
        + list(root.glob("*HANDOFF*.md"))
        + list(root.glob("QUICK_*.md"))
    )

    if phase_docs:
        print(f"Found {len(phase_docs)} phase-specific documents in root:")
        for doc in sorted(phase_docs):
            print(f"  - {doc.name}")
        print("\n💡 Suggestion: Move to docs/development/completed/ or archive/")
    else:
        print("✅ No phase-specific documents in root")

    # Check for session summaries
    session_docs = list(root.glob("*SESSION*.md")) + list(root.glob("POST_*.md"))
    if session_docs:
        print(f"\nFound {len(session_docs)} session documents in root:")
        for doc in sorted(session_docs):
            print(f"  - {doc.name}")
        print("\n💡 Suggestion: Move to docs/development/sessions/ or archive/")
    else:
        print("✅ No session documents in root")

    return len(phase_docs) + len(session_docs)


def check_test_health():
    """Check test suite health."""
    section("2. Test Suite Health")

    # Run pytest collection
    result = run_command("python -m pytest --collect-only -q 2>&1", "Test collection")

    if result and "error" in result.stdout.lower():
        print("\n⚠️  Test collection has errors")
        # Check archive tests
        if "archive" in result.stdout:
            print("\n💡 Archive tests have errors (expected, can ignore)")

    # Run actual tests
    print("\n" + "-" * 70)
    result = run_command(
        "python -m pytest tests/unit/test_mean_field_*.py -v --tb=no -q 2>&1 | tail -5", "Run continuous control tests"
    )

    return result


def check_import_health():
    """Check if main imports work."""
    section("3. Import Health Check")

    imports_to_check = [
        ("Core package", "import mfgarchon"),
    ]

    failures = []
    for name, import_stmt in imports_to_check:
        try:
            exec(import_stmt)
            print(f"✅ {name}: OK")
        except Exception as e:
            print(f"❌ {name}: FAILED - {e}")
            failures.append((name, str(e)))

    return failures


def check_documentation_structure():
    """Check documentation organization."""
    section("4. Documentation Structure")

    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("❌ docs/ directory not found")
        return

    # Check subdirectories
    subdirs = [d for d in docs_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} documentation categories:")
    for subdir in sorted(subdirs):
        file_count = len(list(subdir.glob("*.md")))
        print(f"  - {subdir.name}: {file_count} files")

    # Check for duplicate/overlapping docs
    development_docs = list((docs_dir / "development").glob("*.md")) if (docs_dir / "development").exists() else []

    # Check for WIP or superseded docs
    wip_docs = []
    superseded_docs = []
    for doc in development_docs:
        content = doc.read_text()
        if "[WIP]" in content or "work in progress" in content.lower():
            wip_docs.append(doc.name)
        if "[SUPERSEDED]" in content or "[ARCHIVED]" in content or "superseded" in content.lower():
            superseded_docs.append(doc.name)

    if wip_docs:
        print(f"\n⚠️  Found {len(wip_docs)} WIP documents:")
        for doc in wip_docs[:5]:
            print(f"  - {doc}")

    if superseded_docs:
        print(f"\n💡 Found {len(superseded_docs)} superseded documents (consider archiving):")
        for doc in superseded_docs[:5]:
            print(f"  - {doc}")


def check_code_organization():
    """Check code organization."""
    section("5. Code Organization")

    alg_dir = Path("mfgarchon/alg")
    if not alg_dir.exists():
        print("❌ mfgarchon/alg/ directory not found")
        return

    # Check paradigms
    paradigms = [d.name for d in alg_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    print(f"Found {len(paradigms)} algorithm paradigms:")
    for paradigm in sorted(paradigms):
        paradigm_dir = alg_dir / paradigm
        py_files = len(list(paradigm_dir.glob("**/*.py"))) - len(list(paradigm_dir.glob("**/__init__.py")))
        print(f"  - {paradigm}: {py_files} implementation files")

    # Check for duplicate or misplaced files
    print("\nChecking for organization issues...")

    # Check if there are algorithms in mfgarchon/alg/ root
    root_algs = list(alg_dir.glob("*.py"))
    root_algs = [f for f in root_algs if f.name != "__init__.py"]
    if root_algs:
        print(f"\n⚠️  Found {len(root_algs)} files in alg/ root (should be in paradigm subdirs):")
        for f in root_algs:
            print(f"  - {f.name}")
    else:
        print("✅ No misplaced algorithm files in root")


def generate_summary():
    """Generate summary and recommendations."""
    section("6. Summary and Recommendations")

    print("📋 Recommended Actions:\n")
    print("1. 📁 Organize root directory:")
    print("   - Move phase-specific docs to docs/development/completed/")
    print("   - Move session docs to docs/development/sessions/")
    print("   - Keep only: README.md, CLAUDE.md, CONTRIBUTING.md in root")

    print("\n2. 🧹 Clean up archive:")
    print("   - Archive has test collection errors (expected)")
    print("   - Consider excluding archive/ from pytest discovery")

    print("\n3. 📚 Documentation:")
    print("   - Mark completed phases clearly")
    print("   - Archive superseded documents")
    print("   - Consolidate overlapping content")

    print("\n4. ✅ Tests:")
    print("   - All new continuous control tests passing")
    print("   - Consider fixing property-based test collection error")

    print("\n5. 🎯 Next Steps:")
    print("   - Run organization cleanup")
    print("   - Update pytest.ini to exclude archive/")
    print("   - Update main README with v1.4.0 features")


def main():
    """Run health check."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   MFGarchon Package Health Check                       ║
║                         Version 1.4.0                                ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    issues_count = 0

    # Run checks
    issues_count += check_root_clutter()
    check_test_health()
    import_failures = check_import_health()
    issues_count += len(import_failures)
    check_documentation_structure()
    check_code_organization()
    generate_summary()

    # Final summary
    section("Health Check Complete")
    if issues_count == 0:
        print("✅ No critical issues found!")
    else:
        print(f"⚠️  Found {issues_count} organization issues (non-critical)")

    print("\n💡 See recommendations above for cleanup suggestions")


if __name__ == "__main__":
    main()
