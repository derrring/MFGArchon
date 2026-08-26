#!/usr/bin/env python3
"""
Manual script to check and update ruff version across the repository.

Usage:
    python scripts/update_ruff_version.py --check    # Check for updates only
    python scripts/update_ruff_version.py --update   # Check and apply updates
    python scripts/update_ruff_version.py --force VERSION  # Force specific version

Examples:
    python scripts/update_ruff_version.py --check
    python scripts/update_ruff_version.py --update
    python scripts/update_ruff_version.py --force 0.14.0
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ONE OWNER for "where the ruff pin is". Three encodings of this lived in this file and disagreed
# pairwise on shapes that are all valid YAML: `\s+` spans a blank line but not a comment, and the
# comment-tolerant form written for #2123 spanned a comment but not a blank line -- so widening one
# reader narrowed it on the other axis, and `get_current_version` would print a version that
# `update_files` then refused to bump. Both readers use this now. The trailing group is the pin's
# own indentation and `rev:` key; the caller appends what it wants to do with the value.
#
# It cannot run past the ruff block: the repeated group matches only blank or comment lines, so any
# other key (`hooks:`, the next `- repo:`) ends the match without a `rev:`.
RUFF_PIN = r"astral-sh/ruff-pre-commit[^\S\n]*\n(?:[^\S\n]*(?:#[^\n]*)?\n)*[^\S\n]*rev:[^\S\n]*"


def get_current_version() -> str:
    """Get current ruff version from .pre-commit-config.yaml."""
    config_path = Path(".pre-commit-config.yaml")

    if not config_path.exists():
        print("❌ Error: .pre-commit-config.yaml not found")
        sys.exit(1)

    content = config_path.read_text()

    # Find ruff version
    match = re.search(RUFF_PIN + r"v([0-9.]+)", content)

    if not match:
        print("❌ Error: Could not find ruff version in .pre-commit-config.yaml")
        sys.exit(1)

    return match.group(1)


def get_latest_version() -> str:
    """Get latest ruff version from GitHub API."""
    try:
        # Imported here, not at module scope: `requests` is in neither pyproject.toml nor
        # environment.yml (control: scipy is in both), and it reaches this environment only through
        # conda and Sphinx. A module-scope import made every consumer of this file need a package
        # nothing declares -- which #2123's own test was the first to notice, by failing collection.
        import requests

        response = requests.get("https://api.github.com/repos/astral-sh/ruff/releases/latest", timeout=10)
        response.raise_for_status()
        version = response.json()["tag_name"].lstrip("v")
        return version
    except Exception as e:
        print(f"❌ Error fetching latest version: {e}")
        sys.exit(1)


def compare_versions(current: str, latest: str) -> str:
    """Compare version strings and return status."""
    current_parts = [int(x) for x in current.split(".")]
    latest_parts = [int(x) for x in latest.split(".")]

    if current_parts < latest_parts:
        return "outdated"
    elif current_parts == latest_parts:
        return "current"
    else:
        return "ahead"


def update_files(new_version: str) -> list[str]:
    """Update ruff version in configuration files. Returns the paths it wrote."""
    files_updated = []

    # Update .pre-commit-config.yaml
    config_path = Path(".pre-commit-config.yaml")
    content = config_path.read_text()
    # `RUFF_PIN` rather than `\s+`: a comment between the `repo:` line and its `rev:` is valid
    # YAML and `\s+` cannot span it, so the substitution silently matched nothing and `main()`
    # printed "No files needed updating" and exited 0. The workflow's `sed` handles that shape
    # correctly, so the two bumpers disagreed on it. (#2123)
    updated = re.sub(f"({RUFF_PIN})v[0-9.]+", rf"\1v{new_version}", content)

    if updated != content:
        config_path.write_text(updated)
        files_updated.append(".pre-commit-config.yaml")

    # Postcondition. NOT "main() only calls this when the versions differ" -- `--force` does not
    # compare, it calls straight through, so `--force 0.16.0` on a config already at 0.16.0
    # legitimately writes nothing. What makes this sound is that it checks the resulting VALUE of
    # the pin, not whether a write happened: already-current passes, matched-nothing raises. Reads
    # the pin back the way ci.yml does.
    # Split on the repo boundary first: a forward search for `rev:` runs into the NEXT block and
    # reports that block's version, which is a misleading error rather than a wrong verdict.
    blocks = re.split(r"\n(?=\s*-\s*repo:)", config_path.read_text())
    ruff_block = next((b for b in blocks if "astral-sh/ruff-pre-commit" in b), "")
    check = re.search(r"rev:[^\S\n]*v([0-9.]+)", ruff_block)
    if check is None or check.group(1) != new_version:
        got = f"v{check.group(1)}" if check else "no `rev:` at all"
        raise RuntimeError(
            f"after asking for v{new_version} the ruff block has {got}; the bump matched nothing. "
            f"Check the shape of the ruff block in {config_path}."
        )

    # #2123: there is no second pin. This used to rewrite a `ruff==` line in
    # `modern_quality.yml`; that line moved out, the file now says "Ruff formatting and linting
    # (covered by ci.yml quick-checks)" and contains `ruff==` zero times, and `ci.yml` holds no pin
    # either -- ci.yml's quick-checks job READS the version out of `.pre-commit-config.yaml` at
    # runtime, in its `RUFF_VERSION=$(grep ...)` line.
    # A bumper that touches more than the one owner is how the owner stops being one.
    return files_updated


def run_formatting() -> bool:
    """Run ruff format on the codebase."""
    try:
        print("\n📝 Running ruff format...")
        result = subprocess.run(["ruff", "format", "mfgarchon/"], capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print("✅ Formatting complete")
            return True
        else:
            print(f"⚠️  Formatting had issues:\n{result.stderr}")
            return False
    except FileNotFoundError:
        print("⚠️  ruff not found in PATH, skipping formatting")
        return False


def main():
    parser = argparse.ArgumentParser(description="Update ruff version across repository")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check for updates only")
    group.add_argument("--update", action="store_true", help="Check and apply updates")
    group.add_argument("--force", metavar="VERSION", help="Force update to specific version")

    args = parser.parse_args()

    print("🔍 Ruff Version Manager\n")

    # Get current version
    current = get_current_version()
    print(f"📌 Current version: v{current}")

    if args.force:
        # Force specific version
        target_version = args.force.lstrip("v")
        print(f"🎯 Forcing update to: v{target_version}")

        files_updated = update_files(target_version)

        if files_updated:
            print(f"\n✅ Updated {len(files_updated)} file(s):")
            for f in files_updated:
                print(f"   - {f}")

            run_formatting()

            print("\n✨ Done! Next steps:")
            print("   1. Review changes: git diff")
            print("   2. Test locally: pytest tests/")
            print("   3. Run pre-commit: pre-commit run --all-files")
            print("   4. Commit changes: git commit -am 'chore: Update ruff to vX.Y.Z'")
        else:
            print("\n⚠️  No files needed updating")

    else:
        # Check for latest version
        latest = get_latest_version()
        print(f"🆕 Latest version:  v{latest}")

        status = compare_versions(current, latest)

        if status == "current":
            print("\n✅ Ruff is up to date!")
            sys.exit(0)
        elif status == "ahead":
            print("\n⚠️  You're ahead of the latest release")
            print("   (Possibly using a pre-release or beta version)")
            sys.exit(0)
        else:
            # Outdated
            print(f"\n🆕 Update available: v{current} → v{latest}")

            if args.check:
                print("\n📋 To update, run:")
                print("   python scripts/update_ruff_version.py --update")
                sys.exit(0)

            if args.update:
                # Fetch release notes
                try:
                    import requests  # lazy, see get_latest_version

                    response = requests.get(
                        f"https://api.github.com/repos/astral-sh/ruff/releases/tags/v{latest}",
                        timeout=10,
                    )
                    if response.ok:
                        print(f"\n📰 Release notes: {response.json()['html_url']}")
                except Exception:
                    pass

                confirm = input("\n❓ Proceed with update? [y/N] ").lower()

                if confirm != "y":
                    print("❌ Update cancelled")
                    sys.exit(0)

                files_updated = update_files(latest)

                if files_updated:
                    print(f"\n✅ Updated {len(files_updated)} file(s):")
                    for f in files_updated:
                        print(f"   - {f}")

                    run_formatting()

                    print("\n✨ Done! Next steps:")
                    print("   1. Review changes: git diff")
                    print("   2. Test locally: pytest tests/")
                    print("   3. Run pre-commit: pre-commit run --all-files")
                    print(f"   4. Commit: git commit -am 'chore: Update ruff to v{latest}'")
                else:
                    print("\n⚠️  No files needed updating")


if __name__ == "__main__":
    main()
