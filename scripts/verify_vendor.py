"""Fail closed when the complete, pinned SpiderRock SDK checkout has drifted."""

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_PATH = "vendor/spiderrock-connect-sdk"
VENDOR_URL = "https://github.com/SpiderRock/SpiderRock-Connect-SDK.git"
FAMILIES = {"Env-Sandbox", "FIX", "HistoricalData", "MLink", "SRSE", "SpiderStream"}


def git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify(root=ROOT, metadata_only=False):
    root = Path(root)
    lock = json.loads((root / "vendor-lock.json").read_text())
    if lock.get("repository") != VENDOR_URL or lock.get("path") != VENDOR_PATH:
        raise ValueError("Only the official SpiderRock SDK source is approved")
    commit = lock.get("commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Vendor revision must be a full commit SHA")
    if set(lock.get("required_directories", [])) != FAMILIES:
        raise ValueError("All vendor interface families must be retained")
    url = git(
        root,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        f"submodule.{VENDOR_PATH}.url",
    )
    path = git(
        root,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        f"submodule.{VENDOR_PATH}.path",
    )
    if url != VENDOR_URL or path != VENDOR_PATH:
        raise ValueError("Submodule configuration differs from the approved source")
    entry = git(root, "ls-files", "--stage", "--", VENDOR_PATH)
    if entry != f"160000 {commit} 0\t{VENDOR_PATH}":
        raise ValueError("Gitlink differs from vendor-lock.json")
    if metadata_only:
        return commit

    vendor = root / VENDOR_PATH
    if vendor.is_symlink() or not (vendor / ".git").exists():
        raise ValueError("Initialize the vendor submodule before validation")
    if git(vendor, "rev-parse", "HEAD") != commit:
        raise ValueError("Vendor checkout differs from the pinned revision")
    if git(vendor, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("Vendor code has local changes; do not patch vendor files")
    # --strip is deliberately not used here: the leading space is Git's
    # indication that a nested submodule matches its recorded revision.
    nested = subprocess.run(
        ["git", "-C", str(vendor), "submodule", "status", "--recursive"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if any(line and not line.startswith(" ") for line in nested.splitlines()):
        raise ValueError("Nested vendor submodules are missing or have drifted")
    for name in sorted(FAMILIES):
        if not (vendor / name).is_dir():
            raise ValueError(f"Missing vendor interface family: {name}")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Check the lock and gitlink only; NOT sufficient for a release build",
    )
    args = parser.parse_args()
    try:
        commit = verify(metadata_only=args.metadata_only)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Vendor integrity check failed: {exc}\n")
    print(f"SpiderRock SDK verified: {commit}")


if __name__ == "__main__":
    main()
