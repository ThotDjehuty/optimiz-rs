import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_vendor", ROOT / "scripts" / "verify_vendor.py"
)
vendor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vendor)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    lock = json.loads((ROOT / "vendor-lock.json").read_text())
    (tmp_path / "vendor-lock.json").write_text(json.dumps(lock))

    def fake_git(root, *args):
        if args[0] == "config":
            return lock["repository"] if args[-1].endswith(".url") else lock["path"]
        if args[0] == "ls-files":
            return f"160000 {lock['commit']} 0\t{lock['path']}"
        raise AssertionError(args)

    monkeypatch.setattr(vendor, "git", fake_git)
    return tmp_path, lock


def test_pinned_metadata(checkout):
    root, lock = checkout
    assert vendor.verify(root, metadata_only=True) == lock["commit"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("repository", "https://example.com/unapproved.git"),
        ("commit", "main"),
        ("path", "../other"),
        ("required_directories", ["MLink"]),
    ],
)
def test_rejects_unapproved_metadata(checkout, key, value):
    root, lock = checkout
    lock[key] = value
    (root / "vendor-lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        vendor.verify(root, metadata_only=True)


def test_release_requires_initialized_checkout(checkout):
    root, _ = checkout
    with pytest.raises(ValueError, match="Initialize"):
        vendor.verify(root)


def test_gitlink_must_match_lock(checkout, monkeypatch):
    root, _ = checkout
    original = vendor.git

    def wrong_gitlink(root, *args):
        return (
            "160000 wrong 0\tvendor/spiderrock-connect-sdk"
            if args[0] == "ls-files"
            else original(root, *args)
        )

    monkeypatch.setattr(vendor, "git", wrong_gitlink)
    with pytest.raises(ValueError, match="Gitlink"):
        vendor.verify(root, metadata_only=True)


@pytest.mark.parametrize(
    "head,dirty,nested,message",
    [
        ("0" * 40, "", "", "pinned revision"),
        (None, " M MLink/client.py", "", "local changes"),
        (None, "", "-deadbeef nested", "Nested"),
        (None, "", "+deadbeef nested", "Nested"),
        (None, "", "Udeadbeef nested", "Nested"),
    ],
)
def test_release_rejects_drift(checkout, monkeypatch, head, dirty, nested, message):
    root, lock = checkout
    sdk = root / lock["path"]
    sdk.mkdir(parents=True)
    (sdk / ".git").touch()
    original = vendor.git

    def fake_git(path, *args):
        if args[0] == "rev-parse":
            return head or lock["commit"]
        if args[0] == "status":
            return dirty
        return original(path, *args)

    monkeypatch.setattr(vendor, "git", fake_git)
    monkeypatch.setattr(
        vendor.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=nested)
    )
    with pytest.raises(ValueError, match=message):
        vendor.verify(root)


def test_release_requires_all_interface_directories(checkout, monkeypatch):
    root, lock = checkout
    sdk = root / lock["path"]
    sdk.mkdir(parents=True)
    (sdk / ".git").touch()
    original = vendor.git

    def fake_git(path, *args):
        if args[0] == "rev-parse":
            return lock["commit"]
        if args[0] == "status":
            return ""
        return original(path, *args)

    monkeypatch.setattr(vendor, "git", fake_git)
    monkeypatch.setattr(
        vendor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" abcdef nested\n"),
    )
    with pytest.raises(ValueError, match="Missing vendor"):
        vendor.verify(root)
    for name in vendor.FAMILIES:
        (sdk / name).mkdir()
    assert vendor.verify(root) == lock["commit"]
