"""Immutable, checksummed calculation exports; no credentials or transport."""

import hashlib
import importlib.metadata
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from optimizr.analytics import normalize_record, risk_report

MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
SNAPSHOT_NAME = re.compile(r"analytics-[0-9a-f]{32}\.snapshot\.json")


def encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def export_snapshot(
    directory: Path,
    rows: List[Dict[str, Any]],
    periods_per_year: float,
    source_table: str,
) -> Path:
    if not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*\.[A-Za-z_]\w*", source_table):
        raise ValueError("source_table must be catalog.schema.table")
    normalized = [normalize_record(row) for row in rows]
    snapshot = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_table": source_table,
        "optimizer_version": importlib.metadata.version("optimiz-rs"),
        "rows": normalized,
        "report": risk_report(normalized, periods_per_year),
    }
    content = encode(snapshot)
    if len(content) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Snapshot exceeds 16 MiB")
    name = f"analytics-{uuid.uuid4().hex}.snapshot.json"
    manifest = {
        "schema_version": 1,
        "file": name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    # The marker is published last. Consumers never discover unfinished snapshots.
    (directory / name).write_bytes(content)
    marker = directory / name.replace(".snapshot.json", ".manifest.json")
    marker.write_bytes(encode(manifest))
    return marker


def verify_snapshot(manifest: Dict[str, Any], content: bytes) -> Dict[str, Any]:
    if (
        manifest.get("schema_version") != 1
        or not isinstance(manifest.get("file"), str)
        or not SNAPSHOT_NAME.fullmatch(manifest["file"])
        or type(manifest.get("bytes")) is not int
        or not 0 < manifest["bytes"] <= MAX_SNAPSHOT_BYTES
        or len(content) != manifest["bytes"]
        or hashlib.sha256(content).hexdigest() != manifest.get("sha256")
    ):
        raise ValueError("Invalid snapshot manifest, size or checksum")
    snapshot = json.loads(content)
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot schema")
    # Re-execute the same compiled contract on the receiver before publication.
    actual = risk_report(snapshot["rows"], snapshot["report"]["periods_per_year"])
    expected = snapshot["report"]
    if actual.keys() != expected.keys():
        raise ValueError("Invalid risk report fields")
    for field in ("schema_version", "aggregation", "row_count", "periods_per_year", "errors"):
        if actual[field] != expected[field]:
            raise ValueError(f"Risk report mismatch: {field}")
    if len(actual["results"]) != len(expected["results"]):
        raise ValueError("Risk report result count mismatch")
    import math

    for left, right in zip(actual["results"], expected["results"]):
        if left.keys() != right.keys() or left["metrics"].keys() != right["metrics"].keys():
            raise ValueError("Risk report metric fields mismatch")
        for key in left:
            if key != "metrics" and left[key] != right[key]:
                raise ValueError(f"Risk report mismatch: {key}")
        for key, value in left["metrics"].items():
            other = right["metrics"][key]
            if value is None or other is None:
                equal = value is other
            else:
                equal = math.isclose(value, other, rel_tol=1e-9, abs_tol=1e-12)
            if not equal:
                raise ValueError(f"Risk report numerical mismatch: {key}")
    return dict(snapshot)
