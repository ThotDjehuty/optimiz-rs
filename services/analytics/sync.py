"""Pull completed Databricks snapshots into the dashboard's local data volume."""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path, PurePosixPath

from optimizr.analytics import normalize_record
from optimizr.snapshot import MAX_SNAPSHOT_BYTES, encode, verify_snapshot

from .data import DatasetStore

LOG = logging.getLogger(__name__)
MANIFEST_NAME = re.compile(r"analytics-([0-9a-f]{32})\.manifest\.json")


def download(client, path, limit):
    response = client.files.download(path)
    if response.contents is None:
        raise RuntimeError("Databricks returned no file content")
    with response.contents as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Databricks export exceeds its size limit")
    return content


def import_snapshot(directory, manifest, content):
    snapshot = verify_snapshot(manifest, content)
    name = "databricks-" + manifest["file"].split(".")[0][len("analytics-") :] + ".csv"
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text, fieldnames=["event_time", "symbol", "price", "source", "price_kind"]
    )
    writer.writeheader()
    writer.writerows(normalize_record(row) for row in snapshot["rows"])
    csv_bytes = text.getvalue().encode()
    if len(csv_bytes) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Exported CSV exceeds 16 MiB")
    report = {
        key: snapshot[key] for key in ("created_at", "source_table", "optimizer_version", "report")
    }
    report["csv_sha256"] = hashlib.sha256(csv_bytes).hexdigest()
    report["snapshot_sha256"] = manifest["sha256"]
    directory.mkdir(parents=True, exist_ok=True)
    # New immutable names: publish the CSV last, so discovery implies a complete pair.
    with tempfile.TemporaryDirectory(prefix=".import-", dir=directory) as staging:
        stage = Path(staging)
        (stage / name).write_bytes(csv_bytes)
        (stage / (name + ".report.json")).write_bytes(encode(report))
        for filename in (name + ".report.json", name):
            target = directory / filename
            if target.exists() or target.is_symlink():
                if target.is_symlink() or target.read_bytes() != (stage / filename).read_bytes():
                    raise ValueError("An immutable imported dataset has changed")
            else:
                os.replace(stage / filename, target)
    return name


def sync_once(client, volume, directory):
    if not re.fullmatch(r"/Volumes/[^/]+/[^/]+/[^/]+", volume):
        raise ValueError("Export volume must be /Volumes/catalog/schema/volume")
    directory = Path(directory)
    imported = []
    count = 0
    for entry in client.files.list_directory_contents(volume):
        path = PurePosixPath(entry.path or "")
        match = MANIFEST_NAME.fullmatch(path.name)
        if entry.is_directory or not match:
            continue
        if str(path.parent) != volume:
            raise ValueError("Databricks returned a manifest outside the export volume")
        count += 1
        if count > 1000:
            raise ValueError("Export volume exceeds 1000 manifests; apply an archival policy")
        manifest = json.loads(download(client, str(path), 2048))
        if manifest.get("file") != f"analytics-{match[1]}.snapshot.json":
            raise ValueError("Manifest filename does not match its snapshot")
        name = f"databricks-{match[1]}.csv"
        if (directory / name).exists():
            previous = DatasetStore(directory).report(name)
            if previous["snapshot_sha256"] != manifest.get("sha256"):
                raise ValueError("A previously imported manifest has changed")
            continue
        content = download(client, str(path.parent / manifest["file"]), MAX_SNAPSHOT_BYTES)
        imported.append(import_snapshot(directory, manifest, content))
        LOG.info("Imported %s", name)
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-volume", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("DATA_DIR", "/data")))
    parser.add_argument("--interval-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.interval_seconds != 0 and args.interval_seconds < 30:
        parser.error("interval-seconds must be zero (once) or at least 30")
    logging.basicConfig(level=logging.INFO)
    from databricks.sdk import WorkspaceClient

    client = WorkspaceClient()
    while True:
        imported = sync_once(client, args.export_volume, args.data_dir / "datasets")
        LOG.info("Sync completed: %d new snapshots", len(imported))
        if args.interval_seconds == 0:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
