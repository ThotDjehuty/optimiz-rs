"""Bounded readers for administrator-managed datasets and bridge JSONL output."""

import csv
import hashlib
import json
import os
import re
import threading
from collections import deque
from pathlib import Path

from optimizr.analytics import MAX_ROWS, normalize_record, risk_metrics as risk_metrics

MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 4096
DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.csv")


class DatasetStore:
    def __init__(self, directory: Path, max_rows=MAX_ROWS):
        self.directory = directory.resolve()
        self.max_rows = max_rows

    def path(self, dataset_id):
        if not DATASET_ID.fullmatch(dataset_id):
            raise ValueError("Invalid dataset identifier")
        for candidate in self.directory.iterdir():
            if candidate.name == dataset_id:
                return self._checked_path(candidate)
        raise FileNotFoundError("Dataset not found")

    def _checked_path(self, candidate):
        if not DATASET_ID.fullmatch(candidate.name):
            raise ValueError("Invalid dataset identifier")
        if candidate.is_symlink() or candidate.resolve().parent != self.directory:
            raise ValueError("Dataset symlinks are not permitted")
        if not candidate.is_file():
            raise FileNotFoundError("Dataset not found")
        if candidate.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Dataset exceeds 16 MiB")
        return candidate

    def list(self):
        result = []
        if self.directory.exists():
            for path in self.directory.iterdir():
                try:
                    self._checked_path(path)
                except (ValueError, OSError):
                    continue
                result.append(path)
        return [
            path.name
            for path in sorted(result, key=lambda path: path.stat().st_mtime, reverse=True)[:100]
        ]

    def report(self, dataset_id):
        csv_path = self.path(dataset_id)
        report_path = csv_path.with_name(csv_path.name + ".report.json")
        if report_path.is_symlink() or report_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Invalid or oversized risk report")
        report = json.loads(report_path.read_bytes())
        if hashlib.sha256(csv_path.read_bytes()).hexdigest() != report["csv_sha256"]:
            raise ValueError("Risk report does not match the dataset")
        return report

    def load(self, dataset_id):
        with self.path(dataset_id).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not {"event_time", "symbol", "price"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV requires event_time, symbol, price columns")
            if len(reader.fieldnames) > 64:
                raise ValueError("Too many CSV columns")
            rows = []
            for row in reader:
                if len(rows) >= self.max_rows:
                    return rows, True
                rows.append(normalize_record(row))
            return rows, False


class LiveTail:
    """One incremental, rotation-aware reader per process; no vendor transport."""

    def __init__(self, path: Path, max_rows=MAX_ROWS):
        self.path = path
        self.rows = deque(maxlen=max_rows)
        self.sequence = 0
        self.identity = None
        self.offset = 0
        self.pending = b""
        self.discarding = False
        self.rejected = 0
        self.lock = threading.Lock()

    def poll(self):
        accepted = 0
        try:
            with self.path.open("rb") as source:
                stat = os.fstat(source.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if identity != self.identity or stat.st_size < self.offset:
                    self.identity = identity
                    self.offset = max(0, stat.st_size - MAX_FILE_BYTES)
                    self.pending = b""
                    self.discarding = self.offset > 0
                source.seek(self.offset)
                chunk = source.read(256 * 1024)
                self.offset = source.tell()
        except FileNotFoundError:
            return 0
        for part in chunk.splitlines(keepends=True):
            complete = part.endswith(b"\n")
            if self.discarding:
                if complete:
                    self.discarding = False
                continue
            self.pending += part
            if len(self.pending) > MAX_LINE_BYTES:
                self.pending = b""
                self.discarding = not complete
                self.rejected += 1
                continue
            if complete:
                try:
                    row = normalize_record(json.loads(self.pending))
                except (ValueError, TypeError, UnicodeDecodeError, RecursionError):
                    self.rejected += 1
                else:
                    with self.lock:
                        self.sequence += 1
                        self.rows.append((self.sequence, row))
                    accepted += 1
                self.pending = b""
        return accepted

    def since(self, cursor):
        with self.lock:
            snapshot = cursor is None or (self.rows and cursor < self.rows[0][0] - 1)
            rows = [row for seq, row in self.rows if snapshot or seq > cursor]
            return {"type": "snapshot" if snapshot else "update", "rows": rows}, self.sequence
