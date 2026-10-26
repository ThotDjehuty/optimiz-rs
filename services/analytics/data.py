"""Bounded readers for administrator-managed datasets and bridge JSONL output."""

import csv
import json
import math
import os
import re
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

MAX_ROWS = 10_000
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_LINE_BYTES = 4096
DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.csv")


def normalize_record(record):
    if not isinstance(record, dict):
        raise TypeError("Expected an object")
    symbol = record.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 512:
        raise ValueError("symbol must be a nonempty string of at most 512 characters")
    value = record.get("price")
    if isinstance(value, bool):
        raise TypeError("price must be numeric")
    try:
        price = float(value)
        timestamp = datetime.fromisoformat(str(record["event_time"]).replace("Z", "+00:00"))
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise ValueError("Expected numeric price and ISO-8601 event_time") from exc
    if not math.isfinite(price) or price <= 0 or timestamp.tzinfo is None:
        raise ValueError("price must be finite and positive; event_time must include timezone")
    try:
        utc_time = timestamp.astimezone(timezone.utc).isoformat()
    except OverflowError as exc:
        raise ValueError("event_time is outside the supported range") from exc
    normalized = {
        "event_time": utc_time,
        "symbol": symbol.strip(),
        "price": price,
    }
    for field, limit, default in (("source", 128, "unknown"), ("price_kind", 64, "unspecified")):
        value = record.get(field, default)
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"{field} must be a string of at most {limit} characters")
        normalized[field] = value or default
    return normalized


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
                if len(result) >= 100:
                    break
                try:
                    self._checked_path(path)
                except (ValueError, OSError):
                    continue
                result.append(path.name)
        return sorted(result)

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


def risk_metrics(rows, symbol, periods_per_year=252.0):
    """Price-to-return conversion only; all risk mathematics belongs to OptimizR."""
    import numpy as np
    from optimizr import compute_risk_metrics_py

    if not callable(compute_risk_metrics_py):
        raise TypeError("The compiled OptimizR risk binding is required")
    selected = sorted(
        (row for row in rows if row["symbol"] == symbol), key=lambda row: row["event_time"]
    )
    if len(selected) < 5:
        raise ValueError("At least five prices for one symbol are required")
    prices = np.asarray([row["price"] for row in selected], dtype=np.float64)
    returns = np.diff(prices) / prices[:-1]
    if not np.all(np.isfinite(returns)):
        raise ValueError("Price changes exceed finite numeric limits")
    result = compute_risk_metrics_py(returns, 0.0, periods_per_year)
    return {
        "symbol": symbol,
        "observations": len(returns),
        "periods_per_year": periods_per_year,
        "risk_free_rate": 0.0,
        "metrics": {
            key: value if math.isfinite(float(value)) else None for key, value in result.items()
        },
    }
