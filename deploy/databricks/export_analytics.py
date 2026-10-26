# Databricks notebook source
"""Bounded snapshot export, not a streaming bridge or automatic GCS transfer."""

import importlib.metadata
import json
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from optimizr import compute_risk_metrics_py

if not callable(compute_risk_metrics_py):
    raise RuntimeError("The installed optimizer wheel must include its compiled risk binding")

dbutils.widgets.text("input_table", "")
dbutils.widgets.text("output_volume", "")
dbutils.widgets.text("max_rows", "10000")

table = dbutils.widgets.get("input_table")
volume = dbutils.widgets.get("output_volume")
max_rows = int(dbutils.widgets.get("max_rows"))
if not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*\.[A-Za-z_]\w*", table):
    raise ValueError("input_table must be an unquoted catalog.schema.table")
if not re.fullmatch(r"/Volumes/[^/]+/[^/]+/[^/]+", volume):
    raise ValueError("output_volume must reference an existing UC volume")
if not 1 <= max_rows <= 10000:
    raise ValueError("max_rows must be between 1 and 10000")

# Source timestamps must represent UTC; normalize Spark/Pandas output for the
# service's event_time ISO-8601-with-timezone ingestion contract.
spark.conf.set("spark.sql.session.timeZone", "UTC")
frame = (
    spark.table(table)
    .select("event_time", "symbol", "price")
    .orderBy("event_time", "symbol", "price")
    .limit(max_rows)
    .toPandas()
)
if frame.empty:
    raise ValueError("Input table has no rows")
prices = np.asarray(frame["price"], dtype=float)
if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
    raise ValueError("All selected prices must be finite positive numeric values")
timestamps = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
if timestamps.isna().any():
    raise ValueError("event_time must not be null")
if not all(isinstance(s, str) and 0 < len(s.strip()) <= 512 for s in frame["symbol"]):
    raise ValueError("symbol must be a nonempty string of at most 512 characters")
frame["event_time"] = timestamps.map(lambda value: value.isoformat())
frame["symbol"] = frame["symbol"].str.strip()
frame["source"] = "databricks"

# A unique complete export is published by the job; operators must transfer it
# into the VM /data/datasets directory only after this run succeeds.
target = Path(volume) / f"analytics-{uuid.uuid4().hex}.csv"
frame.to_csv(target, index=False)
dbutils.notebook.exit(
    json.dumps(
        {
            "csv_path": str(target),
            "rows": len(frame),
            "optimizer_version": importlib.metadata.version("optimiz-rs"),
            "transfer_required": "Copy the completed CSV to analytics /data/datasets using an approved pipeline.",
        }
    )
)
