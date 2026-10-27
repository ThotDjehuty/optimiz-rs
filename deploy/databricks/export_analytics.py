# Databricks notebook source
"""Compute bounded per-symbol risk and publish a verified-transfer manifest."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from optimizr.snapshot import export_snapshot
from optimizr.analytics import normalize_sr4p_quote

dbutils.widgets.text("input_table", "")
dbutils.widgets.text("output_volume", "")
dbutils.widgets.text("max_rows", "10000")
dbutils.widgets.text("periods_per_year", "")
dbutils.widgets.text("input_format", "normalized")
dbutils.widgets.text("session_id", "")
dbutils.widgets.text("ticker", "")

table = dbutils.widgets.get("input_table")
volume = dbutils.widgets.get("output_volume")
max_rows = int(dbutils.widgets.get("max_rows"))
periods_per_year = float(dbutils.widgets.get("periods_per_year"))
input_format = dbutils.widgets.get("input_format")
if not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*\.[A-Za-z_]\w*", table):
    raise ValueError("input_table must be an unquoted catalog.schema.table")
if not re.fullmatch(r"/Volumes/[^/]+/[^/]+/[^/]+", volume):
    raise ValueError("output_volume must reference an existing UC volume")
if not 1 <= max_rows <= 10000:
    raise ValueError("max_rows must be between 1 and 10000")

# Source timestamps must represent UTC; normalize Spark/Pandas output for the
# service's event_time ISO-8601-with-timezone ingestion contract.
spark.conf.set("spark.sql.session.timeZone", "UTC")
source = spark.table(table)
if input_format == "sr4p_quotes":
    from pyspark.sql import functions as F

    session_id = dbutils.widgets.get("session_id")
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[A-Za-z0-9]+", session_id):
        raise ValueError("An explicit completed SR4P session_id is required")
    source = source.filter(F.col("session_id") == session_id)
    ticker = dbutils.widgets.get("ticker")
    if ticker:
        source = source.filter(F.col("okey_tk") == ticker)
    columns = [
        "session_id",
        "okey_at",
        "okey_cp",
        "okey_dt",
        "okey_tk",
        "okey_ts",
        "okey_xx",
        "obid",
        "oask",
        "timestamp",
    ]
    records = source.select(*columns).limit(max_rows + 1).toPandas().to_dict(orient="records")
    if len(records) > max_rows:
        raise ValueError(
            "Session selection exceeds max_rows; narrow the ticker or supply a bounded view"
        )
    rows = [normalize_sr4p_quote(record, session_id) for record in records]
    target = export_snapshot(Path(volume), rows, periods_per_year, table)
    dbutils.notebook.exit(json.dumps({"manifest_path": str(target), "rows": len(rows)}))
elif input_format != "normalized":
    raise ValueError("input_format must be normalized or sr4p_quotes")

columns = ["event_time", "symbol", "price"]
columns += [name for name in ("source", "price_kind") if name in source.columns]
frame = (
    source.select(*columns).orderBy("event_time", "symbol", "price").limit(max_rows + 1).toPandas()
)
if frame.empty:
    raise ValueError("Input table has no rows")
if len(frame) > max_rows:
    raise ValueError(
        "Input exceeds max_rows; supply a bounded research view, not a truncated history"
    )
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
if "source" not in frame:
    frame["source"] = "databricks"

target = export_snapshot(Path(volume), frame.to_dict(orient="records"), periods_per_year, table)
dbutils.notebook.exit(
    json.dumps(
        {
            "manifest_path": str(target),
            "rows": len(frame),
            "transfer": "The analytics sync service imports completed manifests from this volume.",
        }
    )
)
