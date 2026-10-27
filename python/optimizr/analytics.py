"""Shared, bounded price-to-risk contract for Databricks and the dashboard."""

import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

MAX_ROWS = 10_000


def normalize_sr4p_quote(record: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Map the existing silver_live_implied_quote_adj schema, not an invented feed."""
    if not session_id or record.get("session_id") != session_id:
        raise ValueError("Quote must belong to the explicitly selected research session")
    key = {}
    for name in ("okey_at", "okey_cp", "okey_dt", "okey_tk", "okey_ts"):
        value = record.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing full option key component: {name}")
        key[name] = value
    if any(isinstance(record.get(name), bool) for name in ("okey_xx", "obid", "oask")):
        raise ValueError("Strike and quote prices must be numeric, not boolean")
    strike = float(record["okey_xx"])
    bid, ask = float(record["obid"]), float(record["oask"])
    if not all(math.isfinite(value) and value > 0 for value in (strike, bid, ask)) or bid > ask:
        raise ValueError("Invalid strike or two-sided option quote")
    key["okey_xx"] = str(strike)
    return normalize_record(
        {
            "event_time": record["timestamp"],
            "symbol": json.dumps(key, sort_keys=True, separators=(",", ":")),
            "price": bid / 2 + ask / 2,
            "source": "sr4p:" + session_id,
            "price_kind": "option_midpoint",
        }
    )


def normalize_record(record: Any) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("Expected an object")
    symbol = record.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 512:
        raise ValueError("symbol must be a nonempty string of at most 512 characters")
    value = record.get("price")
    if value is None or isinstance(value, bool):
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
    normalized = {"event_time": utc_time, "symbol": symbol.strip(), "price": price}
    for field, limit, default in (("source", 128, "unknown"), ("price_kind", 64, "unspecified")):
        value = record.get(field, default)
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"{field} must be a string of at most {limit} characters")
        normalized[field] = value or default
    return normalized


def risk_metrics(
    rows: List[Dict[str, Any]], symbol: str, periods_per_year: float = 252.0
) -> Dict[str, Any]:
    import numpy as np
    from optimizr import compute_risk_metrics_py

    if not callable(compute_risk_metrics_py):
        raise TypeError("The compiled OptimizR risk binding is required")
    if not math.isfinite(periods_per_year) or not 1 <= periods_per_year <= 1_000_000:
        raise ValueError("periods_per_year must be finite and between 1 and 1000000")
    selected = sorted(
        (row for row in rows if row["symbol"] == symbol), key=lambda row: row["event_time"]
    )
    if len(selected) < 5:
        raise ValueError("At least five prices for one symbol are required")
    prices: np.ndarray = np.asarray([row["price"] for row in selected], dtype=np.float64)
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


def risk_report(rows: List[Dict[str, Any]], periods_per_year: float) -> Dict[str, Any]:
    """Keep metrics separate; reject ambiguous prices rather than inventing a blend."""
    if not rows or len(rows) > MAX_ROWS:
        raise ValueError("A risk snapshot requires between 1 and 10000 rows")
    if not math.isfinite(periods_per_year) or not 1 <= periods_per_year <= 1_000_000:
        raise ValueError("periods_per_year must be finite and between 1 and 1000000")
    rows = [normalize_record(row) for row in rows]
    counts = Counter(row["symbol"] for row in rows)
    results = []
    errors = []
    for symbol in sorted(counts):
        selected = [row for row in rows if row["symbol"] == symbol]
        if len({row["event_time"] for row in selected}) != len(selected):
            errors.append({"symbol": symbol, "detail": "Duplicate observation timestamps"})
        elif len({row["price_kind"] for row in selected}) != 1:
            errors.append({"symbol": symbol, "detail": "Mixed price kinds"})
        elif counts[symbol] < 5:
            errors.append({"symbol": symbol, "detail": "At least five prices are required"})
        else:
            results.append(risk_metrics(selected, symbol, periods_per_year))
    return {
        "schema_version": 1,
        "aggregation": "none",
        "row_count": len(rows),
        "periods_per_year": periods_per_year,
        "results": results,
        "errors": errors,
    }
