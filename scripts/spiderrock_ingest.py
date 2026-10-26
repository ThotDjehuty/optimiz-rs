"""Read-only MLink snapshots using the pinned vendor parser, never a custom decoder."""

import argparse
import importlib.util
import io
import json
import math
import os
import re
import signal
import sys
import time
from collections import OrderedDict
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from verify_vendor import ROOT, VENDOR_PATH, verify

ENDPOINTS = {
    "sandbox": "https://mlink-test.nms.saturn.spiderrockconnect.com/rest/proto",
    "live": "https://mlink-live.nms.saturn.spiderrockconnect.com/rest/proto",
}
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024


@contextmanager
def response_deadline(seconds=30):
    """Interrupt stalled HTTP reads in this standalone Linux-container process."""
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("Run the SpiderRock bridge in its Linux container")

    def expired(signum, frame):
        raise TimeoutError("MLink response deadline exceeded")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def load_vendor():
    commit = verify()
    sys.dont_write_bytecode = True
    path = ROOT / VENDOR_PATH / "MLink" / "_python" / "PROTO" / "clientRest.py"
    spec = importlib.util.spec_from_file_location("spiderrock_vendor_rest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, commit


def fetch_snapshot(session, endpoint, api_key, ticker, limit):
    params = {
        "cmd": "getmsgs",
        "msgtype": "LiveImpliedQuoteAdj",
        "where": f"ticker:eq:{ticker}",
        "apikey": api_key,
        "view": "obid|oask|timestamp|srcTimestamp|netTimestamp",
        "l": limit,
    }
    start = time.monotonic()
    with response_deadline(), session.get(
        endpoint,
        params=params,
        headers={"Accept": "application/x-protobuf"},
        timeout=(5, 10),
        allow_redirects=False,
        stream=True,
    ) as response:
        if response.status_code != 200:
            raise ValueError("MLink request failed")
        data = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            data.extend(chunk)
            if len(data) > MAX_RESPONSE_BYTES or time.monotonic() - start > 30:
                raise ValueError("MLink response exceeded configured bounds")
    return bytes(data)


def decode_snapshot(vendor, body, limit):
    # The vendor example prints diagnostics, including parsing failures. Keep
    # them private and reject errors rather than exporting a partial batch.
    diagnostics = io.StringIO()
    with redirect_stdout(diagnostics):
        records, result = vendor.parse_custom_protocol_response(body)
    output = diagnostics.getvalue()
    if any(
        marker in output for marker in ("Failed to parse", "Skipping", "Unexpected", "No _meta")
    ):
        raise ValueError("Vendor decoder rejected a message")
    if result is None:
        raise ValueError("Missing vendor QueryResult")
    success = result.DESCRIPTOR.fields_by_name["result"].enum_type.values_by_name.get("QRESULT_OK")
    if success is None or result.result != success.number:
        raise ValueError("Vendor query was not successful")
    if len(records) > limit:
        raise ValueError("Vendor response exceeded requested record limit")
    return records


def normalize(record, environment, commit):
    from google.protobuf.json_format import MessageToDict

    if not record.HasField("pkey") or not record.HasField("timestamp"):
        raise ValueError("Record is missing its vendor key or timestamp")
    key = MessageToDict(record.pkey, preserving_proto_field_name=True)
    if not key.get("okey"):
        raise ValueError("Record is missing its option key")
    bid, ask = float(record.obid), float(record.oask)
    if not all(math.isfinite(value) and value > 0 for value in (bid, ask)) or ask < bid:
        raise ValueError("Record has no valid two-sided price")
    return {
        # Preserve the complete option key instead of merging expiries/strikes.
        "symbol": json.dumps(key["okey"], sort_keys=True, separators=(",", ":")),
        "event_time": record.timestamp.ToJsonString(),
        "price": (bid + ask) / 2,
        "source": f"spiderrock:{environment}:snapshot",
        "price_kind": "option_midpoint",
        "bid": bid,
        "ask": ask,
        "received_time": datetime.now(timezone.utc).isoformat(),
        "src_timestamp": str(record.src_timestamp),
        "net_timestamp": str(record.net_timestamp),
        "vendor_commit": commit,
    }


def append_records(path, records):
    if not records:
        return
    encoded = "".join(json.dumps(row, allow_nan=False) + "\n" for row in records)
    if path.exists() and path.stat().st_size + len(encoded.encode()) > MAX_OUTPUT_BYTES:
        path.replace(path.with_suffix(path.suffix + ".1"))
    # One writer per output file. Historical retention is a separate archive job.
    with path.open("a", encoding="utf-8") as output:
        output.write(encoded)
        output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=ENDPOINTS, default="sandbox")
    parser.add_argument("--ticker", required=True, help="Vendor ticker, e.g. AAPL-NMS-EQT")
    parser.add_argument("--output", type=Path, required=True, help="Dashboard live JSONL file")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Z0-9._]{1,32}-[A-Z0-9]{1,12}-[A-Z0-9]{1,12}", args.ticker):
        parser.error("--ticker must be a single vendor ticker key, not a query")
    if not math.isfinite(args.interval) or not 1 <= args.interval <= 3600:
        parser.error("--interval must be between 1 and 3600 seconds")
    if not 1 <= args.limit <= 1000:
        parser.error("--limit must be between 1 and 1000")
    try:
        api_key = Path(os.environ["SPIDERROCK_API_KEY_FILE"]).read_text().strip()
        if not api_key:
            raise ValueError("Empty credential")
        vendor, commit = load_vendor()
        import requests

        seen = OrderedDict()
        failures = 0
        with requests.Session() as session:
            # Never send the query-string credential through an ambient proxy.
            session.trust_env = False
            while True:
                try:
                    body = fetch_snapshot(
                        session, ENDPOINTS[args.environment], api_key, args.ticker, args.limit
                    )
                    records = decode_snapshot(vendor, body, args.limit)
                    rows, identities = [], []
                    for record in records:
                        row = normalize(record, args.environment, commit)
                        identity = (row["symbol"], row["event_time"], row["bid"], row["ask"])
                        if identity not in seen:
                            rows.append(row)
                            identities.append(identity)
                    append_records(args.output, rows)
                    for identity in identities:
                        seen[identity] = None
                    while len(seen) > 100000:
                        seen.popitem(last=False)
                    failures = 0
                    if args.once:
                        return
                    time.sleep(args.interval)
                except Exception:  # noqa: BLE001 -- never expose credential-bearing request errors
                    # requests exceptions can contain the URL and API key.
                    # Never log exceptions, query strings, or response bodies.
                    print("SpiderRock snapshot failed; no partial batch published", file=sys.stderr)
                    if args.once:
                        raise SystemExit(1) from None
                    failures += 1
                    time.sleep(min(60, args.interval * 2 ** min(failures, 6)))
    except KeyboardInterrupt:
        return
    except Exception:  # noqa: BLE001 -- fail closed without exposing secret-file contents
        parser.exit(
            1, "SpiderRock startup failed; check vendor integrity and secret-file configuration\n"
        )


if __name__ == "__main__":
    main()
