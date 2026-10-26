import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "spiderrock_ingest", ROOT / "scripts" / "spiderrock_ingest.py"
)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def status(code=1):
    enum = SimpleNamespace(values_by_name={"QRESULT_OK": SimpleNamespace(number=1)})
    return SimpleNamespace(
        result=code,
        DESCRIPTOR=SimpleNamespace(fields_by_name={"result": SimpleNamespace(enum_type=enum)}),
    )


def test_vendor_parser_is_used():
    calls = []

    def parse(body):
        calls.append(body)
        return ["vendor-record"], status()

    vendor = SimpleNamespace(parse_custom_protocol_response=parse)
    assert bridge.decode_snapshot(vendor, b"raw-vendor-response", 1) == ["vendor-record"]
    assert calls == [b"raw-vendor-response"]


@pytest.mark.parametrize("result", [None, status(0), status(2)])
def test_incomplete_or_unsuccessful_snapshot_rejected(result):
    vendor = SimpleNamespace(parse_custom_protocol_response=lambda body: ([], result))
    with pytest.raises(ValueError):
        bridge.decode_snapshot(vendor, b"", 1)


def test_parser_errors_are_not_published(capsys):
    def parse(body):
        print("Failed to parse message: sensitive payload")
        return [], status()

    with pytest.raises(ValueError, match="decoder"):
        bridge.decode_snapshot(SimpleNamespace(parse_custom_protocol_response=parse), b"", 1)
    assert "sensitive" not in capsys.readouterr().out


def test_response_count_is_bounded():
    vendor = SimpleNamespace(parse_custom_protocol_response=lambda body: ([1, 2], status()))
    with pytest.raises(ValueError, match="limit"):
        bridge.decode_snapshot(vendor, b"", 1)


class Response:
    status_code = 200

    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield from self.chunks


@pytest.mark.skipif(not hasattr(bridge.signal, "setitimer"), reason="Linux bridge runtime")
def test_transport_readonly_and_redirects_disabled():
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response([b"vendor", b"data"])

    assert (
        bridge.fetch_snapshot(
            SimpleNamespace(get=get),
            bridge.ENDPOINTS["sandbox"],
            "test-placeholder",
            "AAPL-NMS-EQT",
            5,
        )
        == b"vendordata"
    )
    _, options = calls[0]
    assert options["params"]["cmd"] == "getmsgs"
    assert options["params"]["msgtype"] == "LiveImpliedQuoteAdj"
    assert options["params"]["where"] == "ticker:eq:AAPL-NMS-EQT"
    assert options["allow_redirects"] is False
    assert options["timeout"] == (5, 10)


@pytest.mark.skipif(not hasattr(bridge.signal, "setitimer"), reason="Linux bridge runtime")
def test_transport_is_bounded(monkeypatch):
    monkeypatch.setattr(bridge, "MAX_RESPONSE_BYTES", 2)
    session = SimpleNamespace(get=lambda *a, **kw: Response([b"123"]))
    with pytest.raises(ValueError, match="bounds"):
        bridge.fetch_snapshot(session, bridge.ENDPOINTS["sandbox"], "", "AAPL-NMS-EQT", 1)


@pytest.mark.skipif(not hasattr(bridge.signal, "setitimer"), reason="Linux bridge runtime")
def test_deadline_interrupts_blocked_read():
    previous = bridge.signal.getsignal(bridge.signal.SIGALRM)
    with pytest.raises(TimeoutError, match="deadline"), bridge.response_deadline(0.02):
        time.sleep(5)
    assert bridge.signal.getsignal(bridge.signal.SIGALRM) == previous
    assert bridge.signal.getitimer(bridge.signal.ITIMER_REAL) == (0.0, 0.0)


def test_jsonl_append_and_rotation(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "MAX_OUTPUT_BYTES", 20)
    target = tmp_path / "market.jsonl"
    bridge.append_records(target, [{"price": 1}])
    bridge.append_records(target, [{"price": 2}])
    assert json.loads(target.read_text()) == {"price": 2}
    assert json.loads(target.with_suffix(".jsonl.1").read_text()) == {"price": 1}


def test_pinned_vendor_parser_roundtrip():
    """Runs in CI's recursive checkout without contacting any vendor endpoint."""
    if not (ROOT / bridge.VENDOR_PATH / ".git").exists():
        pytest.skip("Requires initialized vendor checkout and bridge dependencies")
    vendor, commit = bridge.load_vendor()
    record = vendor.LiveImpliedQuoteAdj()
    record._meta.message_type = "LiveImpliedQuoteAdj"
    record.pkey.okey.ticker = "AAPL"
    record.pkey.okey.expiration.year = 2027
    record.pkey.okey.expiration.month = 1
    record.pkey.okey.expiration.day = 15
    record.pkey.okey.strike = 200
    record.obid = 1.0
    record.oask = 2.0
    record.timestamp.FromJsonString("2026-10-26T10:00:00Z")
    result = vendor.QueryResult()
    result._meta.message_type = "QueryResult"
    result.result = (
        result.DESCRIPTOR.fields_by_name["result"].enum_type.values_by_name["QRESULT_OK"].number
    )
    result.num_messages_sent = 1

    def frame(message):
        payload = message.SerializeToString()
        return b"\r\nP" + b"00000" + f"{len(payload):06d}".encode() + payload

    decoded = bridge.decode_snapshot(vendor, frame(record) + frame(result), 1)
    row = bridge.normalize(decoded[0], "sandbox", commit)
    assert row["price"] == 1.5
    assert row["price_kind"] == "option_midpoint"
    assert row["event_time"] == "2026-10-26T10:00:00Z"
    assert json.loads(row["symbol"])["ticker"] == "AAPL"
    record.oask = 0
    with pytest.raises(ValueError, match="two-sided"):
        bridge.normalize(record, "sandbox", commit)
