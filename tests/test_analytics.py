import csv
import json
import math
import sys
import types
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from services.analytics import telemetry
from services.analytics.app import create_app
from services.analytics.data import (
    MAX_FILE_BYTES,
    DatasetStore,
    LiveTail,
    normalize_record,
    risk_metrics,
)

TOKEN = "test-only-token-" + "x" * 32
RECORD = {"event_time": "2026-01-01T10:00:00Z", "symbol": "XYZ", "price": 100}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYTICS_TOKEN", TOKEN)
    monkeypatch.delenv("ANALYTICS_TOKEN_FILE", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("PUBLIC_ORIGIN", raising=False)
    monkeypatch.delenv("LIVE_JSONL_PATH", raising=False)
    (tmp_path / "datasets").mkdir()
    with TestClient(create_app()) as result:
        yield result


def login(client):
    response = client.post(
        "/login",
        data={"token": TOKEN},
        headers={"Origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "httponly" in response.headers["set-cookie"].lower()


def test_auth_every_data_surface(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/", follow_redirects=False).status_code == 303
    for path in (
        "/api/datasets",
        "/api/risk?symbol=XYZ",
        "/api/reports/test.csv",
        "/static/app.js",
    ):
        assert client.get(path).status_code == 401
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/live"):
        pass
    assert client.post("/login", data={"token": TOKEN}).status_code == 403
    assert (
        client.post(
            "/login", data={"token": "wrong"}, headers={"Origin": "http://testserver"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/login", content=b"x" * 3000, headers={"Origin": "http://testserver"}
        ).status_code
        == 413
    )
    login(client)
    assert client.get("/").status_code == 200
    assert client.get("/api/datasets").status_code == 200
    with pytest.raises(WebSocketDisconnect), client.websocket_connect(
        "/ws/live", headers={"Origin": "http://attacker"}
    ):
        pass
    with client.websocket_connect("/ws/live", headers={"Origin": "http://testserver"}) as ws:
        assert ws.receive_json() == {"type": "snapshot", "rows": []}
    assert client.post("/logout", headers={"Origin": "http://testserver"}).status_code == 200
    assert client.get("/api/datasets").status_code == 401


def test_bearer_and_session_tampering(client):
    headers = {"Authorization": "Bearer " + TOKEN}
    assert client.get("/api/datasets", headers=headers).status_code == 200
    client.cookies.set("analytics_session", "99999999999.bad")
    assert client.get("/api/datasets").status_code == 401
    assert (
        client.get("/api/risk?symbol=XYZ&periods_per_year=nan", headers=headers).status_code == 422
    )


def test_fails_closed_without_strong_token(monkeypatch):
    monkeypatch.delenv("ANALYTICS_TOKEN", raising=False)
    monkeypatch.delenv("ANALYTICS_TOKEN_FILE", raising=False)
    with pytest.raises(RuntimeError, match="32"):
        create_app()


def test_mounted_token_file(monkeypatch, tmp_path):
    secret = tmp_path / "analytics_token"
    secret.write_text(TOKEN + "\n")
    monkeypatch.setenv("ANALYTICS_TOKEN_FILE", str(secret))
    monkeypatch.setenv("ANALYTICS_TOKEN", "ignored environment value")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with TestClient(create_app()) as client:
        assert (
            client.get("/api/datasets", headers={"Authorization": "Bearer " + TOKEN}).status_code
            == 200
        )
    secret.unlink()
    with pytest.raises(RuntimeError, match="TOKEN_FILE"):
        create_app()


def test_telemetry_exports_metrics_only_unless_traces_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    for name in (
        "TracerProvider",
        "MeterProvider",
        "PeriodicExportingMetricReader",
        "BatchSpanProcessor",
        "OTLPMetricExporter",
        "OTLPSpanExporter",
    ):
        monkeypatch.setattr(telemetry, name, Mock())
    instance = telemetry.Telemetry()
    telemetry.OTLPMetricExporter.assert_called_once()
    telemetry.OTLPSpanExporter.assert_not_called()
    instance.close()
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    instance = telemetry.Telemetry()
    telemetry.OTLPSpanExporter.assert_called_once()
    instance.close()


@pytest.mark.parametrize(
    "change",
    [
        {"price": "nan"},
        {"price": float("inf")},
        {"price": 10**500},
        {"price": -1},
        {"price": True},
        {"event_time": "2026-01-01"},
        {"event_time": "invalid"},
        {"symbol": ""},
        {"symbol": "x" * 513},
    ],
)
def test_normalize_rejects_invalid_data(change):
    with pytest.raises((ValueError, TypeError)):
        normalize_record({**RECORD, **change})


def test_normalize_utc_and_numeric_csv():
    result = normalize_record(
        {**RECORD, "price": "101.25", "event_time": "2026-01-01T12:00:00+02:00"}
    )
    assert result["price"] == 101.25
    assert result["event_time"] == "2026-01-01T10:00:00+00:00"
    assert result["source"] == "unknown"
    assert result["price_kind"] == "unspecified"
    assert normalize_record({**RECORD, "source": "MLink"})["source"] == "MLink"
    assert normalize_record({**RECORD, "symbol": "x" * 256})["symbol"] == "x" * 256
    with pytest.raises(ValueError, match="source"):
        normalize_record({**RECORD, "source": ["not", "flat"]})
    record = normalize_record(
        {**RECORD, "source": "spiderrock:sandbox:snapshot", "price_kind": "option_midpoint"}
    )
    assert record["source"] == "spiderrock:sandbox:snapshot"
    assert record["price_kind"] == "option_midpoint"
    with pytest.raises(ValueError, match="price_kind"):
        normalize_record({**RECORD, "price_kind": "x" * 65})
    with pytest.raises(ValueError, match="source"):
        normalize_record({**RECORD, "source": "x" * 129})


def write_csv(path, count=7):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(RECORD))
        writer.writeheader()
        for i in range(count):
            writer.writerow(
                {**RECORD, "event_time": f"2026-01-{i+1:02}T10:00:00Z", "price": 100 + i}
            )


def test_dataset_identifiers_limits_and_symlinks(tmp_path):
    directory = tmp_path / "datasets"
    directory.mkdir()
    write_csv(directory / "prices.csv")
    write_csv(tmp_path / "outside.csv")
    (directory / "alias.csv").symlink_to(tmp_path / "outside.csv")
    store = DatasetStore(directory, max_rows=3)
    assert store.list() == ["prices.csv"]
    assert len(store.load("prices.csv")[0]) == 3
    assert store.load("prices.csv")[1] is True
    for name in ("../outside.csv", "/etc/passwd", "alias.csv", "prices.csv/extra", "bad.parquet"):
        with pytest.raises(ValueError):
            store.load(name)
    with (directory / "large.csv").open("wb") as stream:
        stream.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="16 MiB"):
        store.load("large.csv")


def test_dataset_lookup_only_uses_enumerated_candidates(monkeypatch, tmp_path):
    listed = tmp_path / "listed.csv"
    write_csv(listed)
    write_csv(tmp_path / "unlisted.csv")
    store = DatasetStore(tmp_path)
    path_type = type(store.directory)
    original_iterdir = path_type.iterdir

    def entries(path):
        if path == store.directory:
            return iter([listed])
        return original_iterdir(path)

    monkeypatch.setattr(path_type, "iterdir", entries)
    assert len(store.load("listed.csv")[0]) == 7
    with pytest.raises(FileNotFoundError):
        store.load("unlisted.csv")


def test_dataset_api_and_invalid_csv(client, tmp_path):
    write_csv(tmp_path / "datasets/prices.csv")
    login(client)
    assert client.get("/api/datasets").json()["datasets"] == ["prices.csv"]
    data = client.get("/api/datasets/prices.csv").json()
    assert len(data["rows"]) == 7
    assert isinstance(data["rows"][0]["price"], float)
    assert data["truncated"] is False
    assert client.get("/api/datasets/unknown.csv").status_code == 404
    (tmp_path / "datasets/invalid.csv").write_text("secret,value\nx,y\n")
    assert client.get("/api/datasets/invalid.csv").status_code == 400
    (tmp_path / "datasets/invalid.csv").write_text("event_time,symbol,price\n" + "x" * 150000)
    assert client.get("/api/datasets/invalid.csv").status_code == 400


def test_live_websocket_receives_new_bridge_records(client, tmp_path):
    directory = tmp_path / "live"
    directory.mkdir()
    login(client)
    with client.websocket_connect("/ws/live", headers={"Origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "snapshot"
        (directory / "market.jsonl").write_text(json.dumps(RECORD) + "\n")
        for _ in range(10):
            message = ws.receive_json()
            if message["rows"]:
                assert message["rows"] == [normalize_record(RECORD)]
                break
        else:
            pytest.fail("Bridge record was not delivered")


def test_production_origin_and_secure_cookie(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYTICS_TOKEN", TOKEN)
    monkeypatch.delenv("ANALYTICS_TOKEN_FILE", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PUBLIC_ORIGIN", "https://analytics.example")
    monkeypatch.setenv("COOKIE_SECURE", "true")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with TestClient(create_app(), base_url="https://analytics.example") as client:
        response = client.post(
            "/login",
            data={"token": TOKEN},
            headers={"Origin": "https://analytics.example"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "secure" in response.headers["set-cookie"].lower()
        assert client.get("/api/datasets").status_code == 200


def test_risk_api_rejects_insufficient_data(client, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "optimizr",
        types.SimpleNamespace(compute_risk_metrics_py=lambda *args: {}),
    )
    login(client)
    assert client.get("/api/risk?symbol=XYZ").status_code == 400


def test_live_partial_invalid_oversized_rotation_and_bounds(tmp_path):
    path = tmp_path / "live.jsonl"
    tail = LiveTail(path, max_rows=2)
    assert tail.poll() == 0
    encoded = json.dumps(RECORD).encode()
    path.write_bytes(encoded[:20])
    assert tail.poll() == 0
    with path.open("ab") as stream:
        stream.write(encoded[20:] + b"\ninvalid\n" + b"x" * 5000 + b"\n")
        stream.write(json.dumps({**RECORD, "price": 10**500}).encode() + b"\n")
        stream.write(b"[" * 1100 + b"]" * 1100 + b"\n")
        stream.write(encoded + b"\n" + encoded + b"\n")
    assert tail.poll() == 3
    assert tail.rejected == 4
    snapshot, cursor = tail.since(0)
    assert snapshot["type"] == "snapshot"
    assert len(snapshot["rows"]) == 2
    assert tail.since(cursor)[0] == {"type": "update", "rows": []}
    path.rename(tmp_path / "old.jsonl")
    path.write_bytes(encoded + b"\n")
    assert tail.poll() == 1
    assert len(tail.since(cursor)[0]["rows"]) == 1
    path.write_bytes(b"")
    assert tail.poll() == 0
    path.write_bytes(encoded + b"\n")
    assert tail.poll() == 1


def test_risk_delegates_to_public_optimizer(monkeypatch):
    import numpy as np

    called = {}

    def compute(returns, rate, periods):
        called.update(returns=returns, rate=rate, periods=periods)
        return {"sharpe_ratio": float("inf"), "volatility": 0.12}

    monkeypatch.setitem(
        sys.modules, "optimizr", types.SimpleNamespace(compute_risk_metrics_py=compute)
    )
    rows = [
        normalize_record({**RECORD, "event_time": f"2026-01-{i+1:02}T00:00:00Z", "price": 100 + i})
        for i in range(5)
    ]
    result = risk_metrics(rows[::-1], "XYZ", 365)
    np.testing.assert_allclose(
        called["returns"], np.diff(np.arange(100, 105)) / np.arange(100, 104)
    )
    assert called["rate"] == 0.0
    assert called["periods"] == 365
    assert result["metrics"]["sharpe_ratio"] is None
    assert math.isfinite(result["metrics"]["volatility"])
    with pytest.raises(ValueError, match="five"):
        risk_metrics(rows, "OTHER")


def test_imported_report_api(client, tmp_path):
    from optimizr.snapshot import export_snapshot
    from services.analytics.sync import import_snapshot

    rows = [
        {**RECORD, "event_time": f"2026-01-{i+1:02}T10:00:00Z", "price": 100 + i} for i in range(6)
    ]
    marker = export_snapshot(tmp_path, rows, 252, "catalog.schema.quotes")
    manifest = json.loads(marker.read_bytes())
    name = import_snapshot(
        tmp_path / "datasets", manifest, (tmp_path / manifest["file"]).read_bytes()
    )
    login(client)
    response = client.get(f"/api/reports/{name}")
    assert response.status_code == 200
    assert response.json()["report"]["aggregation"] == "none"
    assert response.json()["report"]["results"][0]["symbol"] == "XYZ"
    assert client.get("/api/reports/absent.csv").status_code == 404
    (tmp_path / "datasets" / name).write_text("corrupt")
    assert client.get(f"/api/reports/{name}").status_code == 409
