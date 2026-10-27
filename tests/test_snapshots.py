import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from optimizr.analytics import normalize_sr4p_quote, risk_metrics, risk_report
from optimizr.snapshot import encode, export_snapshot, verify_snapshot
from services.analytics.data import DatasetStore
from services.analytics.sync import import_snapshot, sync_once


def rows():
    return [
        {
            "event_time": f"2026-01-{day + 1:02}T10:00:00Z",
            "symbol": symbol,
            "price": price,
            "price_kind": "option_midpoint",
        }
        for symbol in ("AAA", "BBB")
        for day, price in enumerate((100, 103, 101, 104, 102, 105))
    ]


def exported(tmp_path):
    marker = export_snapshot(tmp_path, rows(), 252, "catalog.schema.quotes")
    manifest = json.loads(marker.read_bytes())
    return marker, manifest, (tmp_path / manifest["file"]).read_bytes()


def test_report_preserves_individual_symbols_and_metrics():
    report = risk_report(rows()[::-1], 252)
    assert report["aggregation"] == "none"
    assert report["row_count"] == 12
    assert report["errors"] == []
    assert [result["symbol"] for result in report["results"]] == ["AAA", "BBB"]
    assert all(result["observations"] == 5 for result in report["results"])
    assert report["results"][0]["metrics"] == risk_metrics(rows(), "AAA")["metrics"]


@pytest.mark.parametrize("periods", [0, float("nan"), float("inf"), 1_000_001])
def test_invalid_annualization_rejected(periods):
    with pytest.raises(ValueError, match="periods_per_year"):
        risk_report(rows(), periods)


def test_ambiguous_or_insufficient_symbols_are_explicit_errors():
    data = rows()
    data[1]["event_time"] = data[0]["event_time"]
    data[-1]["price_kind"] = "trade"
    report = risk_report(data, 252)
    assert report["results"] == []
    assert report["errors"] == [
        {"symbol": "AAA", "detail": "Duplicate observation timestamps"},
        {"symbol": "BBB", "detail": "Mixed price kinds"},
    ]
    assert "five" in risk_report(rows()[:3], 252)["errors"][0]["detail"]


def test_snapshot_bounds():
    for data in ([], rows() * 834):
        with pytest.raises(ValueError, match="10000"):
            risk_report(data, 252)


def test_export_import_replay_and_content_integrity(tmp_path):
    _, manifest, content = exported(tmp_path)
    target = tmp_path / "datasets"
    name = import_snapshot(target, manifest, content)
    store = DatasetStore(target)
    loaded, truncated = store.load(name)
    assert len(loaded) == 12 and not truncated
    report = store.report(name)
    assert report["source_table"] == "catalog.schema.quotes"
    assert len(report["report"]["results"]) == 2
    assert import_snapshot(target, manifest, content) == name
    with pytest.raises(ValueError, match="checksum"):
        import_snapshot(target, manifest, content + b" ")
    (target / name).write_text("changed")
    with pytest.raises(ValueError, match="does not match"):
        store.report(name)
    with pytest.raises(ValueError, match="immutable"):
        import_snapshot(target, manifest, content)


def test_recomputed_metrics_reject_tampered_calculations(tmp_path):
    _, manifest, content = exported(tmp_path)
    snapshot = json.loads(content)
    snapshot["report"]["results"][0]["metrics"]["volatility"] += 1
    changed = encode(snapshot)
    manifest.update(bytes=len(changed), sha256=hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="numerical mismatch"):
        verify_snapshot(manifest, changed)


def test_stream_transfer_is_bounded_and_restart_idempotent(tmp_path):
    marker, manifest, content = exported(tmp_path)
    volume = "/Volumes/catalog/schema/exports"
    files = {
        f"{volume}/{marker.name}": marker.read_bytes(),
        f"{volume}/{manifest['file']}": content,
    }
    downloads = []

    def download(path):
        downloads.append(path)
        return SimpleNamespace(contents=io.BytesIO(files[path]))

    client = SimpleNamespace(
        files=SimpleNamespace(
            download=download,
            list_directory_contents=lambda path: [
                SimpleNamespace(path=f"{path}/{marker.name}", is_directory=False)
            ],
        )
    )
    directory = tmp_path / "datasets"
    assert len(sync_once(client, volume, directory)) == 1
    assert sync_once(client, volume, directory) == []
    assert downloads.count(f"{volume}/{manifest['file']}") == 1
    files[f"{volume}/{marker.name}"] = b"x" * 2049
    with pytest.raises(ValueError, match="size limit"):
        sync_once(client, volume, directory)


def test_path_escape_is_rejected(tmp_path):
    _, manifest, content = exported(tmp_path)
    manifest["file"] = "../../anything"
    with pytest.raises(ValueError, match="manifest"):
        import_snapshot(tmp_path / "datasets", manifest, content)


def test_sr4p_adapter_preserves_full_key_session_clock_and_price_kind():
    record = {
        "session_id": "20261016T134525Z-j123",
        "okey_at": "EQT",
        "okey_cp": "Call",
        "okey_dt": "2026-12-18",
        "okey_tk": "ABC",
        "okey_ts": "NMS",
        "okey_xx": 100,
        "obid": 2,
        "oask": 4,
        "timestamp": "2026-10-16T14:00:00Z",
    }
    row = normalize_sr4p_quote(record, record["session_id"])
    assert row["price"] == 3
    assert row["price_kind"] == "option_midpoint"
    assert row["event_time"] == "2026-10-16T14:00:00+00:00"
    assert set(json.loads(row["symbol"])) == {
        "okey_at",
        "okey_cp",
        "okey_dt",
        "okey_tk",
        "okey_ts",
        "okey_xx",
    }
    with pytest.raises(ValueError, match="session"):
        normalize_sr4p_quote(record, "different")
    with pytest.raises(ValueError, match="two-sided"):
        normalize_sr4p_quote({**record, "obid": 5}, record["session_id"])
    with pytest.raises(ValueError, match="key component"):
        normalize_sr4p_quote({**record, "okey_cp": None}, record["session_id"])
