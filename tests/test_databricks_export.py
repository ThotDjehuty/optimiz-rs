import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


class NotebookExit(Exception):
    pass


class Frame:
    def __init__(self, frame):
        self.frame = frame
        self.columns = list(frame.columns)

    def select(self, *columns):
        return Frame(self.frame[list(columns)])

    def orderBy(self, *columns):
        return Frame(self.frame.sort_values(list(columns)))

    def limit(self, count):
        return Frame(self.frame.iloc[:count])

    def toPandas(self):
        return self.frame.copy()


def run_export(monkeypatch, tmp_path, count=6):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "exports").mkdir(exist_ok=True)
    params = {
        "input_table": "catalog.schema.quotes",
        "output_volume": "/Volumes/catalog/schema/exports",
        "max_rows": "10000",
        "periods_per_year": "252",
    }
    frame = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=count, tz="UTC"),
            "symbol": ["AAA"] * count,
            "price": [100 + i for i in range(count)],
        }
    )
    from optimizr import snapshot

    real_export = snapshot.export_snapshot
    monkeypatch.setattr(
        snapshot,
        "export_snapshot",
        lambda directory, rows, periods, table: real_export(
            tmp_path / "exports", rows, periods, table
        ),
    )
    utils = SimpleNamespace(
        widgets=SimpleNamespace(
            text=lambda name, value: params.setdefault(name, value), get=params.get
        ),
        notebook=SimpleNamespace(exit=lambda value: (_ for _ in ()).throw(NotebookExit(value))),
    )
    spark = SimpleNamespace(
        conf=SimpleNamespace(set=lambda *args: None), table=lambda name: Frame(frame)
    )
    path = Path(__file__).resolve().parents[1] / "deploy/databricks/export_analytics.py"
    return runpy.run_path(str(path), init_globals={"spark": spark, "dbutils": utils})


def test_notebook_executes_calculation_and_publishes_manifest(monkeypatch, tmp_path):
    with pytest.raises(NotebookExit) as result:
        run_export(monkeypatch, tmp_path)
    output = json.loads(str(result.value))
    manifest = json.loads(Path(output["manifest_path"]).read_bytes())
    snapshot = json.loads((tmp_path / "exports" / manifest["file"]).read_bytes())
    assert snapshot["report"]["results"][0]["observations"] == 5
    assert snapshot["report"]["results"][0]["metrics"]


def test_notebook_refuses_silent_truncation(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="exceeds max_rows"):
        run_export(monkeypatch, tmp_path, 10001)
    assert not list((tmp_path / "exports").iterdir())
