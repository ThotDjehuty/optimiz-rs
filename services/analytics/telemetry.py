"""OTLP traces and metrics without exporting tokens, records, or file paths."""

import os

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class Telemetry:
    def __init__(self):
        resource = Resource.create(
            {"service.name": os.getenv("OTEL_SERVICE_NAME", "optimizr-analytics")}
        )
        self.traces = TracerProvider(resource=resource)
        readers = []
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            self.traces.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
        self.meters = MeterProvider(resource=resource, metric_readers=readers)
        self.tracer = self.traces.get_tracer(__name__)
        meter = self.meters.get_meter(__name__)
        self.requests = meter.create_counter("analytics.http.requests")
        self.ingested = meter.create_counter("analytics.ingest.records")
        self.rejected = meter.create_counter("analytics.ingest.rejected")
        self.duration = meter.create_histogram("analytics.http.duration", unit="s")

    def close(self):
        self.traces.shutdown()
        self.meters.shutdown()
