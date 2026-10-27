FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY services/analytics/package*.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY services/analytics/build.mjs services/analytics/frontend.js services/analytics/frontend.test.mjs services/analytics/style.css ./
RUN npm test && npm run check && npm run build

FROM rust:1.93-bookworm AS optimizer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev libopenblas-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /buildenv
ENV PATH="/buildenv/bin:${PATH}"
RUN pip install --no-cache-dir maturin==1.15.0
WORKDIR /build
COPY Cargo.* pyproject.toml README.md LICENSE ./
COPY src ./src
COPY python ./python
RUN test -f Cargo.lock || cargo generate-lockfile
RUN maturin build --release --locked --features python-bindings --interpreter python3 \
    --out /wheels --compatibility linux

FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0-pthread \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 analytics \
    && useradd --uid 10001 --gid analytics --no-create-home analytics \
    && mkdir -p /data/datasets /data/live \
    && chown -R analytics:analytics /data
WORKDIR /app
COPY LICENSE /usr/share/licenses/optimiz-rs/LICENSE
COPY services/analytics/requirements.txt /app/requirements.txt
COPY services/analytics/requirements-sync.txt /app/requirements-sync.txt
COPY --from=optimizer /wheels /wheels
RUN pip install --no-cache-dir -r /app/requirements.txt /wheels/*.whl \
    && python -m venv /syncenv \
    && /syncenv/bin/pip install --no-cache-dir -r /app/requirements-sync.txt /wheels/*.whl \
    && rm -rf /wheels \
    && python -c "from optimizr import compute_risk_metrics_py; assert callable(compute_risk_metrics_py)"
COPY services/analytics/*.py services/analytics/index.html /app/services/analytics/
COPY --from=frontend /build/static /app/services/analytics/static
ENV DATA_DIR=/data PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 COOKIE_SECURE=true
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"]
CMD ["uvicorn", "services.analytics.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-proxy-headers", "--ws-max-size", "4096", "--limit-concurrency", "64"]
