"""Run with uvicorn services.analytics.app:create_app --factory --port 8080."""

import asyncio
import csv
import hashlib
import hmac
import json
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .data import MAX_ROWS, DatasetStore, LiveTail, risk_metrics
from .telemetry import Telemetry

BASE = Path(__file__).parent
COOKIE = "analytics_session"
SESSION_SECONDS = 8 * 3600
LOGIN = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>OptimizR sign in</title><h1>OptimizR analytics</h1>
<p>Local shared-token access. Production requires a TLS identity-aware proxy.</p>
<form method="post" action="/login"><label>Access token
<input name="token" type="password" required autocomplete="current-password" maxlength="1024">
</label><button>Sign in</button></form></html>"""


def create_app():
    token_file = os.getenv("ANALYTICS_TOKEN_FILE")
    if token_file:
        try:
            with Path(token_file).open(encoding="utf-8") as stream:
                raw_token = stream.read(4097)
            if len(raw_token) > 4096:
                raise RuntimeError("ANALYTICS_TOKEN_FILE exceeds the size limit")
            token = raw_token.rstrip("\r\n")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError("Unable to read ANALYTICS_TOKEN_FILE") from exc
    else:
        token = os.getenv("ANALYTICS_TOKEN", "")
    if not 32 <= len(token) <= 1024:
        raise RuntimeError("Analytics token must contain between 32 and 1024 characters")
    secret = token.encode()
    secure_cookie = os.getenv("COOKIE_SECURE", "true").lower() == "true"
    public_origin = os.getenv("PUBLIC_ORIGIN", "").rstrip("/")
    root = Path(os.getenv("DATA_DIR", "/data"))
    datasets = DatasetStore(root / "datasets")
    live = LiveTail(Path(os.getenv("LIVE_JSONL_PATH", str(root / "live/market.jsonl"))))
    telemetry = Telemetry()

    def signed_session():
        expires = str(int(time.time()) + SESSION_SECONDS)
        signature = hmac.new(secret, expires.encode(), hashlib.sha256).hexdigest()
        return expires + "." + signature

    def authenticated(connection):
        authorization = connection.headers.get("authorization", "")
        if authorization.startswith("Bearer "):
            return hmac.compare_digest(authorization[7:].encode(), secret)
        cookie = connection.cookies.get(COOKIE, "")
        try:
            expires, signature = cookie.split(".")
            expiry = int(expires)
            if not time.time() < expiry <= time.time() + SESSION_SECONDS:
                return False
            expected = hmac.new(secret, expires.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(signature.encode(), expected.encode())
        except (ValueError, UnicodeError):
            return False

    def same_origin(connection):
        origin = connection.headers.get("origin")
        expected = public_origin or (
            f"{'https' if connection.url.scheme in ('https', 'wss') else 'http'}"
            f"://{connection.headers.get('host', '')}"
        )
        return origin == expected

    async def ingest():
        while True:
            rejected = live.rejected
            try:
                with telemetry.tracer.start_as_current_span("analytics.ingest"):
                    count = await run_in_threadpool(live.poll)
                    telemetry.ingested.add(count)
                    telemetry.rejected.add(live.rejected - rejected)
            except OSError:
                telemetry.rejected.add(1, {"reason": "io_error"})
            await asyncio.sleep(0.5)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(ingest())
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        telemetry.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.live = live
    app.state.connections = 0

    @app.middleware("http")
    async def security_and_telemetry(request: Request, call_next):
        start = time.monotonic()
        with telemetry.tracer.start_as_current_span("analytics.http") as span:
            if request.url.path not in ("/healthz", "/login") and not authenticated(request):
                response = (
                    RedirectResponse("/login", status_code=303)
                    if request.url.path == "/"
                    else JSONResponse({"detail": "Authentication required"}, status_code=401)
                )
            else:
                response = await call_next(request)
            attributes = {
                "http.request.method": request.method,
                "http.response.status_code": response.status_code,
            }
            for key, value in attributes.items():
                span.set_attribute(key, value)
            telemetry.requests.add(1, attributes)
            telemetry.duration.record(time.monotonic() - start, attributes)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "same-origin",
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; "
                    "worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
                    "connect-src 'self'; img-src 'self' data:; "
                    "frame-ancestors 'none'; form-action 'self'; base-uri 'none'"
                ),
            }
        )
        return response

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/login")
    def login_page():
        return HTMLResponse(LOGIN)

    @app.post("/login")
    async def login(request: Request):
        if not same_origin(request):
            raise HTTPException(403, "Same-origin login required")
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > 2048:
                raise HTTPException(413, "Login request too large")
        try:
            supplied = parse_qs(body.decode()).get("token", [""])[0].encode()
        except UnicodeError:
            raise HTTPException(400, "Invalid login request") from None
        if not hmac.compare_digest(supplied, secret):
            raise HTTPException(401, "Invalid token")
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            COOKIE,
            signed_session(),
            max_age=SESSION_SECONDS,
            httponly=True,
            secure=secure_cookie,
            samesite="strict",
        )
        return response

    @app.post("/logout")
    def logout(request: Request):
        if not same_origin(request):
            raise HTTPException(403, "Same-origin logout required")
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(COOKIE)
        return response

    @app.get("/")
    def dashboard():
        return FileResponse(BASE / "index.html")

    @app.get("/api/datasets")
    def list_datasets():
        return {"datasets": datasets.list(), "max_rows": MAX_ROWS, "format": "csv"}

    def load(dataset_id):
        try:
            return datasets.load(dataset_id)
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found") from None
        except (ValueError, TypeError, UnicodeError, OSError, csv.Error) as exc:
            raise HTTPException(400, "Invalid or oversized CSV dataset") from exc

    @app.get("/api/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        with telemetry.tracer.start_as_current_span("analytics.dataset.load"):
            rows, truncated = load(dataset_id)
        return {"rows": rows, "truncated": truncated, "max_rows": MAX_ROWS}

    @app.get("/api/reports/{dataset_id}")
    def report(dataset_id: str):
        try:
            return datasets.report(dataset_id)
        except FileNotFoundError:
            raise HTTPException(404, "No imported Databricks report for this dataset") from None
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError):
            raise HTTPException(409, "Invalid or mismatched Databricks report") from None

    @app.get("/api/risk")
    def risk(
        symbol: str = Query(min_length=1, max_length=512),
        dataset_id: str = Query(default="live", max_length=84),
        periods_per_year: float = Query(default=252.0, ge=1.0, le=1_000_000.0),
    ):
        rows = live.since(None)[0]["rows"] if dataset_id == "live" else load(dataset_id)[0]
        try:
            with telemetry.tracer.start_as_current_span("analytics.optimizer.risk"):
                return risk_metrics(rows, symbol, periods_per_year)
        except (ImportError, RuntimeError, TypeError):
            raise HTTPException(503, "Compiled OptimizR risk binding unavailable") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.websocket("/ws/live")
    async def websocket(websocket: WebSocket):
        # Browser cookies must never authorize a cross-site WebSocket handshake.
        if not authenticated(websocket) or (
            COOKIE in websocket.cookies and not same_origin(websocket)
        ):
            await websocket.close(code=1008)
            return
        if app.state.connections >= 16:
            await websocket.close(code=1013)
            return
        app.state.connections += 1
        try:
            await websocket.accept()
            cursor = None
            while authenticated(websocket):
                payload, cursor = live.since(cursor)
                await asyncio.wait_for(websocket.send_json(payload), timeout=10)
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=0.5)
                    if message["type"] == "websocket.disconnect":
                        break
                except asyncio.TimeoutError:
                    pass
            else:
                await websocket.close(code=1008)
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError):
            pass
        finally:
            app.state.connections -= 1

    app.mount("/static", StaticFiles(directory=BASE / "static", check_dir=False), name="static")
    return app
