"""FastAPI application: package supply for after-sales replacement details."""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from anyio import to_thread
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import packing, service
from .db import connect, database_path, db_lock, init_db
from .models import Health, Manifest, ManifestList, TicketLimitError

CSV_MEDIA_TYPE = "text/csv; charset=utf-8"


def _open() -> sqlite3.Connection:
    return connect(database_path())


async def _in_db(func: Callable[..., Any], *args: Any) -> Any:
    """Run one SQLite-touching call on a worker thread, holding the DB lock.

    The lock wait happens on the event loop (no worker thread is consumed),
    and the worker thread is borrowed only for the short call itself.
    """
    async with db_lock():
        return await to_thread.run_sync(func, *args)


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = _open()
    try:
        await _in_db(init_db, conn)
    finally:
        conn.close()
    yield


app = FastAPI(
    title="售后换件明细分包供数",
    version="1.0.0",
    description=(
        "按批次接收工单表与换件表，按工单号左连接后依工单号字典序分包，"
        "发布不可变清单版本，并按版本号与包号逐包提供 CSV 下载。"
    ),
    lifespan=lifespan,
)

app.add_exception_handler(
    service.NotFound,
    lambda request, exc: JSONResponse(status_code=404, content={"detail": str(exc)}),
)
app.add_exception_handler(
    service.BadRequest,
    lambda request, exc: JSONResponse(status_code=400, content={"detail": str(exc)}),
)
app.add_exception_handler(
    packing.InputError,
    lambda request, exc: JSONResponse(status_code=422, content={"detail": str(exc)}),
)
app.add_exception_handler(
    service.CorruptSnapshot,
    lambda request, exc: JSONResponse(status_code=409, content={"detail": str(exc)}),
)


@app.exception_handler(packing.TicketTooLarge)
async def handle_ticket_too_large(
    request: Request, exc: packing.TicketTooLarge
) -> JSONResponse:
    payload = TicketLimitError.model_validate(
        {
            "detail": {
                "ticket_no": exc.ticket_no,
                "row_count": exc.row_count,
                "row_limit": exc.row_limit,
                "message": str(exc),
            }
        }
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.get("/health", response_model=Health, tags=["ops"])
async def health() -> Health:
    def check() -> None:
        conn = _open()
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()

    await _in_db(check)
    return Health(status="ok", database=str(database_path()))


@app.post(
    "/api/batches/{batch_id}/publish",
    response_model=Manifest,
    status_code=201,
    responses={
        422: {
            "model": TicketLimitError,
            "description": "某工单连接行数超过每包上限，本次不发布",
        }
    },
    tags=["publish"],
)
async def publish_batch(
    batch_id: str,
    tickets: UploadFile = File(..., description="工单 CSV：ticket_no,station"),
    parts: UploadFile = File(..., description="换件 CSV：ticket_no,part_code"),
    row_limit: int = Form(..., gt=0, description="每包行数上限"),
) -> Manifest:
    """接收同批两表与每包行数上限，校验后发布一个新清单版本。

    两表按工单号左连接，重复行各自保留，无换件的工单行也保留。任一工单
    连接后的行数超过上限时返回 422（含工单号与连接行数），本批不写库，
    原清单继续有效。只有整笔写事务提交成功后才返回新版本；失败时调用方
    仍以最近一次成功发布的清单为准。
    """
    batch = service.require_batch_id(batch_id)
    limit = service.parse_row_limit(row_limit)
    ticket_bytes, part_bytes = await to_thread.run_sync(
        _read_uploads, tickets, parts
    )
    conn = _open()
    try:
        manifest = await _in_db(
            service.publish, conn, batch, ticket_bytes, part_bytes, limit
        )
    finally:
        conn.close()
    return Manifest.model_validate(service.with_download_urls(manifest))


def _read_uploads(tickets: UploadFile, parts: UploadFile) -> tuple[bytes, bytes]:
    return tickets.file.read(), parts.file.read()


@app.get(
    "/api/batches/{batch_id}/manifests",
    response_model=ManifestList,
    tags=["manifest"],
)
async def list_batch_manifests(batch_id: str) -> ManifestList:
    """列出版本号与各包首末工单号、行数；不展开连接明细。"""
    batch = service.require_batch_id(batch_id)
    conn = _open()
    try:
        manifests = await _in_db(service.list_manifests, conn, batch)
    finally:
        conn.close()
    return ManifestList(
        batch_id=batch,
        manifests=[
            Manifest.model_validate(service.with_download_urls(m)) for m in manifests
        ],
    )


@app.get(
    "/api/batches/{batch_id}/manifests/latest",
    response_model=Manifest,
    tags=["manifest"],
)
async def latest_batch_manifest(batch_id: str) -> Manifest:
    batch = service.require_batch_id(batch_id)
    conn = _open()
    try:
        manifest = await _in_db(service.latest_manifest, conn, batch)
    finally:
        conn.close()
    return Manifest.model_validate(service.with_download_urls(manifest))


@app.get("/api/manifests/{version}", response_model=Manifest, tags=["manifest"])
async def get_manifest(version: str) -> Manifest:
    """按版本号查看清单；发布新版后旧清单仍可查询。"""
    conn = _open()
    try:
        manifest = await _in_db(service.get_manifest, conn, version)
    finally:
        conn.close()
    return Manifest.model_validate(service.with_download_urls(manifest))


@app.get(
    "/api/manifests/{version}/manifest.csv",
    response_class=PlainTextResponse,
    responses={200: {"content": {CSV_MEDIA_TYPE: {}}}},
    tags=["manifest"],
)
async def download_manifest_csv(version: str) -> PlainTextResponse:
    """下载清单 CSV：版本号、各包首末工单号与行数（不展开明细）。"""
    conn = _open()
    try:
        manifest = await _in_db(service.get_manifest, conn, version)
    finally:
        conn.close()
    return PlainTextResponse(
        service.manifest_csv(manifest),
        media_type=CSV_MEDIA_TYPE,
        headers=_csv_headers(f"{version}-manifest.csv"),
    )


@app.get(
    "/api/manifests/{version}/packages/{package_no}.csv",
    responses={200: {"content": {CSV_MEDIA_TYPE: {}}}},
    tags=["download"],
)
async def download_package(version: str, package_no: int) -> StreamingResponse:
    """Stream one package's CSV, reading only that version's source records.

    The CSV is generated in the original row order and sent as it is produced.
    Every SQLite access is a bounded batch run on a worker thread while the
    process DB lock is held; between batches the stream yields back to the
    event loop and releases both the lock and the worker thread. A client that
    pauses receiving therefore only suspends the stream via transport
    backpressure, while publishing still acquires the lock and commits.
    An unknown version/package fails before the first byte is streamed (404).
    """
    conn = _open()
    try:
        # Prepare the reader (metadata lookup + cursor) under the lock so an
        # unknown version/package is a 404 before streaming starts.
        reader = await _in_db(service.PackageCsvReader, conn, version, package_no)
    except BaseException:
        conn.close()
        raise
    return StreamingResponse(
        _stream_package(conn, reader),
        media_type=CSV_MEDIA_TYPE,
        headers=_csv_headers(f"{version}-p{package_no}.csv"),
    )


async def _stream_package(
    conn: sqlite3.Connection, reader: service.PackageCsvReader
):
    try:
        yield reader.header().encode("utf-8")
        while True:
            # The lock is held only while one bounded batch is fetched; the
            # await on yield (backpressure) happens with no lock and no worker.
            chunk: Optional[str] = await _in_db(reader.read)
            if chunk is None:
                break
            yield chunk.encode("utf-8")
    finally:
        await to_thread.run_sync(reader.close)
        conn.close()


def _csv_headers(filename: str) -> dict:
    return {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
    }
