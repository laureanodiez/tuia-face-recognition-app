from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse

from lib.config import settings
from lib.schemas import (
    AsyncTaskCreated,
    InsertRequest,
    PredictRequest,
    StatusResponse,
    UploadResponse,
)
from lib.services.face_service import FaceService
from lib.services.task_manager import TaskManager
from lib.storage.embedding_store import EmbeddingStore
from lib.storage.pgvector_store import PgVectorEmbeddingStore
import logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["face-recognition"])
task_manager = TaskManager()

if settings.use_pgvector:
    logger.info("Using PostgreSQL vector store")
    embedding_store = PgVectorEmbeddingStore(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        embedding_dim=settings.embedding_dim,
    )
else:
    logger.info("Using JSON file vector store")
    embedding_store = EmbeddingStore(settings.embeddings_path)


def _resolved_model_path() -> Path:
    """MODEL_PATH may be a directory; join with MODEL_NAME when set."""
    base = settings.model_path
    name = (settings.model_name or "").strip()
    if name and base.is_dir():
        return base / name
    return base


def _resolved_detector_path() -> Path | None:
    """DETECTOR_MODEL_NAME is optional: unset means the detector does not need a file."""
    name = (settings.detector_model_name or "").strip()
    return settings.model_path / name if name else None


face_service = FaceService(
    store=embedding_store,
    similarity_metric=settings.similarity_metric,
    similarity_threshold=settings.similarity_threshold,
    face_size=settings.face_size,
    model_path=_resolved_model_path(),
    output_path=settings.output_path,
    detector_path=_resolved_detector_path(),
    det_score_threshold=settings.det_score_threshold,
    det_input_size=settings.det_input_size,
    det_pad_frac=settings.det_pad_frac,
    min_face_size=settings.min_face_size,
    align_margin=settings.align_margin,
)


def _safe_file_under(root: Path, relpath: str) -> Path:
    root = root.resolve()
    rel = (relpath or "").replace("\\", "/").strip("/")
    if not rel or ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="invalid path")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path outside allowed root") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return candidate


def _file_to_public_url(path: Path) -> str | None:
    try:
        p = path.resolve()
    except OSError:
        return None
    for base, prefix in (
        (settings.output_path, "/files/output"),
        (settings.data_path, "/files/data"),
    ):
        try:
            rel = p.relative_to(base.resolve())
            return f"{prefix}/{rel.as_posix()}"
        except ValueError:
            continue
    return None


def _urls_for_status(link: str) -> tuple[str | None, str | None]:
    if link in ("", "none"):
        return None, None
    p = Path(link)
    if not p.is_file():
        return None, None
    artifact = _file_to_public_url(p)
    source: str | None = None
    if p.suffix.lower() == ".json":
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            sp = payload.get("source_path")
            if isinstance(sp, str) and sp:
                source = _file_to_public_url(Path(sp))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return artifact, source


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    uploads_dir = settings.output_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    raw_name = (file.filename or "upload").strip()
    suffix = Path(raw_name).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
        suffix = ".jpg"
    dest = uploads_dir / f"up_{uuid4().hex}{suffix}"
    data = await file.read()
    dest.write_bytes(data)
    rel = dest.resolve().relative_to(settings.output_path.resolve())
    return UploadResponse(
        path=str(dest.resolve()),
        download_url=f"/files/output/{rel.as_posix()}",
    )


@router.get("/files/output/{file_path:path}")
async def download_output_file(file_path: str) -> FileResponse:
    path = _safe_file_under(settings.output_path, file_path)
    media, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, filename=path.name, media_type=media or "application/octet-stream")


@router.get("/files/data/{file_path:path}")
async def download_data_file(file_path: str) -> FileResponse:
    path = _safe_file_under(settings.data_path, file_path)
    media, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, filename=path.name, media_type=media or "application/octet-stream")


@router.post("/insert", response_model=AsyncTaskCreated)
async def insert(payload: InsertRequest, response: Response) -> AsyncTaskCreated:
    response.status_code = status.HTTP_202_ACCEPTED
    job_id = task_manager.create_job()

    async def _process() -> str:
        await asyncio.sleep(0.05)
        logger.info(f"Registering identity: {payload.identity} with image: {payload.image_path}")
        record = face_service.register_identity(
            identity=payload.identity,
            image_path=payload.image_path,
            metadata=payload.metadata,
        )
        return record.path

    task_manager.schedule(job_id, _process())
    return AsyncTaskCreated(job_id=job_id)


@router.post("/predict", response_model=AsyncTaskCreated)
async def predict(payload: PredictRequest, response: Response) -> AsyncTaskCreated:
    response.status_code = status.HTTP_202_ACCEPTED
    job_id = task_manager.create_job()

    async def _process() -> str:
        await asyncio.sleep(0.05)
        logger.info(f"Predicting image: {payload.source_path}")
        return face_service.predict(payload.source_path, settings.output_path)

    task_manager.schedule(job_id, _process())
    return AsyncTaskCreated(job_id=job_id)


@router.get("/status/{job_id}", response_model=StatusResponse)
async def status_by_id(job_id: str) -> StatusResponse:
    state = task_manager.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    artifact_url: str | None = None
    source_image_url: str | None = None
    if state.status == "done":
        artifact_url, source_image_url = _urls_for_status(state.link)
    return StatusResponse(
        status=state.status,
        link=state.link,
        reason=state.error,
        artifact_url=artifact_url,
        source_image_url=source_image_url,
    )
