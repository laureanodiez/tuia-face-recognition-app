"""Pipeline facial compartido: detección -> alineación -> embedding.

Este módulo es la ÚNICA fuente de verdad del preprocesamiento. Lo importan tanto las
notebooks (1-pre-processing, 2-train) como `FaceService` (backend), para evitar el error
número uno del TP: "anda en la notebook pero no en la app".

Convenciones (fijas en todo el proyecto):
- Imágenes de entrada: BGR uint8 (como `cv2.imread`).
- Landmarks: array (5, 2) en orden k0 ojo izq, k1 ojo der, k2 nariz, k3 boca izq, k4 boca der.
- Alineación: transformación de similitud a la plantilla ArcFace (`norm_crop`).
- Embedding: BGR->RGB, (x - 127.5) / 127.5, CHW, batch, normalizado L2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils.face_align import norm_crop

ARCFACE_MEAN = 127.5
ARCFACE_STD = 127.5


@dataclass(frozen=True)
class RawDetection:
    """Una detección cruda del detector, en coordenadas de la imagen recibida."""

    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    kps: np.ndarray  # (5, 2)


def make_session(model_path: str | Path) -> onnxruntime.InferenceSession:
    """Sesión ONNX explícita en CPU (evita ambigüedad de providers en onnxruntime >= 1.9)."""
    return onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def load_scrfd(
    detector_path: str | Path, det_score_threshold: float = 0.5, det_input_size: int = 640
) -> SCRFD:
    """Carga SCRFD (`det_10g.onnx`) UNA sola vez. Reutilizar el objeto en cada llamada."""
    detector = SCRFD(model_file=str(detector_path), session=make_session(detector_path))
    detector.prepare(
        ctx_id=-1,  # CPU
        input_size=(det_input_size, det_input_size),
        det_thresh=det_score_threshold,
    )
    return detector


def detect_raw(detector: SCRFD, image_bgr: np.ndarray, pad_frac: float = 0.0) -> list[RawDetection]:
    """Detecta rostros y devuelve caja, score y 5 landmarks, ordenados por score descendente.

    pad_frac > 0 agrega un borde negro de `pad_frac` * (alto/ancho) a cada lado ANTES de detectar.
    SCRFD se entrenó con fotos que tienen contexto alrededor de la cara y falla en recortes ajustados
    donde el rostro llena todo el cuadro (p. ej. el dataset Pins). Las coordenadas devueltas se
    corrigen para que sigan siendo las de la imagen ORIGINAL (las cajas pueden salirse un poco de ella).
    """
    px = py = 0
    if pad_frac > 0:
        h, w = image_bgr.shape[:2]
        py, px = int(round(h * pad_frac)), int(round(w * pad_frac))
        image_bgr = cv2.copyMakeBorder(image_bgr, py, py, px, px, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    bboxes, kpss = detector.detect(image_bgr, max_num=0, metric="default")
    if bboxes is None or len(bboxes) == 0:
        return []
    offset = np.array([px, py], dtype=np.float32)
    out: list[RawDetection] = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2, score = (float(v) for v in bboxes[i])
        kps = np.asarray(kpss[i], dtype=np.float32) if kpss is not None else np.zeros((5, 2), np.float32)
        out.append(RawDetection(box=(x1 - px, y1 - py, x2 - px, y2 - py), score=score, kps=kps - offset))
    return sorted(out, key=lambda d: d.score, reverse=True)


def detect_with_fallback(detector: SCRFD, image_bgr: np.ndarray, pad_frac: float = 0.5) -> list[RawDetection]:
    """Detección normal; si no encuentra nada, reintenta con borde (para recortes ajustados de rostro)."""
    dets = detect_raw(detector, image_bgr)
    if not dets and pad_frac > 0:
        dets = detect_raw(detector, image_bgr, pad_frac=pad_frac)
    return dets


def iou_xyxy(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over union entre dos cajas (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def align_with_landmarks(image_bgr: np.ndarray, kps: np.ndarray, face_size: int = 112) -> np.ndarray:
    """Recorte alineado `face_size x face_size` (BGR uint8) usando los 5 landmarks."""
    return norm_crop(image_bgr, np.asarray(kps, dtype=np.float32), image_size=face_size)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / max(float(np.linalg.norm(x)), eps)


def preprocess_aligned_face(face_bgr: np.ndarray, input_size: int = 112) -> np.ndarray:
    """BGR uint8 (H, W, 3) -> tensor float32 (1, 3, S, S) normalizado como ArcFace."""
    if face_bgr.shape[:2] != (input_size, input_size):
        face_bgr = cv2.resize(face_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA)
    rgb = face_bgr[:, :, ::-1].astype(np.float32)
    rgb = (rgb - ARCFACE_MEAN) / ARCFACE_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def model_input_size(session: onnxruntime.InferenceSession) -> int:
    """Lado de entrada que espera el modelo, leído del propio ONNX (no hardcodeado)."""
    shape = session.get_inputs()[0].shape  # [N, 3, H, W]
    return int(shape[2]) if isinstance(shape[2], int) else 112


def embed_onnx(session: onnxruntime.InferenceSession, face_bgr: np.ndarray) -> np.ndarray:
    """Embedding L2-normalizado (float32, 1-D) de un rostro ya alineado."""
    size = model_input_size(session)
    tensor = preprocess_aligned_face(face_bgr, size)
    out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return l2_normalize(np.asarray(out, dtype=np.float32).reshape(-1))
