from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
import logging
import os
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    cors_origins: str = Field(
        default="*",
        description="Comma-separated allowed origins, or * for any.",
    )
    app_name: str = "Facial Recognition TP1"
    model_name: str | None = os.getenv("MODEL_NAME")
    similarity_metric: str = os.getenv("SIMILARITY_METRIC", "cosine")
    similarity_threshold: float = os.getenv("SIMILARITY_THRESHOLD", 0.55)
    embeddings_path: Path = Path(os.getenv("EMBEDDINGS_PATH", "data/embeddings.json"))
    data_path: Path = Path(os.getenv("DATA_PATH", "data"))
    output_path: Path = Path(os.getenv("OUTPUT_PATH", "output"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "lib/models"))
    detector_model_name: str | None = os.getenv("DETECTOR_MODEL_NAME") or None
    det_score_threshold: float = os.getenv("DET_SCORE_THRESHOLD", 0.5)
    det_input_size: int = os.getenv("DET_INPUT_SIZE", 640)
    det_pad_frac: float = os.getenv("DET_PAD_FRAC", 0.5)
    min_face_size: int = os.getenv("MIN_FACE_SIZE", 20)
    align_margin: float = os.getenv("ALIGN_MARGIN", 0.5)
    max_workers: int = os.getenv("MAX_WORKERS", 2)
    face_size: int = os.getenv("FACE_SIZE", 112)
    embedding_dim: int = os.getenv("EMBEDDING_DIM", 512)
    use_pgvector: bool = os.getenv("USE_PGVECTOR", "True").lower() == "true"
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = os.getenv("POSTGRES_PORT", 5432)
    postgres_db: str = os.getenv("POSTGRES_DB", "vector_db")
    postgres_user: str = os.getenv("POSTGRES_USER", "user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "password")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
