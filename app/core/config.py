"""Server-side configuration; never populated from HTTP requests."""

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_TOP_K = 20
MAX_QUERY_CHARS = 1024


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    dataset_dir: Path = Path("data/public/scifact")
    dataset_label: str = Field(
        default="scifact", min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$"
    )
    dense_model_dir: Path = Path("models/dense")
    rerank_model_dir: Path = Path("models/reranker")
    dense_index: Path = Path("indexes/scifact-minilm.npz")
    policy_dir: Path = Path("experiments/configs/phase4-transfer")
    default_top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    default_method: Literal["bm25", "dense", "hybrid", "reranked", "router"] = "bm25"
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    batch_size: int = Field(default=16, ge=1, le=128)
    threads: int = Field(default=4, ge=1, le=64)
    candidate_depth: int = Field(default=50, ge=1, le=1000)
    rerank_candidates: int = Field(default=20, ge=1, le=100)
    max_body_bytes: int = Field(default=16384, ge=1024, le=65536)
    max_queued_requests: int = Field(default=2, ge=0, le=16)
    max_queue_wait_seconds: float = Field(default=2, ge=0, le=60, allow_inf_nan=False)
    host: str = Field(default="127.0.0.1", min_length=1, max_length=255)
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: tuple[str, ...] = ()

    @field_validator("cors_origins")
    @classmethod
    def exact_origins(cls, origins: tuple[str, ...]) -> tuple[str, ...]:
        for origin in origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
                or "*" in origin
            ):
                raise ValueError("CORS origins must be exact HTTP(S) origins")
        return tuple(origin.rstrip("/") for origin in origins)

    @model_validator(mode="after")
    def consistent_depth(self):
        if self.rerank_candidates > self.candidate_depth:
            raise ValueError("rerank_candidates must not exceed candidate_depth")
        return self

    @classmethod
    def from_env(cls):
        values = {
            name: os.environ["RANKFORGE_" + name.upper()]
            for name in cls.model_fields
            if "RANKFORGE_" + name.upper() in os.environ
        }
        if "cors_origins" in values:
            values["cors_origins"] = tuple(
                x.strip() for x in values["cors_origins"].split(",") if x.strip()
            )
        return cls(**values)
