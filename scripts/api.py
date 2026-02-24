"""
FastAPI server for MH-SIGNALS pipeline.

Run:
    uvicorn scripts.api:app --host 0.0.0.0 --port 8000
    # or with auto-reload during development:
    uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("MH_CONFIG", "configs/pipeline.yaml")

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from mhsignals import MHSignalsPipeline
        logger.info("Loading pipeline from %s …", CONFIG_PATH)
        _pipeline = MHSignalsPipeline.from_config(CONFIG_PATH)
        logger.info("Pipeline ready.")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_pipeline()
    yield


app = FastAPI(
    title="MH-SIGNALS API",
    description="Mental Health Signal Detection and Response Generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ── Request / Response schemas ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    post: str

    @field_validator("post")
    @classmethod
    def post_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("post cannot be empty")
        return v.strip()


class AnalyzeResponse(BaseModel):
    intents: List[str]
    concern: str
    crisis_level: str
    crisis_detected: bool
    reply: str
    disclaimer: str
    post_excerpt: str


# ── Helpers ─────────────────────────────────────────────────────────────────

def _excerpt(text: str, max_chars: int = 200) -> str:
    """Return first 1-2 sentences or max_chars of text."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    excerpt = sentences[0]
    if len(sentences) > 1 and len(excerpt) + len(sentences[1]) + 1 <= max_chars:
        excerpt += " " + sentences[1]
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit(" ", 1)[0] + "…"
    return excerpt


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    pipeline = _get_pipeline()
    try:
        resp = pipeline(req.post)
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))

    return AnalyzeResponse(
        intents=resp.intents,
        concern=resp.concern,
        crisis_level=resp.crisis_level,
        crisis_detected=resp.crisis_detected,
        reply=resp.reply,
        disclaimer=resp.disclaimer,
        post_excerpt=_excerpt(resp.post),
    )


# ── Static frontend ────────────────────────────────────────────────────────

if FRONTEND_DIR.is_dir():
    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
