"""Modal deployment entrypoint for the UG Handbook RAG API."""

from __future__ import annotations

import os
import shutil
import sys

import modal


APP_NAME = "ug-handbook-rag"
APP_ROOT = "/root/rag_api"
SEED_CHROMA_DIR = "/root/seed_chroma_db"
SEED_DOCS_DIR = "/root/seed_documents"
DATA_VOLUME_NAME = "ug-handbook-rag-data"
DATA_VOLUME_PATH = "/vol/data"


data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

# Holds the GEMINI_API_KEY. Create once with:
#   modal secret create gemini-api-key GEMINI_API_KEY=<your-key>
gemini_secret = modal.Secret.from_name("gemini-api-key")


# Mount the application source. These are the files actually needed at runtime
# plus the offline chunker scripts so you can re-index on the volume if needed.
APP_FILES = [
    "api.py",
    "rag_chat.py",
    "document_context.py",
    "document_files.py",
    "transcript.py",
    "cv.py",
    "speech.py",
    "directions.py",
    "chunker.py",
    "policy_chunker.py",
    "contextualize.py",
    "chunk_all.py",
    "embed_and_load.py",
]


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "beautifulsoup4",
        "chromadb",
        "fastapi[standard]",
        "google-auth",
        "google-genai",
        "numpy",
        "pillow",
        "pydantic",
        "python-multipart",
        "rank-bm25",
        "sentence-transformers",
        "torch",
        "uvicorn[standard]",
    )
)

for _f in APP_FILES:
    image = image.add_local_file(_f, remote_path=f"{APP_ROOT}/{_f}")

# Data files referenced at runtime (not Python source)
image = image.add_local_file("ug_urls.json", remote_path=f"{APP_ROOT}/ug_urls.json")

image = image.add_local_dir("chroma_db", remote_path=SEED_CHROMA_DIR)
image = image.add_local_dir("documents", remote_path=SEED_DOCS_DIR)


app = modal.App(APP_NAME, image=image)


def _configure_runtime_env() -> None:
    """Point the FastAPI app at Modal-mounted persistent storage."""
    os.environ["CHROMA_DIR"] = f"{DATA_VOLUME_PATH}/chroma_db"
    os.environ["DOCS_DIR"] = f"{DATA_VOLUME_PATH}/documents"
    os.environ.setdefault("HF_HOME", f"{DATA_VOLUME_PATH}/hf-home")
    os.environ.setdefault("HF_HUB_CACHE", f"{DATA_VOLUME_PATH}/hf-home/hub")
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        f"{DATA_VOLUME_PATH}/sentence-transformers",
    )
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", f"{DATA_VOLUME_PATH}/hf-home/transformers"
    )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for key in (
        "CHROMA_DIR",
        "DOCS_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "TRANSFORMERS_CACHE",
    ):
        os.makedirs(os.environ[key], exist_ok=True)

    if APP_ROOT not in sys.path:
        sys.path.insert(0, APP_ROOT)


def _seed_chroma(force: bool = False) -> bool:
    """Copy the bundled ChromaDB into the persistent volume on first run."""
    dst_dir = os.environ["CHROMA_DIR"]

    if force and os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    # Treat a pre-existing dir with any sqlite files as already-seeded.
    if os.path.exists(dst_dir) and any(
        f.endswith(".sqlite3") for f in os.listdir(dst_dir)
    ):
        return False

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(SEED_CHROMA_DIR, dst_dir)
    return True


def _seed_documents(force: bool = False) -> list[str]:
    """Copy source markdown documents onto the volume (for re-chunking)."""
    dst_dir = os.environ["DOCS_DIR"]
    os.makedirs(dst_dir, exist_ok=True)

    copied: list[str] = []
    for filename in sorted(os.listdir(SEED_DOCS_DIR)):
        src = os.path.join(SEED_DOCS_DIR, filename)
        dst = os.path.join(dst_dir, filename)
        if not force and os.path.exists(dst):
            continue
        if os.path.isdir(src):
            continue
        shutil.copy2(src, dst)
        copied.append(filename)
    return copied


@app.function(
    volumes={DATA_VOLUME_PATH: data_volume},
    secrets=[gemini_secret],
    timeout=1800,
)
def prepare_data(
    seed_chroma: bool = True,
    seed_docs: bool = True,
    force_seed: bool = False,
):
    """One-off utility to (re)seed the volume from the bundled image data."""
    _configure_runtime_env()
    chroma_seeded = _seed_chroma(force=force_seed) if seed_chroma else False
    copied_docs = _seed_documents(force=force_seed) if seed_docs else []

    import api as rag_api

    data_volume.commit()
    return {
        "chroma_dir": os.environ["CHROMA_DIR"],
        "docs_dir": os.environ["DOCS_DIR"],
        "chroma_seeded": chroma_seeded,
        "docs_copied": copied_docs,
        "chunks_indexed": rag_api.rag.collection.count() if rag_api.rag else None,
    }


@app.function(
    cpu=4.0,
    memory=16384,
    volumes={DATA_VOLUME_PATH: data_volume},
    secrets=[gemini_secret],
    timeout=1800,
    min_containers=1,
    scaledown_window=900,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI application on Modal."""
    _configure_runtime_env()
    seeded = _seed_chroma()
    copied_docs = _seed_documents()
    if seeded or copied_docs:
        data_volume.commit()

    import api as rag_api

    return rag_api.app
