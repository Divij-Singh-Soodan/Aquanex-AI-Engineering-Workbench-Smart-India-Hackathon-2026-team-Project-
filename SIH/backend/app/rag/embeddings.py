"""
CPU-pinned local embeddings.

Ollama owns the RTX 3050 6GB pool. Every in-process embedding session is forced
onto CPU so ONNX Runtime and PyTorch cannot allocate that VRAM.
"""

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# SentenceTransformers / HuggingFace device string. Never "cuda" or "auto".
EMBEDDING_DEVICE = "cpu"


def pin_onnx_runtime_to_cpu() -> None:
    """Force every ONNX Runtime session in this process onto CPUExecutionProvider."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    if getattr(ort, "_aquanex_cpu_pinned", False):
        return

    original = ort.InferenceSession

    def cpu_only_session(*args: Any, **kwargs: Any):
        kwargs["providers"] = ["CPUExecutionProvider"]
        return original(*args, **kwargs)

    ort.InferenceSession = cpu_only_session  # type: ignore[misc, assignment]
    ort._aquanex_cpu_pinned = True  # type: ignore[attr-defined]


def pin_torch_to_cpu() -> None:
    """Force PyTorch device defaults to CPU to avoid allocating VRAM on RTX 3050."""
    try:
        import torch
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass


def pin_all_embeddings_to_cpu() -> None:
    """Ensure all local inference engines (ONNX, PyTorch) strictly run on CPU."""
    pin_onnx_runtime_to_cpu()
    pin_torch_to_cpu()


def load_local_embedder(model_name: str | None = None):
    """
    Load the local embedding model strictly on CPU.

    FastEmbed has no ``device`` argument; ``cuda=Device.CPU`` plus
    ``CPUExecutionProvider`` is its CPU pin. SentenceTransformers receives
    ``device="cpu"`` directly. All model loaders use local files only.
    """
    pin_all_embeddings_to_cpu()
    name = model_name or settings.EMBEDDING_MODEL
    device = EMBEDDING_DEVICE

    # Explicit torch.device validation if PyTorch is installed
    try:
        import torch
        _ = torch.device("cpu")
    except Exception:
        pass

    try:
        from fastembed import TextEmbedding
        from fastembed.common.types import Device
    except ImportError:
        logger.debug("FastEmbed is unavailable; trying sentence-transformers")
    else:
        # FastEmbed registers the default model under its qualified name.
        fastembed_name = "sentence-transformers/all-MiniLM-L6-v2" if name == "all-MiniLM-L6-v2" else name
        embedder = TextEmbedding(
            model_name=fastembed_name,
            cuda=Device.CPU,
            providers=["CPUExecutionProvider"],
            local_files_only=True,
        )
        logger.info("Loaded FastEmbed %s on device=%s", fastembed_name, device)
        return embedder

    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(name, device=device, local_files_only=True)
        logger.info("Loaded SentenceTransformer %s on device=%s (local_files_only=True)", name, device)
        return embedder
    except ImportError:
        logger.debug("sentence-transformers is not installed; trying HuggingFaceEmbeddings")

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embedder = HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": device, "local_files_only": True},
            encode_kwargs={"device": device},
        )
        logger.info("Loaded HuggingFaceEmbeddings %s on device=%s (local_files_only=True)", name, device)
        return embedder
    except Exception as e:
        logger.error("Failed to load local embedder on device=%s: %s", device, e)
        raise


def embed_local_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts using the available local embedding implementation."""
    embedder = load_local_embedder()
    if hasattr(embedder, "embed"):
        return [vector.tolist() for vector in embedder.embed(texts)]
    if hasattr(embedder, "encode"):
        return embedder.encode(texts).tolist()
    return embedder.embed_documents(texts)
