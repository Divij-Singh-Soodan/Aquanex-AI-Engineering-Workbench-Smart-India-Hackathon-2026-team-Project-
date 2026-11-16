# ---------------------------------------------------------------------------
# Force offline model loading when settings are imported outside app.main.
# ---------------------------------------------------------------------------
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# ---------------------------------------------------------------------------

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# RTX 3050 (6GB) laptop tier. Larger windows spill Ollama layers onto CPU.
EDGE_LAPTOP_CONTEXT_WINDOW = 4096
EDGE_LAPTOP_DEFAULT_MODEL = "qwen2.5:3b"


class Settings(BaseSettings):
    SERVER_NODE_IP: str = "192.168.29.253"
    ADMIN_NODE_IP: str = "192.168.29.253"
    USER_NODE_IP: str = "192.168.29.253"
    OLLAMA_HOST: str = "http://127.0.0.1:11434"
    VLLM_HOST: str = "http://127.0.0.1:8001/v1"
    API_BASE_URL: str = "http://127.0.0.1:8000"
    ALLOW_EMULATION: bool = False
    MODEL_TIER: str = "EDGE_LAPTOP_8B"
    TENSOR_PARALLEL_SIZE: int = 1
    MAX_MODEL_LEN: int = EDGE_LAPTOP_CONTEXT_WINDOW
    MODEL_ROUTER: str = EDGE_LAPTOP_DEFAULT_MODEL
    MODEL_REASONING: str = EDGE_LAPTOP_DEFAULT_MODEL
    MODEL_VISION: str = "deepseek-r1:8b"
    MODEL_CODER: str = "deepseek-r1:8b"

    @property
    def active_model_profile(self) -> dict:
        profiles = {
            "EDGE_LAPTOP_8B": {
                "tier_name": "Hackathon Demo Rig / Field Laptop (8B)",
                "hardware_spec": "Intel i7 / Ryzen 7, RTX 3050/3060/4060 (6-8GB) or Apple Silicon M-Series",
                "router": EDGE_LAPTOP_DEFAULT_MODEL,
                "reasoning": EDGE_LAPTOP_DEFAULT_MODEL,
                "vision": "deepseek-r1:8b",
                "coder": "deepseek-r1:8b",
                "tensor_parallel_size": 1,
                "context_window": EDGE_LAPTOP_CONTEXT_WINDOW,
                "vram_allocation_gb": 6,
            },
        }
        profile = dict(profiles.get(self.MODEL_TIER, profiles["EDGE_LAPTOP_8B"]))
        if self.MODEL_TIER == "EDGE_LAPTOP_8B":
            profile["context_window"] = min(int(profile["context_window"]), EDGE_LAPTOP_CONTEXT_WINDOW)
        return profile

    @model_validator(mode="after")
    def cap_edge_laptop_context(self):
        """Keep the 8B laptop tier inside the RTX 3050 context budget."""
        if self.MODEL_TIER == "EDGE_LAPTOP_8B" and self.MAX_MODEL_LEN > EDGE_LAPTOP_CONTEXT_WINDOW:
            self.MAX_MODEL_LEN = EDGE_LAPTOP_CONTEXT_WINDOW
        return self

    DATA_DIR: Path = Path("data")
    MODELS_DIR: Path = Path("models_storage")
    RAG_DOCS_DIR: Path = Path("rag_docs")
    SANDBOX_TIMEOUT_SECONDS: int = 5
    SANDBOX_MEMORY_LIMIT_MB: int = 256
    SANDBOX_MAX_RETRIES: int = 3
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: Path = Path("data/chromadb")
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
    RAG_TOP_K: int = 5
    VAULT_INPUT_DIR: Path = Path("data/vault")
    VAULT_STAGING_DIR: Path = Path("data/staging")
    INTERNAL_MAIL_DOMAIN: str = "mrpl.co.in"
    APPROVAL_TOKEN_SECRET: str = "mrpl_sovereign_approval_secret_2026"
    APPROVAL_TOKEN_TTL_SECONDS: int = 300
    AUDIT_DB_PATH: Path = Path("data/mrpl_audit.db")
    NETWORK_POLL_INTERVAL_SECONDS: float = 2.0
    AIR_GAP_STRICT: bool = True
    SIMULATE_AIR_GAP_BREACH: bool = False
    WAN_INTERFACE_OVERRIDE: str = ""
    ALLOW_UNSANDBOXED_FALLBACK: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = [
        "*",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://192.168.29.253:5173",
        "http://192.168.29.253:8000",
    ]
    model_config = SettingsConfigDict(env_file=[".env", "backend/.env"], env_file_encoding="utf-8", extra="ignore")

settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "uploads").mkdir(exist_ok=True)
(settings.DATA_DIR / "deliverables").mkdir(exist_ok=True)
settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)
settings.VAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.VAULT_STAGING_DIR.mkdir(parents=True, exist_ok=True)
