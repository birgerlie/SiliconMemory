"""Server configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from silicon_memory.llm.config import LLMConfig


def _embedder_available() -> bool:
    """Check if sentence-transformers is installed for E5 embedder."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class ServerConfig:
    """Configuration for the Silicon Memory server."""

    # Network
    host: str = "0.0.0.0"
    port: int = 8420

    # Mode: "full" (REST + MCP + workers), "rest", "mcp"
    mode: str = "full"

    # Storage
    db_path: Path = field(default_factory=lambda: Path("./silicon_memory.db"))

    # LLM
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Reflection / dreaming
    reflect_interval: int = 1800  # seconds (30 min)
    reflect_max_experiences: int = 100
    reflect_auto_commit: bool = True

    # SiliconDB settings
    language: str = "english"
    enable_graph: bool = True
    auto_embedder: bool = field(default_factory=lambda: _embedder_available())
    embedder_model: str = "base"

    # CORS
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
