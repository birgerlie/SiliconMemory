"""Configuration for the LLM provider."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for the OpenAI-compatible LLM provider.

    Defaults target SiliconServe running locally on port 8000.
    """

    base_url: str = "http://localhost:8000/v1"
    model: str = "qwen3-4b"
    api_key: str = "not-needed"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 600.0
    max_concurrency: int = 8
    max_queue_size: int = 100
    max_wait_seconds: float = 30.0
