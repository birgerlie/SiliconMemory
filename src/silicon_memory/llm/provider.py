"""OpenAI v1 compatible LLM provider targeting SiliconServe."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from silicon_memory.llm.config import LLMConfig

logger = logging.getLogger(__name__)


class SiliconLLMProvider:
    """LLMProvider using any OpenAI v1 compatible API.

    Works with SiliconServe (localhost:8000), OpenAI, or any
    compatible endpoint. Implements the LLMProvider protocol
    from core/protocols.py.

    Embedding methods raise NotImplementedError — SiliconDB handles
    all embeddings internally with its built-in E5 model.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._config = config or LLMConfig()
        if base_url is not None:
            self._config.base_url = base_url
        if model is not None:
            self._config.model = model
        if api_key is not None:
            self._config.api_key = api_key

        self._client = AsyncOpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            timeout=self._config.timeout,
        )

    @property
    def model(self) -> str:
        return self._config.model

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        """Generate text from a prompt."""
        temp = temperature if temperature is not None else self._config.temperature
        response = await self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temp,
        )
        return response.choices[0].message.content or ""

    async def generate_structured(
        self,
        prompt: str,
        schema: type,
        max_tokens: int | None = None,
    ) -> Any:
        """Generate structured output matching a JSON schema.

        Instructs the model to respond with JSON matching the schema,
        then parses the response. Falls back to raw JSON parsing if
        the model doesn't support structured output natively.
        """
        schema_dict = _extract_schema(schema)
        system_msg = (
            "You must respond with valid JSON matching this schema. "
            "No markdown, no explanation, just the JSON object.\n\n"
            f"Schema: {json.dumps(schema_dict, indent=2)}"
        )

        response = await self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=0.3,
        )

        raw = response.choices[0].message.content or "{}"
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        parsed = _parse_json_lenient(raw)

        # If schema is a Pydantic model, validate through it
        if hasattr(schema, "model_validate"):
            return schema.model_validate(parsed)

        return parsed

    async def embed(self, text: str) -> list[float]:
        """Not supported — SiliconDB handles embeddings internally."""
        raise NotImplementedError(
            "SiliconDB handles embeddings internally via its built-in E5 model. "
            "Use SiliconDB's search/query methods instead."
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Not supported — SiliconDB handles embeddings internally."""
        raise NotImplementedError(
            "SiliconDB handles embeddings internally via its built-in E5 model. "
            "Use SiliconDB's search/query methods instead."
        )


_CLASSIFY_PROMPT = """\
Classify this text into exactly one category:

- **belief**: A general factual claim, opinion, or universal statement. Timeless knowledge that can be true or false. Examples: "Python is dynamically typed", "PostgreSQL supports JSON", "Microservices increase operational complexity".
- **experience**: Something that happened — an event, meeting, debugging session, observation, interaction, decision, or narrative. Usually has a specific time/context. Examples: "We had a meeting about X", "Deployed the service and it was slow", "Found a bug in the auth module".
- **procedure**: Instructions, steps, a how-to, or a process for doing something. Examples: "To deploy: 1) build, 2) push, 3) verify", "Run pytest with -v flag".

If the text describes something that happened or was observed, classify it as **experience** even if it contains factual claims within the narrative.

Text: {content}

Respond with a single JSON object: {{"type": "belief" | "experience" | "procedure", "confidence": 0.0-1.0}}"""


async def classify_memory_type(
    llm: "SiliconLLMProvider", content: str
) -> tuple[str, float]:
    """Use the LLM to classify content as belief/experience/procedure.

    Returns (type, confidence).
    """
    from pydantic import BaseModel

    class Classification(BaseModel):
        type: str
        confidence: float

    try:
        result = await llm.generate_structured(
            _CLASSIFY_PROMPT.format(content=content[:500]),
            Classification,
        )
        if result.type in ("belief", "experience", "procedure"):
            return result.type, result.confidence
    except Exception:
        logger.warning("LLM classification failed, defaulting to belief")

    return "belief", 0.5


def _parse_json_lenient(raw: str) -> Any:
    """Parse JSON, attempting repair if truncated by max_tokens."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to repair truncated JSON by closing open structures
    repaired = raw.rstrip()
    # Remove trailing incomplete string (unterminated quote)
    if repaired.count('"') % 2 != 0:
        last_quote = repaired.rfind('"')
        repaired = repaired[:last_quote] + '"'

    # Close open brackets/braces
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    # Remove trailing comma before closing
    repaired = repaired.rstrip().rstrip(",")

    repaired += "]" * open_brackets + "}" * open_braces

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Last resort: find the last valid JSON object/array boundary
    for end in range(len(raw), 0, -1):
        candidate = raw[:end]
        open_b = candidate.count("{") - candidate.count("}")
        open_k = candidate.count("[") - candidate.count("]")
        candidate = candidate.rstrip().rstrip(",")
        candidate += "]" * open_k + "}" * open_b
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return json.loads(raw)  # Raise original error


def _extract_schema(schema: type) -> dict[str, Any]:
    """Extract a JSON schema dict from a type."""
    # Pydantic model
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    # dataclass or other — return a hint
    if hasattr(schema, "__dataclass_fields__"):
        fields = {}
        for name, f in schema.__dataclass_fields__.items():
            fields[name] = {"type": "string"}
        return {"type": "object", "properties": fields}
    return {"type": "object"}
