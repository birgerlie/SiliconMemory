"""Content ingestion endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from silicon_memory.llm.scheduler import LLMScheduler
from silicon_memory.memory.silicondb_router import SiliconMemory
from silicon_memory.server.dependencies import get_memory, get_scheduler
from silicon_memory.server.errors import IngestionError
from silicon_memory.server.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Map source_type names to adapter classes
_ADAPTER_REGISTRY: dict[str, str] = {
    "meeting": "silicon_memory.ingestion.meeting.MeetingAdapter",
    "chat": "silicon_memory.ingestion.chat.ChatAdapter",
    "email": "silicon_memory.ingestion.email.EmailAdapter",
    "document": "silicon_memory.ingestion.document.DocumentAdapter",
    "news": "silicon_memory.ingestion.news.NewsAdapter",
}


def _load_adapter(source_type: str):
    """Dynamically load an ingestion adapter by source type."""
    module_path = _ADAPTER_REGISTRY.get(source_type)
    if not module_path:
        raise IngestionError(f"Unknown source type: {source_type}")

    module_name, class_name = module_path.rsplit(".", 1)
    try:
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()
    except (ImportError, AttributeError) as e:
        raise IngestionError(f"Failed to load adapter for '{source_type}': {e}")


@router.post("/ingest")
async def ingest(
    body: IngestRequest,
    memory: SiliconMemory = Depends(get_memory),
    scheduler: LLMScheduler = Depends(get_scheduler),
) -> IngestResponse:
    adapter = _load_adapter(body.source_type)
    result = await memory.ingest_from(
        adapter=adapter,
        content=body.content,
        metadata=body.metadata,
        llm_provider=scheduler,
    )

    return IngestResponse(
        experiences_created=result.experiences_created,
        entities_resolved=result.entities_resolved,
        decisions_detected=result.decisions_detected,
        action_items_detected=result.action_items_detected,
        errors=result.errors,
        source_type=result.source_type,
    )
