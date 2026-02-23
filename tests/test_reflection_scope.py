from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from silicon_memory.core.types import Belief
from silicon_memory.reflection.engine import ReflectionEngine
from silicon_memory.reflection.types import ReflectionConfig


class _BackendStub:
    NODE_TYPE_BELIEF = "belief"

    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.allowed_prefix = "tenant/user/"

    def _search_by_type(self, query: str, node_types: set[str], target: int) -> list[dict]:  # noqa: ARG002
        return list(self._results)

    @staticmethod
    def _rget(item: dict, key: str, default=None):  # noqa: ANN001
        return item.get(key, default)

    def _can_access(self, metadata: dict, _external_id: str = "") -> bool:  # noqa: ARG002
        return _external_id.startswith(self.allowed_prefix)

    def _search_result_to_belief(self, result: dict) -> Belief | None:
        metadata = result.get("metadata") or {}
        belief_id_raw = str(metadata.get("belief_id") or "").strip()
        if not belief_id_raw:
            return None
        return Belief(
            id=UUID(belief_id_raw),
            content=str(metadata.get("content") or ""),
            confidence=0.8,
            tags=set(metadata.get("tags") or []),
        )


@pytest.mark.asyncio
async def test_recent_extracted_beliefs_respect_access_scope() -> None:
    allowed_id = uuid4()
    blocked_id = uuid4()
    processed_id = uuid4()

    results = [
        {
            "external_id": f"tenant/user/belief-{allowed_id}",
            "metadata": {
                "belief_id": str(allowed_id),
                "tags": ["extracted"],
                "reflection_processed": False,
            },
        },
        {
            "external_id": f"other/other/belief-{blocked_id}",
            "metadata": {
                "belief_id": str(blocked_id),
                "tags": ["extracted"],
                "reflection_processed": False,
            },
        },
        {
            "external_id": f"tenant/user/belief-{processed_id}",
            "metadata": {
                "belief_id": str(processed_id),
                "tags": ["extracted"],
                "reflection_processed": True,
            },
        },
    ]
    backend = _BackendStub(results)
    memory = SimpleNamespace(_backend=backend)
    engine = ReflectionEngine(
        memory=memory,
        llm=object(),
        config=ReflectionConfig(empty_extracted_cooldown_s=0.0),
    )

    beliefs, source_external_ids = await engine._query_recent_extracted_beliefs(limit=10)  # noqa: SLF001

    assert [str(b.id) for b in beliefs] == [str(allowed_id)]
    assert source_external_ids == [f"tenant/user/belief-{allowed_id}"]
