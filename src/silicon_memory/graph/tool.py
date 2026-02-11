"""Graph tool for LLM function calling integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from silicon_memory.core.utils import utc_now
from silicon_memory.graph.types import TraversalDirection
from silicon_memory.graph.queries import GraphQuery, ShortestPathFinder
from silicon_memory.graph.explorer import EntityExplorer

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class GraphAction(str, Enum):
    """Actions available through the graph tool."""

    TRAVERSE = "traverse"
    EXPLORE = "explore"
    FIND_PATH = "find_path"
    COMPARE = "compare"
    FIND_COMMON = "find_common"


@dataclass
class GraphToolResponse:
    """Response from graph tool invocation."""

    success: bool
    action: GraphAction
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM consumption."""
        result = {
            "success": self.success,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.data:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        return result


class GraphTool:
    """LLM-callable tool for graph operations.

    Allows LLMs to explore relationships between entities,
    find paths, and understand the knowledge structure.

    Example:
        >>> from silicon_memory import SiliconMemory
        >>> from silicon_memory.graph import GraphTool
        >>>
        >>> with SiliconMemory("/path/to/db") as memory:
        ...     tool = GraphTool(memory)
        ...     response = await tool.invoke("explore", entity="Python")
        ...     print(response.data["summary"])
    """

    def __init__(self, memory: "SiliconMemory") -> None:
        self._memory = memory
        self._explorer = EntityExplorer(memory)
        self._path_finder = ShortestPathFinder(memory)

    async def invoke(
        self,
        action: str,
        **kwargs: Any,
    ) -> GraphToolResponse:
        """Invoke the graph tool with an action.

        Args:
            action: The action to perform
            **kwargs: Action-specific parameters

        Returns:
            GraphToolResponse with results or error
        """
        try:
            action_enum = GraphAction(action.lower())
        except ValueError:
            return GraphToolResponse(
                success=False,
                action=GraphAction.TRAVERSE,
                error=f"Unknown action: {action}. Valid: {[a.value for a in GraphAction]}",
            )

        handlers = {
            GraphAction.TRAVERSE: self._handle_traverse,
            GraphAction.EXPLORE: self._handle_explore,
            GraphAction.FIND_PATH: self._handle_find_path,
            GraphAction.COMPARE: self._handle_compare,
            GraphAction.FIND_COMMON: self._handle_find_common,
        }

        handler = handlers[action_enum]
        try:
            return await handler(**kwargs)
        except Exception as e:
            return GraphToolResponse(
                success=False,
                action=action_enum,
                error=str(e),
            )

    async def _handle_traverse(
        self,
        entity: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
        depth: int = 2,
        min_confidence: float = 0.3,
        limit: int = 20,
        **_: Any,
    ) -> GraphToolResponse:
        """Handle traverse action."""
        query = (GraphQuery(self._memory)
            .start(entity)
            .filter(min_confidence=min_confidence)
            .limit(limit)
            .with_paths())

        if edge_type:
            query.traverse(edge_type=edge_type, direction=direction, depth=depth)
        else:
            query.traverse(direction=direction, depth=depth)

        result = await query.execute()

        return GraphToolResponse(
            success=True,
            action=GraphAction.TRAVERSE,
            data={
                "start_entity": entity,
                "node_count": len(result.nodes),
                "edge_count": len(result.edges),
                "path_count": len(result.paths),
                "query_time_ms": result.query_time_ms,
                "nodes": [
                    {
                        "id": n.id,
                        "label": n.label,
                        "confidence": n.confidence,
                    }
                    for n in result.nodes[:limit]
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "type": e.type,
                        "confidence": e.confidence,
                    }
                    for e in result.edges[:limit]
                ],
                "paths": [
                    {
                        "text": p.as_text(),
                        "length": p.length,
                        "confidence": p.total_confidence,
                    }
                    for p in result.paths[:10]
                ],
            },
        )

    async def _handle_explore(
        self,
        entity: str,
        depth: int = 2,
        min_confidence: float = 0.3,
        **_: Any,
    ) -> GraphToolResponse:
        """Handle explore action."""
        profile = await self._explorer.explore(
            entity,
            depth=depth,
            min_confidence=min_confidence,
        )

        return GraphToolResponse(
            success=True,
            action=GraphAction.EXPLORE,
            data={
                "entity": entity,
                "summary": profile.summary(),
                "total_mentions": profile.total_mentions,
                "average_confidence": profile.average_confidence,
                "beliefs": profile.direct_beliefs[:10],
                "related_entities": [
                    {"entity": e, "relation": r, "confidence": c}
                    for e, r, c in profile.related_entities[:10]
                ],
                "contradictions": [
                    {"belief1": b1, "belief2": b2}
                    for b1, b2 in profile.contradictions[:5]
                ],
                "outgoing_edges": len(profile.outgoing_edges),
                "incoming_edges": len(profile.incoming_edges),
            },
        )

    async def _handle_find_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
        min_confidence: float = 0.3,
        find_all: bool = False,
        **_: Any,
    ) -> GraphToolResponse:
        """Handle find_path action."""
        if find_all:
            paths = await self._path_finder.find_all_paths(
                start, end,
                max_depth=max_depth,
                min_confidence=min_confidence,
            )
            path_data = [
                {
                    "text": p.as_text(),
                    "length": p.length,
                    "confidence": p.total_confidence,
                }
                for p in paths
            ]
        else:
            path = await self._path_finder.find_path(
                start, end,
                max_depth=max_depth,
                min_confidence=min_confidence,
            )
            if path:
                path_data = [{
                    "text": path.as_text(),
                    "length": path.length,
                    "confidence": path.total_confidence,
                }]
            else:
                path_data = []

        return GraphToolResponse(
            success=True,
            action=GraphAction.FIND_PATH,
            data={
                "start": start,
                "end": end,
                "paths_found": len(path_data),
                "paths": path_data,
                "connected": len(path_data) > 0,
            },
        )

    async def _handle_compare(
        self,
        entity1: str,
        entity2: str,
        min_confidence: float = 0.3,
        **_: Any,
    ) -> GraphToolResponse:
        """Handle compare action."""
        comparison = await self._explorer.compare_entities(
            entity1, entity2,
            min_confidence=min_confidence,
        )

        return GraphToolResponse(
            success=True,
            action=GraphAction.COMPARE,
            data=comparison,
        )

    async def _handle_find_common(
        self,
        entities: list[str],
        max_depth: int = 3,
        min_confidence: float = 0.3,
        **_: Any,
    ) -> GraphToolResponse:
        """Handle find_common action."""
        common = await self._explorer.find_common_ancestors(
            entities,
            max_depth=max_depth,
            min_confidence=min_confidence,
        )

        return GraphToolResponse(
            success=True,
            action=GraphAction.FIND_COMMON,
            data={
                "entities": entities,
                "common_ancestors": common,
                "count": len(common),
            },
        )

    @staticmethod
    def get_openai_schema() -> dict[str, Any]:
        """Get OpenAI function calling schema for this tool."""
        return {
            "name": "graph",
            "description": (
                "Explore the knowledge graph. Traverse relationships, "
                "find paths between entities, and understand knowledge structure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [a.value for a in GraphAction],
                        "description": (
                            "The action to perform: "
                            "traverse (follow relationships from entity), "
                            "explore (get comprehensive entity profile), "
                            "find_path (find connection between entities), "
                            "compare (compare two entities), "
                            "find_common (find common ancestors)"
                        ),
                    },
                    "entity": {
                        "type": "string",
                        "description": "Entity to explore/traverse from",
                    },
                    "entity1": {
                        "type": "string",
                        "description": "First entity for comparison",
                    },
                    "entity2": {
                        "type": "string",
                        "description": "Second entity for comparison",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of entities for find_common",
                    },
                    "start": {
                        "type": "string",
                        "description": "Start entity for find_path",
                    },
                    "end": {
                        "type": "string",
                        "description": "End entity for find_path",
                    },
                    "edge_type": {
                        "type": "string",
                        "description": "Type of edges to follow (e.g., 'is_a', 'has')",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["outgoing", "incoming", "both"],
                        "description": "Direction to traverse (default: outgoing)",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum traversal depth (default: 2)",
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (default: 0.3)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 20)",
                    },
                    "find_all": {
                        "type": "boolean",
                        "description": "Find all paths instead of just shortest",
                    },
                },
                "required": ["action"],
            },
        }
