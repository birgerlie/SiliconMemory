"""Graph Query Layer - Relationship traversal and entity exploration.

Provides a fluent API for querying relationships between entities
and exploring the knowledge graph.

Example:
    >>> from silicon_memory import SiliconMemory
    >>> from silicon_memory.graph import GraphQuery, EntityExplorer
    >>>
    >>> with SiliconMemory("/path/to/db") as memory:
    ...     # Fluent query API
    ...     results = await (GraphQuery(memory)
    ...         .start("Python")
    ...         .traverse("is_used_for", depth=2)
    ...         .filter(min_confidence=0.7)
    ...         .execute())
    ...
    ...     # Entity exploration
    ...     explorer = EntityExplorer(memory)
    ...     profile = await explorer.explore("Python")
    ...     print(profile.summary())
"""

from silicon_memory.graph.types import (
    GraphNode,
    GraphEdge,
    GraphPath,
    EntityProfile,
    TraversalDirection,
)
from silicon_memory.graph.queries import GraphQuery, GraphQueryBuilder
from silicon_memory.graph.explorer import EntityExplorer
from silicon_memory.graph.tool import GraphTool

__all__ = [
    # Types
    "GraphNode",
    "GraphEdge",
    "GraphPath",
    "EntityProfile",
    "TraversalDirection",
    # Query builder
    "GraphQuery",
    "GraphQueryBuilder",
    # Explorer
    "EntityExplorer",
    # Tool
    "GraphTool",
]
