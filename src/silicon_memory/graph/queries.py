"""Graph query builder for fluent graph traversal."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from time import perf_counter
from typing import TYPE_CHECKING, Callable
from uuid import UUID

from silicon_memory.core.types import Belief
from silicon_memory.graph.types import (
    GraphNode,
    GraphEdge,
    GraphPath,
    NodeType,
    QueryResult,
    TraversalDirection,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory


class GraphQueryBuilder:
    """Fluent API for building graph queries.

    Example:
        >>> query = (GraphQuery(memory)
        ...     .start("Python")
        ...     .traverse("is_used_for", depth=2)
        ...     .filter(min_confidence=0.7)
        ...     .limit(20))
        >>> results = await query.execute()
    """

    def __init__(self, memory: "SiliconMemory") -> None:
        self._memory = memory
        self._start_entities: list[str] = []
        self._edge_types: list[str] | None = None
        self._direction: TraversalDirection = TraversalDirection.OUTGOING
        self._max_depth: int = 1
        self._min_confidence: float = 0.0
        self._node_types: list[NodeType] | None = None
        self._limit: int = 100
        self._include_paths: bool = False
        self._custom_filter: Callable[[GraphNode], bool] | None = None

    def start(self, *entities: str) -> "GraphQueryBuilder":
        """Set starting entities for traversal.

        Args:
            *entities: One or more entity names to start from

        Returns:
            Self for chaining
        """
        self._start_entities = list(entities)
        return self

    def traverse(
        self,
        edge_type: str | list[str] | None = None,
        direction: TraversalDirection | str = TraversalDirection.OUTGOING,
        depth: int = 1,
    ) -> "GraphQueryBuilder":
        """Configure traversal parameters.

        Args:
            edge_type: Type(s) of edges to follow (None = all)
            direction: Direction to traverse
            depth: Maximum traversal depth

        Returns:
            Self for chaining
        """
        if edge_type is not None:
            self._edge_types = [edge_type] if isinstance(edge_type, str) else edge_type

        if isinstance(direction, str):
            direction = TraversalDirection(direction.lower())
        self._direction = direction

        self._max_depth = depth
        return self

    def filter(
        self,
        min_confidence: float | None = None,
        node_types: list[NodeType] | None = None,
        custom: Callable[[GraphNode], bool] | None = None,
    ) -> "GraphQueryBuilder":
        """Apply filters to results.

        Args:
            min_confidence: Minimum confidence threshold
            node_types: Types of nodes to include
            custom: Custom filter function

        Returns:
            Self for chaining
        """
        if min_confidence is not None:
            self._min_confidence = min_confidence
        if node_types is not None:
            self._node_types = node_types
        if custom is not None:
            self._custom_filter = custom
        return self

    def limit(self, n: int) -> "GraphQueryBuilder":
        """Limit number of results.

        Args:
            n: Maximum number of results

        Returns:
            Self for chaining
        """
        self._limit = n
        return self

    def with_paths(self) -> "GraphQueryBuilder":
        """Include full paths in results.

        Returns:
            Self for chaining
        """
        self._include_paths = True
        return self

    async def execute(self) -> QueryResult:
        """Execute the query.

        Returns:
            QueryResult with matching nodes, edges, and paths
        """
        start_time = perf_counter()

        result = QueryResult()

        if not self._start_entities:
            result.query_time_ms = (perf_counter() - start_time) * 1000
            return result

        # Build graph from beliefs
        nodes, edges = await self._build_graph()

        # Perform BFS traversal
        visited: set[str] = set()
        result_nodes: list[GraphNode] = []
        result_edges: list[GraphEdge] = []
        paths: list[GraphPath] = []

        for start_entity in self._start_entities:
            if start_entity not in nodes:
                continue

            # BFS with depth tracking
            queue: deque[tuple[str, int, GraphPath]] = deque()
            start_path = GraphPath(nodes=[nodes[start_entity]], edges=[])
            queue.append((start_entity, 0, start_path))

            while queue and len(result_nodes) < self._limit:
                current_id, depth, current_path = queue.popleft()

                if current_id in visited:
                    continue
                visited.add(current_id)

                current_node = nodes.get(current_id)
                if current_node and self._passes_filter(current_node):
                    result_nodes.append(current_node)

                    if self._include_paths and current_path.length > 0:
                        paths.append(current_path)

                # Continue traversal if within depth
                if depth < self._max_depth:
                    neighbors = self._get_neighbors(current_id, edges)
                    for neighbor_id, edge in neighbors:
                        if neighbor_id not in visited:
                            neighbor_node = nodes.get(neighbor_id)
                            if neighbor_node:
                                new_path = GraphPath(
                                    nodes=current_path.nodes + [neighbor_node],
                                    edges=current_path.edges + [edge],
                                )
                                queue.append((neighbor_id, depth + 1, new_path))

                                # Track edges
                                if edge not in result_edges:
                                    result_edges.append(edge)

        result.nodes = result_nodes[:self._limit]
        result.edges = result_edges
        result.paths = paths
        result.total_count = len(result_nodes)
        result.query_time_ms = (perf_counter() - start_time) * 1000

        return result

    async def _build_graph(self) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
        """Build graph from beliefs."""
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []

        # Query beliefs for each starting entity
        for entity in self._start_entities:
            beliefs = await self._memory._backend.get_beliefs_by_entity(entity)

            for belief in beliefs:
                if belief.confidence < self._min_confidence:
                    continue

                if belief.triplet:
                    # Add subject node
                    subj = belief.triplet.subject
                    if subj not in nodes:
                        nodes[subj] = GraphNode(
                            id=subj,
                            label=subj,
                            type=NodeType.ENTITY,
                            confidence=belief.confidence,
                            belief_id=belief.id,
                        )

                    # Add object node
                    obj = belief.triplet.object
                    if obj not in nodes:
                        nodes[obj] = GraphNode(
                            id=obj,
                            label=obj,
                            type=NodeType.ENTITY,
                            confidence=belief.confidence,
                        )

                    # Add edge
                    edge = GraphEdge(
                        source=subj,
                        target=obj,
                        type=belief.triplet.predicate,
                        confidence=belief.confidence,
                        metadata={"belief_id": str(belief.id)},
                    )
                    edges.append(edge)

        return nodes, edges

    def _get_neighbors(
        self,
        node_id: str,
        edges: list[GraphEdge],
    ) -> list[tuple[str, GraphEdge]]:
        """Get neighboring nodes based on direction and edge type."""
        neighbors: list[tuple[str, GraphEdge]] = []

        for edge in edges:
            # Check edge type filter
            if self._edge_types and edge.type not in self._edge_types:
                continue

            # Check direction
            if self._direction in (TraversalDirection.OUTGOING, TraversalDirection.BOTH):
                if edge.source == node_id:
                    neighbors.append((edge.target, edge))

            if self._direction in (TraversalDirection.INCOMING, TraversalDirection.BOTH):
                if edge.target == node_id:
                    neighbors.append((edge.source, edge))

        return neighbors

    def _passes_filter(self, node: GraphNode) -> bool:
        """Check if a node passes all filters."""
        # Confidence check
        if node.confidence < self._min_confidence:
            return False

        # Node type check
        if self._node_types and node.type not in self._node_types:
            return False

        # Custom filter
        if self._custom_filter and not self._custom_filter(node):
            return False

        return True


# Alias for convenience
GraphQuery = GraphQueryBuilder


class ShortestPathFinder:
    """Find shortest paths between entities."""

    def __init__(self, memory: "SiliconMemory") -> None:
        self._memory = memory

    async def find_path(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
        min_confidence: float = 0.3,
    ) -> GraphPath | None:
        """Find shortest path between two entities.

        Args:
            start: Starting entity
            end: Target entity
            max_depth: Maximum path length
            min_confidence: Minimum confidence for edges

        Returns:
            GraphPath if found, None otherwise
        """
        # BFS for shortest path
        query = (GraphQuery(self._memory)
            .start(start)
            .traverse(depth=max_depth)
            .filter(min_confidence=min_confidence)
            .with_paths())

        result = await query.execute()

        # Find path ending at target
        for path in result.paths:
            if path.end_node and path.end_node.id == end:
                return path

        return None

    async def find_all_paths(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
        min_confidence: float = 0.3,
        limit: int = 10,
    ) -> list[GraphPath]:
        """Find all paths between two entities up to max_depth.

        Args:
            start: Starting entity
            end: Target entity
            max_depth: Maximum path length
            min_confidence: Minimum confidence for edges
            limit: Maximum paths to return

        Returns:
            List of paths sorted by length
        """
        query = (GraphQuery(self._memory)
            .start(start)
            .traverse(depth=max_depth)
            .filter(min_confidence=min_confidence)
            .with_paths()
            .limit(1000))  # Get more results to find multiple paths

        result = await query.execute()

        # Find all paths ending at target
        paths = [
            path for path in result.paths
            if path.end_node and path.end_node.id == end
        ]

        # Sort by length, then confidence
        paths.sort(key=lambda p: (p.length, -p.total_confidence))

        return paths[:limit]
