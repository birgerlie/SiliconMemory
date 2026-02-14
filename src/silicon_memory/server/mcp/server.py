"""FastMCP server with all 10 memory tools."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID, uuid4

from mcp.server.fastmcp import FastMCP

from silicon_memory.core.decision import Alternative, Assumption, Decision
from silicon_memory.core.types import (
    Belief,
    Experience,
    Procedure,
    Source,
    SourceType,
    Triplet,
)
from silicon_memory.llm.provider import SiliconLLMProvider
from silicon_memory.llm.scheduler import LLMScheduler, Priority
from silicon_memory.memory.silicondb_router import RecallContext, SiliconMemory
from silicon_memory.security.types import UserContext
from silicon_memory.server.config import ServerConfig

logger = logging.getLogger(__name__)


def create_mcp_server(config: ServerConfig) -> FastMCP:
    """Create a FastMCP server with all Silicon Memory tools."""

    mcp = FastMCP(
        "Silicon Memory",
        instructions="Living knowledge network — store, recall, and reflect on knowledge",
    )

    # Shared state
    _llm = SiliconLLMProvider(config=config.llm)
    _scheduler = LLMScheduler(
        _llm,
        max_concurrency=config.llm.max_concurrency,
        max_queue_size=config.llm.max_queue_size,
        max_wait_seconds=config.llm.max_wait_seconds,
    )
    _scheduler_started = False
    _user_ctx = UserContext(user_id="default", tenant_id="default")
    _memory: SiliconMemory | None = None

    def _get_memory() -> SiliconMemory:
        nonlocal _memory
        if _memory is None:
            _memory = SiliconMemory(
                path=config.db_path,
                user_context=_user_ctx,
                language=config.language,
                enable_graph=config.enable_graph,
                auto_embedder=config.auto_embedder,
                embedder_model=config.embedder_model,
                llm_provider=_llm,
            )
        return _memory

    # ===== Tool 1: memory_recall =====

    @mcp.tool()
    async def memory_recall(
        query: str,
        max_facts: int = 20,
        max_experiences: int = 10,
        max_procedures: int = 5,
        min_confidence: float = 0.3,
    ) -> str:
        """Recall relevant memories across all layers — beliefs, experiences, procedures, and working memory.

        Use this as your primary way to retrieve context before responding.
        """
        memory = _get_memory()
        ctx = RecallContext(
            query=query,
            max_facts=max_facts,
            max_experiences=max_experiences,
            max_procedures=max_procedures,
            min_confidence=min_confidence,
        )
        result = await memory.recall(ctx)

        parts = []
        if result.facts:
            parts.append("## Facts")
            for f in result.facts:
                parts.append(f"- [{f.confidence:.0%}] {f.content}")
        if result.experiences:
            parts.append("## Experiences")
            for e in result.experiences:
                parts.append(f"- {e.content}")
        if result.procedures:
            parts.append("## Procedures")
            for p in result.procedures:
                parts.append(f"- {p.content}")
        if result.working_context:
            parts.append("## Working Context")
            for k, v in result.working_context.items():
                parts.append(f"- {k}: {v}")

        if not parts:
            return f"No memories found for: {query}"

        return f"Found {result.total_items} memories for '{query}':\n\n" + "\n".join(parts)

    # ===== Tool 2: memory_store =====

    @mcp.tool()
    async def memory_store(
        content: str,
        type: str = "auto",
        confidence: float = 0.7,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,
        tags: str = "",
    ) -> str:
        """Store a belief, experience, or procedure.

        Args:
            content: The content to store
            type: "belief", "experience", "procedure", or "auto" (LLM classifies)
            confidence: Confidence level (0.0-1.0)
            subject: For beliefs — triplet subject
            predicate: For beliefs — triplet predicate
            object: For beliefs — triplet object
            tags: Comma-separated tags
        """
        from silicon_memory.llm.provider import classify_memory_type

        memory = _get_memory()
        item_id = uuid4()
        tag_set = set(t.strip() for t in tags.split(",") if t.strip()) if tags else set()
        source = Source(id="mcp", type=SourceType.HUMAN, reliability=0.8)

        # Auto-classify using LLM
        memory_type = type
        if memory_type == "auto":
            if not _scheduler_started:
                await _scheduler.start()
                nonlocal _scheduler_started
                _scheduler_started = True
            memory_type, _ = await classify_memory_type(
                _scheduler, content, priority=Priority.HIGH
            )

        if memory_type == "belief":
            triplet = None
            if subject and predicate and object:
                triplet = Triplet(subject=subject, predicate=predicate, object=object)
            belief = Belief(
                id=item_id, content=content, triplet=triplet,
                confidence=confidence, source=source, tags=tag_set,
                user_id=memory.user_context.user_id,
                tenant_id=memory.user_context.tenant_id,
            )
            await memory.commit_belief(belief)
        elif memory_type == "experience":
            exp = Experience(
                id=item_id, content=content, tags=tag_set,
                user_id=memory.user_context.user_id,
                tenant_id=memory.user_context.tenant_id,
            )
            await memory.record_experience(exp)
        elif memory_type == "procedure":
            proc = Procedure(
                id=item_id, name=content[:50], description=content,
                confidence=confidence, source=source, tags=tag_set,
                user_id=memory.user_context.user_id,
                tenant_id=memory.user_context.tenant_id,
            )
            await memory.commit_procedure(proc)
        else:
            return f"Unknown type: {memory_type}. Use 'belief', 'experience', 'procedure', or 'auto'."

        classified = " (auto-classified)" if type == "auto" else ""
        return f"Stored {memory_type} {item_id}{classified}"

    # ===== Tool 3: memory_get =====

    @mcp.tool()
    async def memory_get(type: str, id: str) -> str:
        """Get a specific memory by type and ID.

        Args:
            type: "belief", "experience", "procedure", or "decision"
            id: The UUID of the memory item
        """
        memory = _get_memory()
        uid = UUID(id)

        if type == "belief":
            item = await memory.get_belief(uid)
            if not item:
                return f"Belief {id} not found"
            text = item.content or (item.triplet.as_text() if item.triplet else "N/A")
            return f"Belief [{item.confidence:.0%}] ({item.status.value}): {text}"

        elif type == "experience":
            item = await memory.get_experience(uid)
            if not item:
                return f"Experience {id} not found"
            return f"Experience: {item.content}" + (f"\nOutcome: {item.outcome}" if item.outcome else "")

        elif type == "procedure":
            item = await memory.get_procedure(uid)
            if not item:
                return f"Procedure {id} not found"
            return f"Procedure '{item.name}': {item.description}"

        elif type == "decision":
            item = await memory.get_decision(uid)
            if not item:
                return f"Decision {id} not found"
            return f"Decision '{item.title}' ({item.status.value}): {item.description}"

        return f"Unknown type: {type}"

    # ===== Tool 4: memory_query =====

    @mcp.tool()
    async def memory_query(
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> str:
        """Search beliefs by semantic similarity.

        Args:
            query: Search query
            limit: Max results
            min_confidence: Minimum confidence threshold
        """
        memory = _get_memory()
        beliefs = await memory.query_beliefs(query, limit=limit, min_confidence=min_confidence)

        if not beliefs:
            return f"No beliefs found for: {query}"

        lines = [f"Found {len(beliefs)} beliefs for '{query}':"]
        for b in beliefs:
            text = b.content or (b.triplet.as_text() if b.triplet else "N/A")
            lines.append(f"- [{b.confidence:.0%}] {text} (id: {b.id})")
        return "\n".join(lines)

    # ===== Tool 5: what_do_you_know =====

    @mcp.tool()
    async def what_do_you_know(query: str, min_confidence: float = 0.3) -> str:
        """Query with proof — 'what do you know about X?'

        Returns beliefs with sources, evidence, and contradictions.

        Args:
            query: The topic to query
            min_confidence: Minimum confidence threshold
        """
        memory = _get_memory()
        proof = await memory.what_do_you_know(query, min_confidence=min_confidence)
        return proof.as_report()

    # ===== Tool 6: working_memory =====

    @mcp.tool()
    async def working_memory(
        action: str = "get_all",
        key: str | None = None,
        value: str | None = None,
        ttl_seconds: int = 300,
    ) -> str:
        """Manage working memory (short-term context).

        Args:
            action: "get_all", "get", "set", or "delete"
            key: Key name (required for get/set/delete)
            value: Value to set (required for set, JSON string for complex values)
            ttl_seconds: Time-to-live in seconds (for set)
        """
        memory = _get_memory()

        if action == "get_all":
            ctx = await memory.get_all_context()
            if not ctx:
                return "Working memory is empty"
            lines = ["Working memory:"]
            for k, v in ctx.items():
                lines.append(f"- {k}: {v}")
            return "\n".join(lines)

        if not key:
            return "key is required for get/set/delete"

        if action == "get":
            val = await memory.get_context(key)
            if val is None:
                return f"Key '{key}' not found in working memory"
            return f"{key}: {val}"

        elif action == "set":
            if value is None:
                return "value is required for set"
            # Try to parse as JSON for complex values
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed = value
            await memory.set_context(key, parsed, ttl_seconds)
            return f"Set {key} = {parsed} (TTL: {ttl_seconds}s)"

        elif action == "delete":
            deleted = await memory.delete_context(key)
            return f"Deleted '{key}'" if deleted else f"Key '{key}' not found"

        return f"Unknown action: {action}. Use get_all, get, set, or delete."

    # ===== Tool 7: decision_store =====

    @mcp.tool()
    async def decision_store(
        title: str,
        description: str,
        assumptions: str = "[]",
        tags: str = "",
    ) -> str:
        """Record a decision with its assumptions.

        Args:
            title: Short title for the decision
            description: Full description of what was decided and why
            assumptions: JSON array of {belief_id, description, confidence_at_decision, is_critical}
            tags: Comma-separated tags
        """
        memory = _get_memory()

        parsed_assumptions = []
        try:
            raw = json.loads(assumptions)
            for a in raw:
                parsed_assumptions.append(Assumption(
                    belief_id=UUID(a["belief_id"]),
                    description=a["description"],
                    confidence_at_decision=a.get("confidence_at_decision", 0.5),
                    is_critical=a.get("is_critical", False),
                ))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        tag_set = set(t.strip() for t in tags.split(",") if t.strip()) if tags else set()

        decision = Decision(
            title=title,
            description=description,
            assumptions=parsed_assumptions,
            tags=tag_set,
            user_id=memory.user_context.user_id,
            tenant_id=memory.user_context.tenant_id,
        )

        await memory.commit_decision(decision)
        return f"Decision stored: {decision.id} — '{title}'"

    # ===== Tool 8: decision_recall =====

    @mcp.tool()
    async def decision_recall(query: str, limit: int = 10) -> str:
        """Search past decisions by semantic similarity.

        Args:
            query: Search query
            limit: Max results
        """
        memory = _get_memory()
        decisions = await memory.recall_decisions(query, k=limit)

        if not decisions:
            return f"No decisions found for: {query}"

        lines = [f"Found {len(decisions)} decisions:"]
        for d in decisions:
            lines.append(f"- [{d.status.value}] {d.title}: {d.description[:100]} (id: {d.id})")
        return "\n".join(lines)

    # ===== Tool 9: context_switch =====

    @mcp.tool()
    async def context_switch(
        action: str = "snapshot",
        task_context: str = "",
    ) -> str:
        """Snapshot current context or resume a previous one.

        Args:
            action: "snapshot" to save, "resume" to restore, "list" to see snapshots
            task_context: Task identifier (e.g. "project-alpha/auth-module")
        """
        memory = _get_memory()

        if action == "snapshot":
            if not task_context:
                return "task_context required for snapshot"
            if not _scheduler_started:
                await _scheduler.start()
                nonlocal _scheduler_started
                _scheduler_started = True
            snapshot = await memory.create_snapshot(task_context, llm_provider=_scheduler)
            return (
                f"Snapshot created: {snapshot.id}\n"
                f"Task: {snapshot.task_context}\n"
                f"Summary: {snapshot.summary}\n"
                f"Working memory keys: {list(snapshot.working_memory.keys())}"
            )

        elif action == "resume":
            if not task_context:
                return "task_context required for resume"
            snapshot = await memory.get_latest_snapshot(task_context)
            if not snapshot:
                return f"No snapshot found for: {task_context}"
            # Restore working memory
            for k, v in snapshot.working_memory.items():
                await memory.set_context(k, v, ttl_seconds=3600)
            return (
                f"Resumed: {snapshot.task_context}\n"
                f"Summary: {snapshot.summary}\n"
                f"Restored {len(snapshot.working_memory)} working memory keys"
            )

        elif action == "list":
            snapshots = await memory.list_snapshots(
                task_context=task_context or None, limit=10
            )
            if not snapshots:
                return "No snapshots found"
            lines = ["Snapshots:"]
            for s in snapshots:
                lines.append(f"- {s.task_context} ({s.created_at.isoformat()}) — {s.summary[:80]}")
            return "\n".join(lines)

        return f"Unknown action: {action}. Use snapshot, resume, or list."

    # ===== Tool 10: forget =====

    @mcp.tool()
    async def forget(
        scope: str,
        entity_id: str | None = None,
        session_id: str | None = None,
        topics: str = "",
        query: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Forget memories (GDPR-compliant deletion).

        Args:
            scope: "entity", "session", "topic", "query", or "all"
            entity_id: Required for entity scope
            session_id: Required for session scope
            topics: Comma-separated topics for topic scope
            query: Search query for query scope
            reason: Optional reason for audit trail
        """
        memory = _get_memory()

        if scope == "entity":
            if not entity_id:
                return "entity_id required"
            result = await memory.forget_entity(entity_id, reason=reason)
        elif scope == "session":
            if not session_id:
                return "session_id required"
            result = await memory.forget_session(session_id, reason=reason)
        elif scope == "topic":
            topic_list = [t.strip() for t in topics.split(",") if t.strip()]
            if not topic_list:
                return "topics required"
            result = await memory.forget_topics(topic_list, reason=reason)
        elif scope == "query":
            if not query:
                return "query required"
            result = await memory.selective_forget(query, reason=reason)
        elif scope == "all":
            result = await memory.forget_all(reason=reason)
        else:
            return f"Unknown scope: {scope}. Use entity, session, topic, query, or all."

        return f"Forgot {result.deleted_count} items (scope: {scope})"

    return mcp
