"""Snapshot service for context switch snapshots."""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

from silicon_memory.snapshot.types import ContextSnapshot, SnapshotConfig

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory
    from silicon_memory.storage.silicondb_backend import SiliconDBBackend
    from silicon_memory.reflection.llm import LLMProvider


class SnapshotService:
    """Service for creating and retrieving context snapshots.

    Gathers current working state and generates a summary (via LLM
    or rule-based fallback) for task-switch persistence.
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        backend: "SiliconDBBackend",
        config: SnapshotConfig | None = None,
        llm_provider: "LLMProvider | None" = None,
    ) -> None:
        self._memory = memory
        self._backend = backend
        self._config = config or SnapshotConfig()
        self._llm_provider = llm_provider

    async def create_snapshot(
        self,
        task_context: str,
        llm_provider: "LLMProvider | None" = None,
    ) -> ContextSnapshot:
        """Create a snapshot of the current working state.

        Args:
            task_context: Identifier for the task being snapshotted
            llm_provider: Optional LLM provider override

        Returns:
            The created ContextSnapshot
        """
        provider = llm_provider or self._llm_provider

        # Gather working memory
        working_memory = await self._memory.get_all_context()

        # Gather recent experiences
        recent_exps = await self._memory.get_recent_experiences(
            hours=self._config.recent_hours,
            limit=self._config.max_recent_experiences,
        )
        recent_ids = [exp.id for exp in recent_exps]

        # Generate summary
        if provider:
            summary, next_steps, open_questions = await self._generate_llm_summary(
                task_context, working_memory, recent_exps, provider
            )
        else:
            summary, next_steps, open_questions = self._generate_rule_based_summary(
                task_context, working_memory, recent_exps
            )

        user_ctx = self._memory.user_context

        snapshot = ContextSnapshot(
            task_context=task_context,
            summary=summary,
            working_memory=working_memory,
            recent_experiences=recent_ids,
            next_steps=next_steps,
            open_questions=open_questions,
            session_id=getattr(user_ctx, "session_id", None),
            user_id=user_ctx.user_id,
            tenant_id=user_ctx.tenant_id,
        )

        # Store in backend
        await self._backend.store_snapshot(snapshot)

        return snapshot

    async def on_session_end(
        self,
        task_context: str,
        llm_provider: "LLMProvider | None" = None,
    ) -> ContextSnapshot:
        """Auto-create a snapshot when a session ends.

        This hook should be called when a conversation session ends
        to capture the final working state for later resumption.

        Args:
            task_context: Identifier for the task/session
            llm_provider: Optional LLM provider override

        Returns:
            The created ContextSnapshot
        """
        return await self.create_snapshot(task_context, llm_provider)

    async def get_latest_snapshot(
        self,
        task_context: str,
    ) -> ContextSnapshot | None:
        """Get the most recent snapshot for a task context.

        Args:
            task_context: The task context identifier

        Returns:
            The latest ContextSnapshot or None
        """
        snapshots = await self._backend.query_snapshots_by_context(
            task_context=task_context,
            limit=1,
        )
        return snapshots[0] if snapshots else None

    async def list_snapshots(
        self,
        task_context: str | None = None,
        limit: int = 10,
    ) -> list[ContextSnapshot]:
        """List snapshots, optionally filtered by task context.

        Args:
            task_context: Optional filter
            limit: Maximum results

        Returns:
            List of ContextSnapshot objects sorted by created_at desc
        """
        return await self._backend.query_snapshots_by_context(
            task_context=task_context,
            limit=limit,
        )

    async def _generate_llm_summary(
        self,
        task_context: str,
        working_memory: dict[str, Any],
        recent_experiences: list,
        provider: "LLMProvider",
    ) -> tuple[list[str], list[str], list[str]]:
        """Generate a summary using an LLM.

        Returns:
            Tuple of (summary, next_steps, open_questions)
        """
        # Build prompt
        parts = [
            f"Task context: {task_context}\n",
            "\n## Working Memory\n",
        ]
        for key, value in working_memory.items():
            parts.append(f"- {key}: {value}\n")

        parts.append("\n## Recent Experiences\n")
        for exp in recent_experiences[:10]:
            parts.append(f"- {exp.content}")
            if exp.outcome:
                parts.append(f" → {exp.outcome}")
            parts.append("\n")

        parts.append(
            "\n## Task\n"
            "Summarize the current working state for task resumption. "
            "Return JSON with keys: summary (string), next_steps (list of strings), "
            "open_questions (list of strings).\n"
            "Only return the JSON object, no other text."
        )

        prompt = "".join(parts)

        try:
            response = await provider.complete(
                prompt=prompt,
                system="You are a context summarization agent. Generate concise task resumption summaries.",
                temperature=self._config.llm_temperature,
                max_tokens=1000,
            )

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return (
                    data.get("summary", ""),
                    data.get("next_steps", []),
                    data.get("open_questions", []),
                )
        except Exception:
            pass

        # Fall back to rule-based
        return self._generate_rule_based_summary(
            task_context, working_memory, recent_experiences
        )

    def _generate_rule_based_summary(
        self,
        task_context: str,
        working_memory: dict[str, Any],
        recent_experiences: list,
    ) -> tuple[str, list[str], list[str]]:
        """Generate a summary using heuristic rules.

        Returns:
            Tuple of (summary, next_steps, open_questions)
        """
        max_chars = self._config.fallback_summary_max_chars

        # Build summary from working memory keys and recent experience content
        summary_parts = [f"Context: {task_context}."]

        if working_memory:
            keys_str = ", ".join(list(working_memory.keys())[:10])
            summary_parts.append(f"Working on: {keys_str}.")

        if recent_experiences:
            latest = recent_experiences[0] if recent_experiences else None
            if latest:
                content = latest.content[:200]
                summary_parts.append(f"Latest: {content}")

        summary = " ".join(summary_parts)[:max_chars]

        # Extract open questions: sentences ending with "?"
        open_questions: list[str] = []
        for exp in recent_experiences:
            sentences = re.split(r'[.!?]+', exp.content)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and "?" in exp.content[exp.content.find(sentence):exp.content.find(sentence) + len(sentence) + 1:]:
                    open_questions.append(sentence.strip() + "?")

        # Extract next steps: keyword matching for todo/next/should
        next_steps: list[str] = []
        todo_pattern = re.compile(
            r'(?:TODO|NEXT|FIXME|should|need to|will)\s*:?\s*(.+)',
            re.IGNORECASE,
        )
        for exp in recent_experiences:
            for line in exp.content.split("\n"):
                match = todo_pattern.search(line)
                if match:
                    next_steps.append(match.group(1).strip()[:200])

        return summary, next_steps[:5], open_questions[:5]
