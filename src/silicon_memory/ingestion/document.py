"""Document ingestion adapter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.types import Experience
from silicon_memory.ingestion.types import IngestionConfig, IngestionResult
from silicon_memory.ingestion._helpers import (
    extract_action_items_from_text,
    parse_llm_json_array,
)

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory
    from silicon_memory.reflection.llm import LLMProvider
    from silicon_memory.entities.resolver import EntityResolver


@dataclass
class DocumentSection:
    """A parsed section of a document."""

    title: str = ""
    content: str = ""
    level: int = 0
    section_index: int = 0


@dataclass
class DocumentConfig(IngestionConfig):
    """Configuration for document ingestion."""

    segment_by_headings: bool = True
    extract_action_items: bool = True
    resolve_entities: bool = True
    min_heading_level: int = 1
    max_heading_level: int = 4


class DocumentAdapter:
    """Ingests text and Markdown documents as experiences.

    Supports two document formats:
    - Markdown (detected by ``#`` headings, ``**bold**``, ``- lists``)
    - Plain text (paragraph-based segmentation, ALL-CAPS heading detection)

    Processing pipeline:
    1. Detect format (markdown vs plain text)
    2. Segment by headings or paragraphs
    3. Create one Experience per section
    4. Extract action items (LLM or heuristic)
    5. Optionally resolve entities
    """

    def __init__(
        self,
        config: DocumentConfig | None = None,
        entity_resolver: "EntityResolver | None" = None,
    ) -> None:
        self._config = config or DocumentConfig()
        self._entity_resolver = entity_resolver

    @property
    def source_type(self) -> str:
        return "document"

    async def ingest(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        memory: "SiliconMemory",
        llm_provider: "LLMProvider | None" = None,
    ) -> IngestionResult:
        """Ingest a document into memory.

        Args:
            content: The document text (markdown or plain text)
            metadata: Should include document_id, title
            memory: SiliconMemory instance
            llm_provider: Optional LLM for enhanced segmentation

        Returns:
            IngestionResult with statistics
        """
        result = IngestionResult(source_type=self.source_type)

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        content = content.strip()
        if not content:
            result.errors.append("Empty document content")
            return result

        document_id = metadata.get("document_id", str(uuid4()))
        doc_title = metadata.get("title", "")

        # Step 1: Detect format
        fmt = self._detect_format(content)

        # Step 2: Segment
        if llm_provider and not self._config.segment_by_headings:
            try:
                sections = await self._segment_with_llm(content, llm_provider)
            except Exception as e:
                result.errors.append(
                    f"LLM segmentation failed, falling back to heuristic: {e}"
                )
                sections = self._segment_heuristic(content, fmt)
        else:
            sections = self._segment_heuristic(content, fmt)

        if not sections:
            result.errors.append("No sections produced from document")
            return result

        # Step 3: Create experiences for each section
        user_ctx = memory.user_context

        for i, section in enumerate(sections):
            section.section_index = i
            try:
                exp = Experience(
                    id=uuid4(),
                    content=section.content,
                    context={
                        "document_id": document_id,
                        "title": doc_title,
                        "section_title": section.title,
                        "section_index": section.section_index,
                        "format": fmt,
                        "source_type": "document",
                        **{k: v for k, v in metadata.items()
                           if k not in ("document_id", "title")},
                    },
                    session_id=document_id,
                    user_id=user_ctx.user_id,
                    tenant_id=user_ctx.tenant_id,
                )
                await memory.record_experience(exp)
                result.experiences_created += 1
            except Exception as e:
                result.errors.append(f"Failed to store section {i}: {e}")

        # Step 4: Extract action items (optional)
        if self._config.extract_action_items:
            try:
                if llm_provider:
                    action_items = await self._extract_action_items_llm(
                        sections, llm_provider
                    )
                else:
                    action_items = self._extract_action_items_heuristic(sections)
                result.action_items_detected = len(action_items)
                result.details["action_items"] = action_items
            except Exception as e:
                result.errors.append(f"Action item extraction error: {e}")

        # Step 5: Resolve entities (optional)
        if self._config.resolve_entities and self._entity_resolver:
            try:
                resolved_count = await self._resolve_entities(sections, memory)
                result.entities_resolved = resolved_count
            except Exception as e:
                result.errors.append(f"Entity resolution error: {e}")

        return result

    def _detect_format(self, content: str) -> str:
        """Detect whether content is markdown or plaintext.

        Checks for markdown indicators: ``#`` headings, ``**bold**``,
        ``- list`` markers, ``[links](url)``, code blocks.
        """
        markdown_indicators = [
            re.compile(r"^#{1,6}\s+", re.MULTILINE),  # # headings
            re.compile(r"\*\*.+?\*\*"),  # **bold**
            re.compile(r"^\s*[-*+]\s+", re.MULTILINE),  # - list items
            re.compile(r"\[.+?\]\(.+?\)"),  # [links](url)
            re.compile(r"^```", re.MULTILINE),  # code blocks
        ]

        matches = sum(1 for p in markdown_indicators if p.search(content))
        return "markdown" if matches >= 2 else "plaintext"

    def _segment_heuristic(
        self,
        content: str,
        fmt: str,
    ) -> list[DocumentSection]:
        """Segment document using format-appropriate heuristic."""
        if fmt == "markdown":
            return self._segment_markdown(content)
        return self._segment_plaintext(content)

    def _segment_markdown(self, content: str) -> list[DocumentSection]:
        """Split markdown content on heading boundaries.

        Each section gets the heading as title, body as content,
        and ``#`` count as level.
        """
        heading_pattern = re.compile(
            r"^(#{1,6})\s+(.+)$", re.MULTILINE
        )

        sections: list[DocumentSection] = []
        lines = content.split("\n")
        current_title = ""
        current_level = 0
        current_lines: list[str] = []

        for line in lines:
            match = heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                if level < self._config.min_heading_level or level > self._config.max_heading_level:
                    current_lines.append(line)
                    continue

                # Flush previous section
                if current_lines or current_title:
                    body = "\n".join(current_lines).strip()
                    if body or current_title:
                        sections.append(DocumentSection(
                            title=current_title,
                            content=body,
                            level=current_level,
                        ))

                current_title = match.group(2).strip()
                current_level = level
                current_lines = []
            else:
                current_lines.append(line)

        # Flush last section
        body = "\n".join(current_lines).strip()
        if body or current_title:
            sections.append(DocumentSection(
                title=current_title,
                content=body,
                level=current_level,
            ))

        return sections

    def _segment_plaintext(self, content: str) -> list[DocumentSection]:
        """Split plain text on paragraphs with heading detection.

        Detects ALL-CAPS lines or underlined headings (``===``/``---``)
        as section boundaries. Otherwise splits on blank-line-separated
        paragraphs.
        """
        lines = content.split("\n")
        sections: list[DocumentSection] = []
        current_title = ""
        current_lines: list[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check for ALL-CAPS heading (at least 3 chars, all uppercase letters/spaces)
            is_allcaps_heading = (
                len(stripped) >= 3
                and stripped == stripped.upper()
                and re.match(r"^[A-Z][A-Z\s]+$", stripped)
            )

            # Check for underlined heading (next line is === or ---)
            is_underlined_heading = False
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if stripped and (
                    re.match(r"^={3,}$", next_line) or re.match(r"^-{3,}$", next_line)
                ):
                    is_underlined_heading = True

            if is_allcaps_heading or is_underlined_heading:
                # Flush previous section
                if current_lines or current_title:
                    body = "\n".join(current_lines).strip()
                    if body or current_title:
                        sections.append(DocumentSection(
                            title=current_title,
                            content=body,
                            level=1,
                        ))
                current_title = stripped
                current_lines = []
                if is_underlined_heading:
                    i += 2  # Skip underline
                else:
                    i += 1
                continue

            # Blank line boundary for paragraph grouping
            if not stripped:
                i += 1
                continue

            current_lines.append(stripped)
            i += 1

        # Flush last section
        body = "\n".join(current_lines).strip()
        if body or current_title:
            sections.append(DocumentSection(
                title=current_title,
                content=body,
                level=0 if not current_title else 1,
            ))

        return sections

    async def _segment_with_llm(
        self,
        content: str,
        provider: "LLMProvider",
    ) -> list[DocumentSection]:
        """Use LLM to identify logical sections in document."""
        prompt = (
            "Segment the following document into logical sections. "
            "Return a JSON array where each element has:\n"
            '- "title": section title\n'
            '- "content": section content\n'
            '- "level": heading depth (1=top level)\n'
            "\nOnly return the JSON array, no other text.\n\n"
            f"Document:\n{content[:3000]}"
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are a document structure analyzer.",
            temperature=self._config.llm_temperature,
            max_tokens=2000,
        )

        section_defs = parse_llm_json_array(response)
        sections: list[DocumentSection] = []

        for sec_def in section_defs:
            sections.append(DocumentSection(
                title=sec_def.get("title", ""),
                content=sec_def.get("content", ""),
                level=sec_def.get("level", 1),
            ))

        return sections

    def _extract_action_items_heuristic(
        self,
        sections: list[DocumentSection],
    ) -> list[dict[str, Any]]:
        """Extract action items using keyword patterns."""
        action_items: list[dict[str, Any]] = []
        for section in sections:
            action_items.extend(
                extract_action_items_from_text(
                    section.content, "section_index", section.section_index
                )
            )
        return action_items

    async def _extract_action_items_llm(
        self,
        sections: list[DocumentSection],
        provider: "LLMProvider",
    ) -> list[dict[str, Any]]:
        """Use LLM to extract action items from document sections."""
        text = "\n\n".join(
            f"[Section {s.section_index}: {s.title}]\n{s.content}"
            for s in sections
        )

        prompt = (
            "Extract action items from this document. "
            "Return a JSON array where each element has:\n"
            '- "action": what needs to be done\n'
            '- "owner": who is responsible (or null)\n'
            '- "section_index": which section it came from\n'
            "\nOnly return the JSON array, no other text.\n\n"
            f"{text}"
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are a document action item extractor.",
            temperature=self._config.llm_temperature,
            max_tokens=1500,
        )

        try:
            return parse_llm_json_array(response)
        except (ValueError, json.JSONDecodeError):
            return []

    async def _resolve_entities(
        self,
        sections: list[DocumentSection],
        memory: "SiliconMemory",
    ) -> int:
        """Resolve entity names mentioned in document sections."""
        if not self._entity_resolver:
            return 0

        resolved_count = 0

        for section in sections:
            try:
                result = await self._entity_resolver.resolve(section.content)
                if result.resolved:
                    resolved_count += len(result.resolved)
            except Exception:
                pass

        return resolved_count
