"""Tests for document ingestion adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from silicon_memory.ingestion.types import (
    IngestionAdapter,
    IngestionConfig,
    IngestionResult,
)
from silicon_memory.ingestion.document import (
    DocumentAdapter,
    DocumentConfig,
    DocumentSection,
)
from silicon_memory.core.types import Experience


# ============================================================================
# Unit tests: DocumentSection dataclass
# ============================================================================


class TestDocumentSection:
    """Unit tests for DocumentSection."""

    def test_defaults(self):
        sec = DocumentSection()
        assert sec.title == ""
        assert sec.content == ""
        assert sec.level == 0
        assert sec.section_index == 0

    def test_construction(self):
        sec = DocumentSection(
            title="Introduction",
            content="This is the intro.",
            level=1,
            section_index=0,
        )
        assert sec.title == "Introduction"
        assert sec.content == "This is the intro."
        assert sec.level == 1


# ============================================================================
# Unit tests: DocumentConfig
# ============================================================================


class TestDocumentConfig:
    """Unit tests for DocumentConfig."""

    def test_defaults(self):
        config = DocumentConfig()
        assert config.segment_by_headings is True
        assert config.extract_action_items is True
        assert config.resolve_entities is True
        assert config.min_heading_level == 1
        assert config.max_heading_level == 4
        # Inherited from IngestionConfig
        assert config.max_segment_length == 2000
        assert config.min_segment_length == 50

    def test_inherits_from_ingestion_config(self):
        assert issubclass(DocumentConfig, IngestionConfig)


# ============================================================================
# Unit tests: Protocol satisfaction
# ============================================================================


class TestDocumentProtocol:
    """Test that DocumentAdapter satisfies IngestionAdapter protocol."""

    def test_protocol_satisfaction(self):
        adapter = DocumentAdapter()
        assert isinstance(adapter, IngestionAdapter)

    def test_source_type(self):
        adapter = DocumentAdapter()
        assert adapter.source_type == "document"


# ============================================================================
# Unit tests: Format detection
# ============================================================================


class TestFormatDetection:
    """Test markdown vs plaintext detection."""

    def _adapter(self):
        return DocumentAdapter()

    def test_detect_markdown(self):
        content = (
            "# Introduction\n"
            "\n"
            "This is a **markdown** document.\n"
            "\n"
            "- Item one\n"
            "- Item two\n"
        )
        assert self._adapter()._detect_format(content) == "markdown"

    def test_detect_plaintext(self):
        content = (
            "This is a plain text document.\n"
            "It has no special formatting.\n"
            "Just regular sentences.\n"
        )
        assert self._adapter()._detect_format(content) == "plaintext"

    def test_detect_markdown_with_links(self):
        content = (
            "# Title\n"
            "\n"
            "See [this link](https://example.com) for details.\n"
        )
        assert self._adapter()._detect_format(content) == "markdown"

    def test_detect_markdown_with_code_blocks(self):
        content = (
            "# Code Example\n"
            "\n"
            "```python\n"
            "print('hello')\n"
            "```\n"
        )
        assert self._adapter()._detect_format(content) == "markdown"


# ============================================================================
# Unit tests: Markdown segmentation
# ============================================================================


class TestMarkdownSegmentation:
    """Test segmentation of markdown documents."""

    def _adapter(self, config=None):
        return DocumentAdapter(config=config)

    def test_split_on_headings(self):
        """Test splitting on # headings."""
        content = (
            "# Introduction\n"
            "\n"
            "This is the intro.\n"
            "\n"
            "# Background\n"
            "\n"
            "Some background info.\n"
            "\n"
            "# Conclusion\n"
            "\n"
            "Final thoughts.\n"
        )
        sections = self._adapter()._segment_markdown(content)
        assert len(sections) == 3
        assert sections[0].title == "Introduction"
        assert sections[1].title == "Background"
        assert sections[2].title == "Conclusion"

    def test_heading_levels(self):
        """Test that heading level is captured."""
        content = (
            "# H1 Title\n"
            "\n"
            "Content under H1.\n"
            "\n"
            "## H2 Subtitle\n"
            "\n"
            "Content under H2.\n"
            "\n"
            "### H3 Section\n"
            "\n"
            "Content under H3.\n"
        )
        sections = self._adapter()._segment_markdown(content)
        assert len(sections) == 3
        assert sections[0].level == 1
        assert sections[1].level == 2
        assert sections[2].level == 3

    def test_content_extraction(self):
        """Test that content is correctly associated with sections."""
        content = (
            "# Setup\n"
            "\n"
            "Install the dependencies.\n"
            "Run the setup script.\n"
            "\n"
            "# Usage\n"
            "\n"
            "Import the module.\n"
        )
        sections = self._adapter()._segment_markdown(content)
        assert "Install the dependencies" in sections[0].content
        assert "Import the module" in sections[1].content

    def test_content_before_first_heading(self):
        """Test handling of content before the first heading."""
        content = (
            "Some introductory text.\n"
            "\n"
            "# First Section\n"
            "\n"
            "Section content.\n"
        )
        sections = self._adapter()._segment_markdown(content)
        assert len(sections) == 2
        assert sections[0].title == ""
        assert "introductory text" in sections[0].content

    def test_heading_level_filtering(self):
        """Test that headings outside min/max range are treated as content."""
        content = (
            "# Title\n"
            "\n"
            "##### Deep heading\n"
            "\n"
            "Content here.\n"
        )
        adapter = self._adapter(DocumentConfig(max_heading_level=3))
        sections = adapter._segment_markdown(content)
        # The ##### heading should be treated as content, not a section boundary
        assert len(sections) == 1
        assert "Deep heading" in sections[0].content


# ============================================================================
# Unit tests: Plaintext segmentation
# ============================================================================


class TestPlaintextSegmentation:
    """Test segmentation of plain text documents."""

    def _adapter(self):
        return DocumentAdapter()

    def test_paragraph_based_segmentation(self):
        """Test splitting on blank-line-separated paragraphs."""
        content = (
            "First paragraph here.\n"
            "Still first paragraph.\n"
            "\n"
            "Second paragraph starts here.\n"
            "More of second paragraph.\n"
        )
        sections = self._adapter()._segment_plaintext(content)
        assert len(sections) >= 1
        # Content should be captured
        full_content = " ".join(s.content for s in sections)
        assert "First paragraph" in full_content
        assert "Second paragraph" in full_content

    def test_allcaps_heading_detection(self):
        """Test ALL-CAPS lines detected as section headings."""
        content = (
            "INTRODUCTION\n"
            "This is the intro text.\n"
            "More intro.\n"
            "\n"
            "BACKGROUND\n"
            "Some background information.\n"
        )
        sections = self._adapter()._segment_plaintext(content)
        assert len(sections) == 2
        assert sections[0].title == "INTRODUCTION"
        assert sections[1].title == "BACKGROUND"

    def test_underlined_heading_detection(self):
        """Test underlined headings (=== or ---) as boundaries."""
        content = (
            "Introduction\n"
            "============\n"
            "This is the intro.\n"
            "\n"
            "Background\n"
            "----------\n"
            "Some background.\n"
        )
        sections = self._adapter()._segment_plaintext(content)
        assert len(sections) == 2
        assert sections[0].title == "Introduction"
        assert sections[1].title == "Background"


# ============================================================================
# Unit tests: Action item extraction
# ============================================================================


class TestDocumentActionItemExtraction:
    """Test heuristic action item extraction from documents."""

    def _adapter(self):
        return DocumentAdapter()

    def test_action_keyword_extraction(self):
        """Test extraction of ACTION: and TODO: items."""
        sections = [
            DocumentSection(
                content="We should proceed. ACTION: Deploy by Friday.\nTODO: Update docs.",
                section_index=0,
            ),
        ]
        items = self._adapter()._extract_action_items_heuristic(sections)
        assert len(items) == 2
        assert any("Deploy" in item["action"] for item in items)

    def test_name_will_pattern(self):
        """Test '[Name] will' extraction."""
        sections = [
            DocumentSection(
                content="Alice will prepare the presentation.",
                section_index=0,
            ),
        ]
        items = self._adapter()._extract_action_items_heuristic(sections)
        assert len(items) >= 1
        assert any(item.get("owner") == "Alice" for item in items)

    def test_no_action_items(self):
        """Test no extraction when no patterns match."""
        sections = [
            DocumentSection(content="Just a normal paragraph.", section_index=0),
        ]
        items = self._adapter()._extract_action_items_heuristic(sections)
        assert items == []


# ============================================================================
# Integration tests: Full pipeline
# ============================================================================


class TestDocumentIngestionPipeline:
    """Integration tests for the full document ingestion pipeline."""

    @pytest.fixture
    def mock_memory(self):
        memory = AsyncMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme", session_id="s1"
        )
        memory.record_experience = AsyncMock()
        return memory

    async def test_markdown_pipeline(self, mock_memory):
        """Test full ingestion of a markdown document."""
        doc = (
            "# Project Overview\n"
            "\n"
            "This project aims to build a new API.\n"
            "\n"
            "## Requirements\n"
            "\n"
            "- Fast response times\n"
            "- **Secure** authentication\n"
            "\n"
            "## Action Items\n"
            "\n"
            "TODO: Set up CI/CD pipeline.\n"
            "Alice will create the database schema.\n"
        )

        adapter = DocumentAdapter(
            config=DocumentConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=doc,
            metadata={"document_id": "doc-1", "title": "Project Plan"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 2
        assert result.source_type == "document"
        assert result.action_items_detected >= 1

    async def test_plaintext_pipeline(self, mock_memory):
        """Test full ingestion of a plain text document."""
        doc = (
            "INTRODUCTION\n"
            "This document describes the system architecture.\n"
            "\n"
            "COMPONENTS\n"
            "The system has three main components.\n"
            "Each component handles a specific domain.\n"
        )

        adapter = DocumentAdapter(
            config=DocumentConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=doc,
            metadata={"document_id": "doc-2", "title": "Architecture"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1
        assert result.source_type == "document"

    async def test_experience_context_metadata(self, mock_memory):
        """Test that experiences include correct document context metadata."""
        doc = (
            "# Setup Guide\n"
            "\n"
            "Follow these **steps** to set up.\n"
        )

        adapter = DocumentAdapter(
            config=DocumentConfig(resolve_entities=False)
        )
        await adapter.ingest(
            content=doc,
            metadata={"document_id": "doc-3", "title": "Setup Guide"},
            memory=mock_memory,
        )

        call_args = mock_memory.record_experience.call_args_list
        assert len(call_args) >= 1
        exp = call_args[0][0][0]
        assert exp.context["source_type"] == "document"
        assert exp.context["document_id"] == "doc-3"
        assert exp.context["title"] == "Setup Guide"
        assert exp.context["format"] == "markdown"
        assert exp.session_id == "doc-3"
        assert exp.user_id == "user-1"
        assert exp.tenant_id == "acme"

    async def test_llm_segmentation(self, mock_memory):
        """Test LLM-based segmentation."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(
            return_value='[{"title": "Overview", "content": "Project overview.", "level": 1}]'
        )

        doc = "A document that needs LLM segmentation."

        adapter = DocumentAdapter(
            config=DocumentConfig(
                resolve_entities=False,
                segment_by_headings=False,  # Force LLM path
            )
        )
        result = await adapter.ingest(
            content=doc,
            metadata={"document_id": "doc-4"},
            memory=mock_memory,
            llm_provider=mock_llm,
        )

        assert result.experiences_created >= 1
        mock_llm.complete.assert_awaited()

    async def test_empty_content(self, mock_memory):
        """Test graceful handling of empty input."""
        adapter = DocumentAdapter(
            config=DocumentConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content="",
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created == 0
        assert result.has_errors

    async def test_bytes_input(self, mock_memory):
        """Test that bytes input is decoded."""
        adapter = DocumentAdapter(
            config=DocumentConfig(resolve_entities=False)
        )
        result = await adapter.ingest(
            content=b"# Hello\n\nWorld.\n",
            metadata={"document_id": "doc-5"},
            memory=mock_memory,
        )

        assert result.experiences_created >= 1

    async def test_entity_resolution(self, mock_memory):
        """Test entity resolution on document sections."""
        mock_resolver = AsyncMock()
        mock_resolver.resolve = AsyncMock(
            return_value=MagicMock(resolved=[MagicMock()])
        )

        doc = (
            "# Team\n"
            "\n"
            "Alice is the lead engineer.\n"
        )

        adapter = DocumentAdapter(
            config=DocumentConfig(resolve_entities=True),
            entity_resolver=mock_resolver,
        )
        result = await adapter.ingest(
            content=doc,
            metadata={"document_id": "doc-6"},
            memory=mock_memory,
        )

        assert result.entities_resolved >= 1
        assert mock_resolver.resolve.await_count >= 1

    async def test_malformed_no_sections(self, mock_memory):
        """Test handling when content produces no meaningful sections."""
        adapter = DocumentAdapter(
            config=DocumentConfig(resolve_entities=False)
        )
        # Single whitespace-only content is stripped to empty
        result = await adapter.ingest(
            content="   \n\n  ",
            metadata={},
            memory=mock_memory,
        )

        assert result.experiences_created == 0
        assert result.has_errors
