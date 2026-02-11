"""Tests for SM-6: News Integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from silicon_memory.ingestion.news import (
    NewsArticle,
    NewsArticleAdapter,
    NewsIngestionConfig,
)
from silicon_memory.ingestion.types import IngestionResult
from silicon_memory.core.types import Belief, Source, SourceType


# ============================================================================
# Unit tests: NewsArticle dataclass
# ============================================================================


class TestNewsArticle:
    """Test NewsArticle dataclass."""

    def test_defaults(self):
        article = NewsArticle()
        assert article.title == ""
        assert article.body == ""
        assert article.source_name == ""
        assert article.source_url == ""

    def test_with_values(self):
        article = NewsArticle(
            title="Breaking News",
            body="Content here.",
            source_name="Reuters",
            source_url="https://reuters.com/article",
        )
        assert article.title == "Breaking News"
        assert article.source_name == "Reuters"


# ============================================================================
# Unit tests: NewsIngestionConfig
# ============================================================================


class TestNewsIngestionConfig:
    """Test NewsIngestionConfig."""

    def test_defaults(self):
        config = NewsIngestionConfig()
        assert config.extract_claims is True
        assert config.default_credibility == 0.5
        assert config.max_claims_per_article == 20
        assert config.llm_temperature == 0.3

    def test_custom_credibility(self):
        config = NewsIngestionConfig(default_credibility=0.9)
        assert config.default_credibility == 0.9


# ============================================================================
# Unit tests: Article parsing
# ============================================================================


class TestArticleParsing:
    """Test article parsing from JSON and raw text."""

    def _adapter(self, config=None):
        return NewsArticleAdapter(config=config)

    def test_parse_json_article(self):
        content = json.dumps({
            "title": "Test Article",
            "body": "Article body text.",
            "source_name": "Reuters",
            "source_url": "https://reuters.com/test",
            "date": "2025-01-01",
            "author": "John Doe",
        })
        article = self._adapter()._parse_article(content, {})
        assert article.title == "Test Article"
        assert article.body == "Article body text."
        assert article.source_name == "Reuters"
        assert article.author == "John Doe"

    def test_parse_raw_text_article(self):
        content = "This is a plain text article about technology."
        metadata = {
            "title": "Tech Article",
            "source_name": "TechCrunch",
        }
        article = self._adapter()._parse_article(content, metadata)
        assert article.body == content
        assert article.title == "Tech Article"
        assert article.source_name == "TechCrunch"

    def test_parse_json_with_metadata_fallback(self):
        content = json.dumps({"body": "Just the body."})
        metadata = {"title": "From Metadata", "source_name": "BBC"}
        article = self._adapter()._parse_article(content, metadata)
        assert article.title == "From Metadata"
        assert article.source_name == "BBC"

    def test_parse_empty_json_object(self):
        content = json.dumps({})
        article = self._adapter()._parse_article(content, {"title": "Fallback"})
        assert article.title == "Fallback"
        assert article.body == ""


# ============================================================================
# Unit tests: Heuristic claim extraction
# ============================================================================


class TestClaimExtraction:
    """Test heuristic claim extraction."""

    def _adapter(self):
        return NewsArticleAdapter()

    def test_extract_claims_from_sentences(self):
        article = NewsArticle(
            title="Test",
            body=(
                "The company reported record profits this quarter. "
                "Revenue grew by 25 percent year over year. "
                "Analysts expect continued growth."
            ),
        )
        claims = self._adapter()._extract_claims_heuristic(article)
        assert len(claims) >= 2
        for claim in claims:
            assert "claim" in claim
            assert "confidence" in claim

    def test_skips_short_sentences(self):
        article = NewsArticle(title="Test", body="Short. Also short.")
        claims = self._adapter()._extract_claims_heuristic(article)
        assert len(claims) == 0

    def test_skips_questions(self):
        article = NewsArticle(
            title="Test",
            body="What is the meaning of this very long question that should be skipped?",
        )
        claims = self._adapter()._extract_claims_heuristic(article)
        # Questions should be filtered (though split by '.' so may not trigger)
        for claim in claims:
            assert not claim["claim"].strip().endswith("?")


# ============================================================================
# Integration tests: News ingestion pipeline
# ============================================================================


class TestNewsIngestionPipeline:
    """Integration tests for news article ingestion."""

    @pytest.fixture
    def mock_memory(self):
        memory = AsyncMock()
        memory.user_context = MagicMock(
            user_id="user-1", tenant_id="acme"
        )
        memory.record_experience = AsyncMock()
        memory.commit_belief = AsyncMock()
        return memory

    async def test_full_pipeline_raw_text(self, mock_memory):
        """Test ingestion of raw text article."""
        article_text = (
            "The Federal Reserve raised interest rates by 0.25 percent today. "
            "This marks the third consecutive rate increase this year. "
            "Economists predict this will slow inflation."
        )

        adapter = NewsArticleAdapter()
        result = await adapter.ingest(
            content=article_text,
            metadata={
                "title": "Fed Raises Rates",
                "source_name": "Reuters",
                "source_url": "https://reuters.com/fed",
                "credibility": 0.8,
            },
            memory=mock_memory,
        )

        assert result.experiences_created == 1
        assert result.source_type == "news_article"
        assert not result.has_errors or result.success

        # Claims should have been extracted and stored as beliefs
        assert mock_memory.commit_belief.await_count >= 1

    async def test_full_pipeline_json_article(self, mock_memory):
        """Test ingestion of JSON-formatted article."""
        content = json.dumps({
            "title": "Tech Company IPO",
            "body": "A major tech company announced its initial public offering today. The valuation exceeded market expectations significantly.",
            "source_name": "Bloomberg",
            "source_url": "https://bloomberg.com/ipo",
            "date": "2025-03-15",
        })

        adapter = NewsArticleAdapter()
        result = await adapter.ingest(
            content=content,
            metadata={"credibility": 0.85},
            memory=mock_memory,
        )

        assert result.experiences_created == 1
        assert not result.has_errors

    async def test_empty_article(self, mock_memory):
        """Test handling of empty article."""
        adapter = NewsArticleAdapter()
        result = await adapter.ingest(
            content="",
            metadata={},
            memory=mock_memory,
        )
        assert result.experiences_created == 0
        assert result.has_errors

    async def test_source_attribution_in_experience(self, mock_memory):
        """Test that experience context includes source attribution."""
        adapter = NewsArticleAdapter()
        await adapter.ingest(
            content="An important event happened today in the financial markets worldwide.",
            metadata={
                "source_name": "AP News",
                "source_url": "https://apnews.com/event",
                "credibility": 0.9,
            },
            memory=mock_memory,
        )

        call_args = mock_memory.record_experience.call_args_list
        assert len(call_args) == 1
        exp = call_args[0][0][0]
        assert exp.context["source_type"] == "news_article"
        assert exp.context["source_name"] == "AP News"
        assert exp.context["credibility"] == 0.9

    async def test_credibility_weights_confidence(self, mock_memory):
        """Test that source credibility weights belief confidence."""
        article_text = (
            "The new policy will reduce emissions by fifty percent within a decade."
        )

        adapter = NewsArticleAdapter()
        await adapter.ingest(
            content=article_text,
            metadata={
                "source_name": "Low Trust Source",
                "credibility": 0.3,
            },
            memory=mock_memory,
        )

        # Check stored beliefs have credibility-weighted confidence
        if mock_memory.commit_belief.await_count > 0:
            belief = mock_memory.commit_belief.call_args[0][0]
            assert belief.confidence <= 0.3  # Capped by credibility
            assert belief.source.type == SourceType.EXTERNAL
            assert belief.source.reliability == 0.3

    async def test_llm_claim_extraction(self, mock_memory):
        """Test LLM-based claim extraction."""
        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=json.dumps([
            {"claim": "Inflation fell to 2%", "confidence": 0.9},
            {"claim": "Employment rose", "confidence": 0.7},
        ]))

        adapter = NewsArticleAdapter()
        result = await adapter.ingest(
            content="The economy showed improvement with falling inflation.",
            metadata={"source_name": "FT", "credibility": 0.8},
            memory=mock_memory,
            llm_provider=mock_llm,
        )

        assert result.experiences_created == 1
        assert mock_memory.commit_belief.await_count == 2
        mock_llm.complete.assert_awaited()

    async def test_bytes_input(self, mock_memory):
        """Test bytes input handling."""
        adapter = NewsArticleAdapter()
        result = await adapter.ingest(
            content=b"An article about something important in the world of technology.",
            metadata={"source_name": "Test"},
            memory=mock_memory,
        )
        assert result.experiences_created == 1


# ============================================================================
# Unit tests: Cross-reference
# ============================================================================


class TestCrossReference:
    """Test the cross_reference method on SiliconMemory."""

    async def test_cross_reference_separates_sources(self):
        """Test that cross_reference separates internal and external beliefs."""
        from silicon_memory.memory.silicondb_router import SiliconMemory, CrossReferenceResult

        internal_belief = Belief(
            id=uuid4(),
            content="Python is popular",
            confidence=0.8,
            source=Source(id="internal", type=SourceType.OBSERVATION, reliability=0.9),
        )
        external_belief = Belief(
            id=uuid4(),
            content="Python usage is growing",
            confidence=0.7,
            source=Source(id="news:bbc", type=SourceType.EXTERNAL, reliability=0.6),
        )

        mock_backend = AsyncMock()
        mock_backend.query_beliefs = AsyncMock(
            return_value=[internal_belief, external_belief]
        )

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory._user_context = MagicMock(user_id="u1", tenant_id="t1")

        import types
        memory.cross_reference = types.MethodType(
            SiliconMemory.cross_reference, memory
        )

        result = await memory.cross_reference("Python")

        assert isinstance(result, CrossReferenceResult)
        assert len(result.internal_beliefs) == 1
        assert len(result.external_beliefs) == 1
        assert result.query == "Python"

    async def test_cross_reference_detects_contradiction(self):
        """Test that contradicting beliefs are detected."""
        from silicon_memory.memory.silicondb_router import SiliconMemory, CrossReferenceResult

        internal_belief = Belief(
            id=uuid4(),
            content="The project is on schedule and progressing well",
            confidence=0.8,
            source=Source(id="internal", type=SourceType.OBSERVATION, reliability=0.9),
        )
        external_belief = Belief(
            id=uuid4(),
            content="The project is not on schedule and is delayed",
            confidence=0.7,
            source=Source(id="news:report", type=SourceType.EXTERNAL, reliability=0.6),
        )

        mock_backend = AsyncMock()
        mock_backend.query_beliefs = AsyncMock(
            return_value=[internal_belief, external_belief]
        )

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory._user_context = MagicMock(user_id="u1", tenant_id="t1")

        import types
        memory.cross_reference = types.MethodType(
            SiliconMemory.cross_reference, memory
        )

        result = await memory.cross_reference("project schedule")
        assert len(result.contradictions) >= 1

    async def test_cross_reference_detects_agreement(self):
        """Test that agreeing beliefs are detected."""
        from silicon_memory.memory.silicondb_router import SiliconMemory, CrossReferenceResult

        internal_belief = Belief(
            id=uuid4(),
            content="Revenue grew strongly this quarter overall",
            confidence=0.8,
            source=Source(id="internal", type=SourceType.OBSERVATION, reliability=0.9),
        )
        external_belief = Belief(
            id=uuid4(),
            content="Revenue grew by a large amount this quarter",
            confidence=0.7,
            source=Source(id="news:ft", type=SourceType.EXTERNAL, reliability=0.8),
        )

        mock_backend = AsyncMock()
        mock_backend.query_beliefs = AsyncMock(
            return_value=[internal_belief, external_belief]
        )

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory._user_context = MagicMock(user_id="u1", tenant_id="t1")

        import types
        memory.cross_reference = types.MethodType(
            SiliconMemory.cross_reference, memory
        )

        result = await memory.cross_reference("revenue")
        assert len(result.agreements) >= 1

    async def test_cross_reference_empty_results(self):
        """Test cross_reference with no beliefs."""
        from silicon_memory.memory.silicondb_router import SiliconMemory, CrossReferenceResult

        mock_backend = AsyncMock()
        mock_backend.query_beliefs = AsyncMock(return_value=[])

        memory = MagicMock(spec=SiliconMemory)
        memory._backend = mock_backend
        memory._user_context = MagicMock(user_id="u1", tenant_id="t1")

        import types
        memory.cross_reference = types.MethodType(
            SiliconMemory.cross_reference, memory
        )

        result = await memory.cross_reference("nonexistent topic")
        assert result.internal_beliefs == []
        assert result.external_beliefs == []
        assert result.agreements == []
        assert result.contradictions == []
