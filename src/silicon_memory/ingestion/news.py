"""News article ingestion adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from silicon_memory.core.types import Experience, Source, SourceType
from silicon_memory.ingestion.types import IngestionConfig, IngestionResult
from silicon_memory.ingestion._helpers import parse_llm_json_array

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory
    from silicon_memory.reflection.llm import LLMProvider


@dataclass
class NewsArticle:
    """Parsed news article."""

    title: str = ""
    body: str = ""
    source_name: str = ""
    source_url: str = ""
    date: str = ""
    author: str = ""


@dataclass
class NewsIngestionConfig(IngestionConfig):
    """Configuration for news article ingestion."""

    extract_claims: bool = True
    resolve_entities: bool = False
    default_credibility: float = 0.5
    max_claims_per_article: int = 20


class NewsArticleAdapter:
    """Ingests news articles as experiences with source attribution.

    Processing pipeline:
    1. Parse article (from dict or raw text)
    2. Create experience with source_type: EXTERNAL
    3. Extract key claims with credibility weighting
    4. Store claims as low-confidence beliefs
    """

    def __init__(
        self,
        config: NewsIngestionConfig | None = None,
    ) -> None:
        self._config = config or NewsIngestionConfig()

    @property
    def source_type(self) -> str:
        return "news_article"

    async def ingest(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        memory: "SiliconMemory",
        llm_provider: "LLMProvider | None" = None,
    ) -> IngestionResult:
        """Ingest a news article into memory.

        Args:
            content: Article text or JSON with title/body/source fields
            metadata: Should include source_name, source_url, credibility
            memory: SiliconMemory instance
            llm_provider: Optional LLM for claim extraction

        Returns:
            IngestionResult with statistics
        """
        result = IngestionResult(source_type=self.source_type)

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        content = content.strip()
        if not content:
            result.errors.append("Empty article content")
            return result

        # Parse article
        article = self._parse_article(content, metadata)

        if not article.body:
            result.errors.append("No article body found")
            return result

        user_ctx = memory.user_context
        credibility = metadata.get("credibility", self._config.default_credibility)

        # Create experience for the article
        try:
            exp = Experience(
                id=uuid4(),
                content=article.body,
                context={
                    "source_type": "news_article",
                    "source_name": article.source_name,
                    "source_url": article.source_url,
                    "title": article.title,
                    "date": article.date,
                    "author": article.author,
                    "credibility": credibility,
                    **{k: v for k, v in metadata.items()
                       if k not in ("source_name", "source_url", "credibility",
                                    "title", "date", "author")},
                },
                user_id=user_ctx.user_id,
                tenant_id=user_ctx.tenant_id,
            )
            await memory.record_experience(exp)
            result.experiences_created += 1
        except Exception as e:
            result.errors.append(f"Failed to store article experience: {e}")
            return result

        # Extract claims
        if self._config.extract_claims:
            try:
                claims = await self._extract_claims(article, llm_provider)
                claims = claims[:self._config.max_claims_per_article]
                result.details["claims"] = claims
                result.details["claims_count"] = len(claims)

                # Store claims as beliefs with source credibility weighting
                from silicon_memory.core.types import Belief
                source = Source(
                    id=f"news:{article.source_name or 'unknown'}",
                    type=SourceType.EXTERNAL,
                    reliability=credibility,
                    metadata={
                        "source_url": article.source_url,
                        "article_title": article.title,
                    },
                )

                for claim in claims:
                    try:
                        belief = Belief(
                            id=uuid4(),
                            content=claim.get("claim", ""),
                            confidence=min(1.0, credibility * claim.get("confidence", 0.7)),
                            source=source,
                            tags={"news", "external"},
                            user_id=user_ctx.user_id,
                            tenant_id=user_ctx.tenant_id,
                        )
                        await memory.commit_belief(belief)
                    except Exception as e:
                        result.errors.append(f"Failed to store claim: {e}")
            except Exception as e:
                result.errors.append(f"Claim extraction error: {e}")

        return result

    def _parse_article(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> NewsArticle:
        """Parse article from JSON or raw text."""
        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return NewsArticle(
                    title=data.get("title", metadata.get("title", "")),
                    body=data.get("body", data.get("content", "")),
                    source_name=data.get("source_name", metadata.get("source_name", "")),
                    source_url=data.get("source_url", metadata.get("source_url", "")),
                    date=data.get("date", metadata.get("date", "")),
                    author=data.get("author", metadata.get("author", "")),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Treat as raw text
        return NewsArticle(
            title=metadata.get("title", ""),
            body=content,
            source_name=metadata.get("source_name", ""),
            source_url=metadata.get("source_url", ""),
            date=metadata.get("date", ""),
            author=metadata.get("author", ""),
        )

    async def _extract_claims(
        self,
        article: NewsArticle,
        provider: "LLMProvider | None" = None,
    ) -> list[dict[str, Any]]:
        """Extract key claims from article."""
        if provider:
            return await self._extract_claims_llm(article, provider)
        return self._extract_claims_heuristic(article)

    async def _extract_claims_llm(
        self,
        article: NewsArticle,
        provider: "LLMProvider",
    ) -> list[dict[str, Any]]:
        """Use LLM to extract claims from article."""
        text = article.body[:3000]  # Truncate for token limits

        prompt = (
            "Extract the key factual claims from this article. "
            "Return a JSON array where each element has:\n"
            '- "claim": the factual claim\n'
            '- "confidence": how confident this claim appears (0-1)\n'
            "\nOnly return the JSON array, no other text.\n\n"
            f"Title: {article.title}\n\n{text}"
        )

        response = await provider.complete(
            prompt=prompt,
            system="You are a claim extraction engine. Extract factual claims.",
            temperature=0.2,
            max_tokens=2000,
        )

        try:
            return parse_llm_json_array(response)
        except (ValueError, json.JSONDecodeError):
            return []

    def _extract_claims_heuristic(
        self,
        article: NewsArticle,
    ) -> list[dict[str, Any]]:
        """Extract claims using simple heuristics.

        Treats each sentence that makes a factual assertion as a claim.
        """
        claims: list[dict[str, Any]] = []
        sentences = article.body.replace("\n", " ").split(".")

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            # Skip questions and quotes
            if sentence.endswith("?") or sentence.startswith('"'):
                continue
            claims.append({
                "claim": sentence + ".",
                "confidence": 0.6,
            })

        return claims
