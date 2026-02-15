"""E2E tests for the cognitive reflection pipeline.

Tests the full cognitive pipeline:
- Entity-aware extraction
- Source grounding
- Transitive inference (dream forward)
- Backward dreaming (evidence update)
- Hypothesis generation
- Memory consolidation (generalization, decay)

Requires:
- SiliconDB library (SILICONDB_LIBRARY_PATH)
- SiliconServe running on localhost:8000 with qwen3-80b

Run:
    SILICONDB_LIBRARY_PATH=deps/silicondb/build/lib/libSiliconDB.dylib \
    .venv/bin/pytest tests/test_e2e_cognition.py -v -s
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

# Ensure SiliconMemory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from silicon_memory.core.types import (
    Belief,
    BeliefStatus,
    Experience,
    Source,
    SourceType,
    Triplet,
)
from silicon_memory.security.types import UserContext
from silicon_memory.reflection.types import (
    ExperienceGroup,
    Pattern,
    PatternType,
    ReflectionConfig,
)
from silicon_memory.reflection.inference import (
    TransitiveInferenceEngine,
    InferenceResult,
)
from silicon_memory.reflection.consolidation import (
    MemoryConsolidator,
    ConsolidationStats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.asyncio,
]


@pytest.fixture
def user_ctx():
    return UserContext(user_id="test_admin", tenant_id="test_tenant")


@pytest.fixture
async def memory(user_ctx, tmp_path):
    """Create a fresh SiliconMemory instance for testing."""
    from silicon_memory.memory.silicondb_router import SiliconMemory

    db_path = str(tmp_path / "test_cognition.db")
    mem = SiliconMemory(db_path, user_context=user_ctx)
    await mem.__aenter__()
    yield mem
    await mem.__aexit__(None, None, None)


@pytest.fixture
async def llm():
    """Create LLM scheduler connected to local SiliconServe."""
    from silicon_memory.llm.provider import SiliconLLMProvider
    from silicon_memory.llm.scheduler import LLMScheduler

    provider = SiliconLLMProvider(
        base_url="http://localhost:8000/v1", model="qwen3-80b"
    )
    scheduler = LLMScheduler(provider, max_concurrency=4)
    await scheduler.start()
    yield scheduler
    await scheduler.shutdown()


# ---------------------------------------------------------------------------
# Helper: create test experiences
# ---------------------------------------------------------------------------

async def create_test_experiences(
    memory, doc_id: str, sections: list[tuple[str, str]]
) -> list[Experience]:
    """Create experiences from (title, content) pairs for a document."""
    experiences = []
    for i, (title, content) in enumerate(sections):
        exp = Experience(
            id=uuid4(),
            content=content,
            context={
                "document_id": doc_id,
                "title": f"Test Document: {doc_id}",
                "section_title": title,
                "section_index": i,
                "source_type": "document",
            },
            session_id=doc_id,
            user_id=memory.user_context.user_id,
            tenant_id=memory.user_context.tenant_id,
        )
        await memory.record_experience(exp)
        experiences.append(exp)
    return experiences


async def create_test_belief(
    memory,
    subject: str,
    predicate: str,
    obj: str,
    confidence: float = 0.8,
) -> Belief:
    """Create and commit a test belief."""
    belief = Belief(
        id=uuid4(),
        content=f"{subject} {predicate} {obj}",
        triplet=Triplet(subject=subject, predicate=predicate, object=obj),
        confidence=confidence,
        source=Source(
            id="test", type=SourceType.OBSERVATION, reliability=0.9
        ),
        status=BeliefStatus.VALIDATED,
        user_id=memory.user_context.user_id,
        tenant_id=memory.user_context.tenant_id,
    )
    await memory.commit_belief(belief)
    return belief


# ---------------------------------------------------------------------------
# Test: Transitive Inference (no LLM needed)
# ---------------------------------------------------------------------------

class TestTransitiveInference:
    """Test transitive inference engine."""

    async def test_two_hop_chain(self, memory):
        """A→B + B→C → infer A→C."""
        # Setup: Maxwell associated with Epstein, Epstein owned island
        b1 = await create_test_belief(
            memory, "Maxwell", "associated with", "Epstein", 0.9
        )
        b2 = await create_test_belief(
            memory, "Epstein", "owned", "Little St. James", 0.95
        )

        engine = TransitiveInferenceEngine(memory)
        result = await engine.infer_forward([b1], max_hops=2)

        assert result.chains_found >= 1
        assert len(result.inferred_beliefs) >= 1

        # Should infer Maxwell connected to Little St. James
        inferred = result.inferred_beliefs[0]
        assert inferred.triplet is not None
        assert inferred.triplet.subject == "Maxwell"
        assert "Little St. James" in inferred.triplet.object
        assert inferred.confidence < min(b1.confidence, b2.confidence)
        assert inferred.source.id == "transitive_inference"

    async def test_three_hop_chain(self, memory):
        """A→B + B→C + C→D → infer A→D."""
        await create_test_belief(
            memory, "Prince Andrew", "visited", "Epstein", 0.8
        )
        await create_test_belief(
            memory, "Epstein", "owned", "71st Street mansion", 0.9
        )
        b3 = await create_test_belief(
            memory, "71st Street mansion", "located in", "New York", 0.95
        )

        # Start from a new belief about Prince Andrew
        new_belief = await create_test_belief(
            memory, "Prince Andrew", "visited", "Epstein", 0.8
        )

        engine = TransitiveInferenceEngine(memory)
        result = await engine.infer_forward([new_belief], max_hops=3)

        # Should find at least the 2-hop chain
        assert result.chains_found >= 1

    async def test_no_self_referential_chains(self, memory):
        """Should not create A→A chains."""
        await create_test_belief(
            memory, "Epstein", "associated with", "Maxwell", 0.9
        )
        b2 = await create_test_belief(
            memory, "Maxwell", "associated with", "Epstein", 0.9
        )

        engine = TransitiveInferenceEngine(memory)
        result = await engine.infer_forward([b2], max_hops=2)

        # Should not infer "Epstein connected to Epstein"
        for inferred in result.inferred_beliefs:
            assert inferred.triplet.subject.lower() != inferred.triplet.object.lower()

    async def test_backward_dreaming_corroboration(self, memory):
        """New evidence corroborating existing belief boosts confidence."""
        existing = await create_test_belief(
            memory, "Maxwell", "recruited for", "Epstein", 0.7
        )
        new = await create_test_belief(
            memory, "Maxwell", "recruited for", "Epstein", 0.85
        )

        engine = TransitiveInferenceEngine(memory)
        result = await engine.infer_backward([new])

        # Should have updated the existing belief
        assert result.backward_updates >= 1

    async def test_backward_dreaming_contradiction(self, memory):
        """Conflicting evidence should reduce existing confidence."""
        existing = await create_test_belief(
            memory, "Epstein", "died by", "suicide", 0.6
        )
        new = await create_test_belief(
            memory, "Epstein", "died by", "homicide", 0.5
        )

        engine = TransitiveInferenceEngine(memory)
        result = await engine.infer_backward([new])

        assert result.backward_updates >= 1


# ---------------------------------------------------------------------------
# Test: Memory Consolidation (no LLM needed)
# ---------------------------------------------------------------------------

class TestMemoryConsolidation:
    """Test memory consolidation engine."""

    async def test_generalization_subject_pattern(self, memory):
        """Multiple beliefs with same subject+predicate → generalization."""
        await create_test_belief(memory, "Maxwell", "recruited", "Victim A", 0.8)
        await create_test_belief(memory, "Maxwell", "recruited", "Victim B", 0.8)
        await create_test_belief(memory, "Maxwell", "recruited", "Victim C", 0.8)

        consolidator = MemoryConsolidator(memory)
        stats = await consolidator.consolidate()

        assert stats.generalizations_created >= 1

    async def test_generalization_object_pattern(self, memory):
        """Multiple beliefs with same predicate+object → generalization."""
        await create_test_belief(memory, "Person A", "visited", "Epstein Island", 0.8)
        await create_test_belief(memory, "Person B", "visited", "Epstein Island", 0.8)
        await create_test_belief(memory, "Person C", "visited", "Epstein Island", 0.8)

        consolidator = MemoryConsolidator(memory)
        stats = await consolidator.consolidate()

        assert stats.generalizations_created >= 1


# ---------------------------------------------------------------------------
# Test: Full Pipeline (requires LLM)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test the full cognitive pipeline end-to-end."""

    async def test_extract_with_source_grounding(self, memory, llm):
        """Extraction should include source document references."""
        from silicon_memory.reflection.engine import ReflectionEngine

        experiences = await create_test_experiences(memory, "DOC-001", [
            ("Case Caption", (
                "UNITED STATES OF AMERICA v. GHISLAINE MAXWELL, "
                "Defendant. Case No. 20-cr-330 (AJN). "
                "SOUTHERN DISTRICT OF NEW YORK. "
                "Before: The Honorable Alison J. Nathan, United States District Judge."
            )),
            ("Background", (
                "Ghislaine Maxwell was convicted on December 29, 2021, of sex trafficking "
                "of a minor and other offenses related to her role in facilitating "
                "Jeffrey Epstein's sexual abuse of minor girls between 1994 and 2004. "
                "Maxwell was sentenced to 240 months of imprisonment on June 28, 2022. "
                "She was also ordered to pay restitution of $724,416."
            )),
            ("Epstein Connection", (
                "The evidence at trial established that Maxwell had a close personal "
                "and professional relationship with Jeffrey Epstein from approximately "
                "1994 to 2004. During this period, Maxwell recruited, groomed, and "
                "ultimately facilitated Epstein's sexual abuse of multiple minor victims. "
                "Maxwell used her social connections and her position of trust to identify "
                "and befriend vulnerable young girls."
            )),
        ])

        config = ReflectionConfig(
            max_experiences_per_batch=10,
            auto_commit_beliefs=True,
            extraction_chunk_size=8,
            extraction_max_chars=24000,
            extraction_max_items=20,
            extraction_max_tokens=4000,
        )
        engine = ReflectionEngine(memory, llm=llm, config=config)
        result = await engine.reflect()

        # Should have processed experiences
        assert result.experiences_processed == 3

        # Should have found patterns
        assert len(result.patterns_found) > 0

        # Check source grounding
        grounded = 0
        for p in result.patterns_found:
            if p.context.get("source"):
                grounded += 1
        assert grounded > 0, "At least some patterns should have source grounding"

        # Should have committed beliefs
        assert len(result.updated_beliefs) > 0

        # Check pattern types - should have facts, relationships, events
        types_found = {p.type for p in result.patterns_found}
        assert PatternType.FACT in types_found, "Should extract facts"

    async def test_entity_aware_extraction(self, memory, llm):
        """Second extraction should use knowledge from first."""
        from silicon_memory.reflection.engine import ReflectionEngine

        # First: establish some baseline knowledge
        await create_test_belief(
            memory, "Maxwell", "convicted of", "sex trafficking", 0.95
        )
        await create_test_belief(
            memory, "Epstein", "died on", "August 10, 2019", 0.95
        )

        # Then: ingest new document
        experiences = await create_test_experiences(memory, "DOC-002", [
            ("Flight Records", (
                "Flight records show that Ghislaine Maxwell traveled on Jeffrey Epstein's "
                "private aircraft, known as the 'Lolita Express', on numerous occasions "
                "between 1995 and 2005. The flight logs indicate destinations including "
                "Little St. James Island, Palm Beach, New York, and various international "
                "locations including London, Paris, and the US Virgin Islands."
            )),
        ])

        config = ReflectionConfig(
            max_experiences_per_batch=10,
            auto_commit_beliefs=True,
            extraction_chunk_size=8,
            extraction_max_chars=24000,
            extraction_max_items=20,
            extraction_max_tokens=4000,
        )
        engine = ReflectionEngine(memory, llm=llm, config=config)
        result = await engine.reflect()

        # Should have extracted patterns
        assert len(result.patterns_found) > 0

        # Check that extraction mentions entities that are NOT just
        # "Maxwell exists" but actual relationships
        relationship_patterns = [
            p for p in result.patterns_found
            if p.type == PatternType.RELATIONSHIP
        ]
        # Flight records should produce relationship patterns
        print(f"Found {len(relationship_patterns)} relationship patterns")
        for p in relationship_patterns:
            print(f"  {p.description}")

    async def test_full_cognitive_pipeline(self, memory, llm):
        """Test the full pipeline: extract → infer → consolidate."""
        from silicon_memory.reflection.engine import ReflectionEngine

        # Ingest multiple related documents
        await create_test_experiences(memory, "DOC-NETWORK-1", [
            ("Epstein Network", (
                "Jeffrey Epstein maintained a vast network of associates across "
                "finance, politics, and royalty. His black book contained contacts "
                "including Prince Andrew, Bill Clinton, Donald Trump, and numerous "
                "other high-profile individuals. Epstein hosted gatherings at his "
                "properties in New York, Palm Beach, Paris, and the US Virgin Islands."
            )),
        ])

        await create_test_experiences(memory, "DOC-NETWORK-2", [
            ("Royal Connections", (
                "Prince Andrew, Duke of York, had a documented friendship with "
                "Jeffrey Epstein. Andrew visited Epstein at his New York mansion "
                "and was photographed with Virginia Giuffre. Prince Andrew was "
                "later stripped of his military affiliations and royal patronages "
                "as a result of the civil lawsuit filed by Giuffre."
            )),
        ])

        await create_test_experiences(memory, "DOC-NETWORK-3", [
            ("Financial Network", (
                "Epstein's financial holdings included a $77 million mansion in New York, "
                "a private island in the US Virgin Islands (Little St. James), a ranch "
                "in New Mexico, and properties in Paris. His wealth was estimated at "
                "over $577 million. Leslie Wexner, the billionaire founder of L Brands, "
                "was one of Epstein's primary financial backers."
            )),
        ])

        config = ReflectionConfig(
            max_experiences_per_batch=10,
            auto_commit_beliefs=True,
            extraction_chunk_size=8,
            extraction_max_chars=24000,
            extraction_max_items=20,
            extraction_max_tokens=4000,
        )
        engine = ReflectionEngine(memory, llm=llm, config=config)

        # Run reflection
        result = await engine.reflect()

        print(f"\n=== Pipeline Result ===")
        print(f"Experiences processed: {result.experiences_processed}")
        print(f"Patterns found: {len(result.patterns_found)}")
        print(f"Beliefs committed: {len(result.updated_beliefs)}")
        print(f"Contradictions: {len(result.contradictions)}")

        for p in result.patterns_found:
            print(f"  [{p.type.value:15s}] {p.description[:100]}")

        # Verify basic assertions
        assert result.experiences_processed >= 3
        assert len(result.patterns_found) > 0
        assert len(result.updated_beliefs) > 0

        # Run dream for deeper analysis
        dream_stats = await engine.dream()
        print(f"\n=== Dream Result ===")
        for k, v in dream_stats.items():
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Test: Norwegian Royal Family discovery (integration)
# ---------------------------------------------------------------------------

class TestNorwegianRoyalDiscovery:
    """Test discovery of Norwegian royal family connections.

    This test specifically targets the relationship between
    the Norwegian royal family and the Epstein network.
    """

    async def test_discover_royal_connections(self, memory, llm):
        """Discover connections through multi-hop inference."""
        from silicon_memory.reflection.engine import ReflectionEngine

        # Seed with documents that establish the network
        await create_test_experiences(memory, "DOC-ROYALS-1", [
            ("European Royals", (
                "Jeffrey Epstein cultivated relationships with European royalty "
                "and aristocracy. His social circle included Prince Andrew of the "
                "United Kingdom, members of the Norwegian royal family, and various "
                "European socialites. Ghislaine Maxwell, the daughter of media mogul "
                "Robert Maxwell, served as a key facilitator for these connections "
                "through her own social standing in European high society."
            )),
            ("Norway Connections", (
                "Epstein's connections to Scandinavia included visits to Norway "
                "and contacts within the Norwegian elite. Crown Prince Haakon of "
                "Norway was reported to have attended events where Epstein was "
                "present. The Norwegian connection was facilitated through mutual "
                "acquaintances in the financial and philanthropic sectors, "
                "including ties to hedge fund circles and charitable foundations "
                "that operated in Northern Europe."
            )),
            ("Financial Ties", (
                "Epstein's financial network extended into Scandinavian banking "
                "and investment circles. His fund management activities involved "
                "dealings with institutions connected to Nordic wealth management. "
                "The Norwegian Government Pension Fund, one of the world's largest "
                "sovereign wealth funds, had indirect connections through shared "
                "investment managers and financial advisors who also worked with "
                "Epstein's entities."
            )),
        ])

        config = ReflectionConfig(
            max_experiences_per_batch=10,
            auto_commit_beliefs=True,
            extraction_chunk_size=8,
            extraction_max_chars=24000,
            extraction_max_items=20,
            extraction_max_tokens=4000,
        )
        engine = ReflectionEngine(memory, llm=llm, config=config)

        # Run reflection
        result = await engine.reflect()

        print(f"\n=== Norwegian Royal Discovery ===")
        print(f"Patterns: {len(result.patterns_found)}")
        print(f"Beliefs: {len(result.updated_beliefs)}")

        # Look for Norwegian/Scandinavian mentions
        norway_patterns = []
        for p in result.patterns_found:
            desc = p.description.lower()
            if any(kw in desc for kw in [
                "norway", "norwegian", "haakon", "scandina", "nordic"
            ]):
                norway_patterns.append(p)

        print(f"\nNorwegian-related patterns: {len(norway_patterns)}")
        for p in norway_patterns:
            print(f"  [{p.type.value}] {p.description}")

        # All patterns for review
        print(f"\nAll patterns:")
        for p in result.patterns_found:
            src = p.context.get("source", {})
            doc_ref = src.get("document_id", "?") if src else "?"
            print(f"  [{p.type.value:15s}] {p.description[:120]}")
            print(f"    source: {doc_ref}, confidence: {p.confidence:.2f}")

        assert len(result.patterns_found) > 0
        # At least some patterns should mention Norwegian connections
        assert len(norway_patterns) > 0, (
            "Should discover Norwegian royal connections from the documents"
        )

        # Now run transitive inference to find indirect connections
        print(f"\n=== Running Transitive Inference ===")
        dream_stats = await engine.dream()
        print(f"Dream stats: {dream_stats}")
