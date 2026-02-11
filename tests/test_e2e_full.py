"""Full end-to-end tests with real SiliconDB + real LLM.

These tests exercise the complete stack:
- Real SiliconDB storage (not mocked)
- Real LLM at localhost:8000 (OpenAI-compatible)
- MemoryTool / DecisionTool interfaces
- Meeting ingestion with LLM segmentation
- News ingestion with LLM claim extraction
- Email ingestion with LLM action item extraction
- Decision synthesis with LLM reasoning

Run with:
    SILICONDB_LIBRARY_PATH=/path/to/lib LLM_BASE_URL=http://localhost:8000 \
        pytest tests/test_e2e_full.py -v

The LLM server must be running and OpenAI-compatible (vLLM, llama.cpp, etc).
"""

from __future__ import annotations

import json
import os
from uuid import uuid4

import httpx
import pytest

from silicon_memory.core.types import Belief, Source, SourceType, Triplet
from silicon_memory.core.decision import Assumption, Decision
from silicon_memory.memory.silicondb_router import RecallContext
from silicon_memory.tools.memory_tool import MemoryTool, MemoryAction
from silicon_memory.tools.decision_tool import DecisionTool

from tests.llm_provider import LocalLLMProvider


# ============================================================================
# Skip conditions
# ============================================================================

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000")

def _llm_available() -> bool:
    """Check if the LLM server is reachable."""
    try:
        resp = httpx.get(f"{LLM_BASE_URL}/v1/models", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("SILICONDB_LIBRARY_PATH"),
        reason="SILICONDB_LIBRARY_PATH not set",
    ),
    pytest.mark.skipif(
        not _llm_available(),
        reason=f"LLM server not available at {LLM_BASE_URL}",
    ),
]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def llm():
    """Create a LocalLLMProvider for the test."""
    provider = LocalLLMProvider(base_url=LLM_BASE_URL)
    yield provider
    await provider.close()


@pytest.fixture
def sample_source() -> Source:
    return Source(
        id="test-source",
        type=SourceType.OBSERVATION,
        reliability=0.9,
        metadata={"name": "E2E Test"},
    )


# ============================================================================
# SM-2: Decision Synthesis with real LLM
# ============================================================================


class TestDecisionSynthesisWithLLM:
    """Test decision synthesis using a real LLM."""

    @pytest.mark.asyncio
    async def test_llm_produces_structured_brief(self, silicon_memory, sample_source, llm):
        """LLM synthesizes a structured decision brief from real beliefs."""
        # Store diverse beliefs about a topic
        beliefs = [
            ("PostgreSQL is a mature relational database", 0.95),
            ("PostgreSQL supports ACID transactions", 0.9),
            ("MongoDB is good for unstructured data", 0.85),
            ("MongoDB scales horizontally easily", 0.8),
            ("Our team has strong SQL experience", 0.9),
            ("The project requires complex joins", 0.85),
        ]
        for content, conf in beliefs:
            await silicon_memory.commit_belief(
                Belief(id=uuid4(), content=content, confidence=conf, source=sample_source)
            )

        # Synthesize with real LLM
        tool = DecisionTool(silicon_memory, llm_provider=llm)
        result = await tool.invoke(question="Should we use PostgreSQL or MongoDB for the new service?")

        assert isinstance(result, dict)
        assert result["question"] == "Should we use PostgreSQL or MongoDB for the new service?"
        # LLM should produce a meaningful recommendation
        assert result["recommendation"], "LLM should produce a recommendation"
        assert result["confidence_in_recommendation"] > 0
        # Should have a summary
        assert "summary" in result
        assert len(result["summary"]) > 20, "Summary should be substantive"
        # Should be JSON-serializable
        json.dumps(result)

    @pytest.mark.asyncio
    async def test_llm_brief_with_contradicting_beliefs(self, silicon_memory, sample_source, llm):
        """LLM handles contradicting beliefs in synthesis."""
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Microservices improve team autonomy", confidence=0.85, source=sample_source)
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Microservices add significant operational complexity", confidence=0.8, source=sample_source)
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Our team is small with limited DevOps experience", confidence=0.9, source=sample_source)
        )

        tool = DecisionTool(silicon_memory, llm_provider=llm)
        result = await tool.invoke(question="Should we adopt microservices architecture?")

        assert isinstance(result, dict)
        assert result["recommendation"]
        # Should identify risks given the contradictions
        assert result.get("risk_count", 0) > 0 or len(result.get("options", [])) > 1


# ============================================================================
# SM-5: Meeting Ingestion with real LLM
# ============================================================================


class TestMeetingIngestionWithLLM:
    """Test meeting transcript ingestion with real LLM segmentation."""

    @pytest.mark.asyncio
    async def test_ingest_meeting_with_llm_segmentation(self, silicon_memory, llm):
        """LLM segments and extracts action items from a meeting transcript."""
        from silicon_memory.ingestion.meeting import MeetingTranscriptAdapter

        transcript = (
            "Alice: Good morning everyone. Let's discuss the Q2 roadmap.\n"
            "Bob: I think we should prioritize the API redesign. The current REST API is slow.\n"
            "Alice: Agreed. What about the frontend migration to React?\n"
            "Carol: The React migration is 60% complete. We need two more sprints.\n"
            "Bob: I'll draft the API specification by Friday.\n"
            "Carol: And I'll finish the component library by next Wednesday.\n"
            "Alice: Great. Let's also discuss the database upgrade.\n"
            "Bob: We should move to PostgreSQL 16 for the JSON improvements.\n"
            "Alice: Bob, can you evaluate the migration effort?\n"
            "Bob: Sure, I'll have an estimate by Monday.\n"
        )

        adapter = MeetingTranscriptAdapter()
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=transcript,
            metadata={
                "meeting_id": "q2-roadmap",
                "title": "Q2 Roadmap Discussion",
            },
            llm_provider=llm,
        )

        assert result.experiences_created > 0, "Should create experiences from transcript"

        # Recall what was discussed
        ctx = RecallContext(
            query="API redesign roadmap",
            max_experiences=10,
        )
        response = await silicon_memory.recall(ctx)
        assert response.total_items > 0

    @pytest.mark.asyncio
    async def test_ingest_then_synthesize_decision(self, silicon_memory, sample_source, llm):
        """Full workflow: ingest meeting → store facts → synthesize decision with LLM."""
        from silicon_memory.ingestion.meeting import MeetingTranscriptAdapter

        # Step 1: Ingest a meeting
        transcript = (
            "Alice: We need to choose a CI/CD platform.\n"
            "Bob: GitHub Actions is free for open source and integrates well.\n"
            "Carol: GitLab CI has better self-hosted support.\n"
            "Alice: Our repos are on GitHub. Integration matters.\n"
            "Bob: GitHub Actions also has a large marketplace of reusable workflows.\n"
        )

        adapter = MeetingTranscriptAdapter()
        await silicon_memory.ingest_from(
            adapter=adapter,
            content=transcript,
            metadata={"meeting_id": "cicd-decision"},
            llm_provider=llm,
        )

        # Step 2: Add some structured facts
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Our repositories are hosted on GitHub", confidence=0.95, source=sample_source)
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="GitHub Actions is free for public repositories", confidence=0.9, source=sample_source)
        )

        # Step 3: Synthesize a decision with LLM
        tool = DecisionTool(silicon_memory, llm_provider=llm)
        result = await tool.invoke(question="Which CI/CD platform should we use?")

        assert isinstance(result, dict)
        assert result["recommendation"]
        assert result["total_evidence"] > 0


# ============================================================================
# SM-6: News Ingestion with real LLM
# ============================================================================


class TestNewsIngestionWithLLM:
    """Test news article ingestion with real LLM claim extraction."""

    @pytest.mark.asyncio
    async def test_ingest_news_with_llm_extraction(self, silicon_memory, sample_source, llm):
        """LLM extracts claims from a news article."""
        from silicon_memory.ingestion.news import NewsArticleAdapter

        # Store an internal belief
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Kubernetes is the standard for container orchestration", confidence=0.9, source=sample_source)
        )

        # Ingest a news article with LLM
        article = json.dumps({
            "title": "Rise of Serverless Computing",
            "body": (
                "Serverless computing is gaining momentum as an alternative to Kubernetes "
                "for many workloads. Companies report 40% cost savings by moving simple "
                "microservices to serverless platforms. However, cold start latency remains "
                "a challenge for latency-sensitive applications."
            ),
        })

        adapter = NewsArticleAdapter()
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=article,
            metadata={"source_name": "TechNews", "credibility": 0.75},
            llm_provider=llm,
        )

        assert result.experiences_created > 0 or result.beliefs_created > 0

        # Cross-reference internal vs external beliefs
        xref = await silicon_memory.cross_reference(
            "container orchestration serverless",
            min_confidence=0.2,
        )
        assert xref.query == "container orchestration serverless"


# ============================================================================
# Full Cognitive Workflow with LLM
# ============================================================================


class TestFullWorkflowWithLLM:
    """Complete cognitive workflow: learn → ingest → decide → reflect."""

    @pytest.mark.asyncio
    async def test_learn_ingest_decide(self, silicon_memory, sample_source, llm):
        """Full cognitive loop with real LLM at every stage."""
        memory_tool = MemoryTool(silicon_memory)
        decision_tool = DecisionTool(silicon_memory, llm_provider=llm)

        # LEARN: Store structured knowledge
        facts = [
            ("Python has excellent async support with asyncio", 0.9),
            ("Go has built-in concurrency with goroutines", 0.9),
            ("Python has a larger ecosystem of data science libraries", 0.95),
            ("Go compiles to native binaries with fast startup", 0.9),
            ("Our team is experienced in Python", 0.85),
            ("The service needs low latency response times", 0.8),
        ]
        for content, conf in facts:
            await memory_tool.invoke("store_fact", content=content, confidence=conf)

        # INGEST: Process a meeting transcript with LLM
        from silicon_memory.ingestion.meeting import MeetingTranscriptAdapter

        transcript = (
            "Alice: For the new API gateway, should we use Python or Go?\n"
            "Bob: Go would give us better latency. But Python is faster to develop.\n"
            "Alice: What about maintenance? We have more Python developers.\n"
            "Bob: True. But the gateway is performance-critical.\n"
        )

        adapter = MeetingTranscriptAdapter()
        await silicon_memory.ingest_from(
            adapter=adapter,
            content=transcript,
            metadata={"meeting_id": "language-decision"},
            llm_provider=llm,
        )

        # DECIDE: Synthesize with real LLM
        brief = await decision_tool.invoke(
            question="Should we use Python or Go for the API gateway?"
        )

        assert isinstance(brief, dict)
        assert brief["question"] == "Should we use Python or Go for the API gateway?"
        assert brief["recommendation"], "LLM should give a recommendation"
        assert brief["total_evidence"] > 0

        # ACT: Store the decision
        await memory_tool.invoke(
            "store_decision",
            title=f"API gateway language: {brief['recommendation'][:50]}",
            description=brief.get("summary", "Based on LLM analysis"),
            decided_by="team",
        )

        # VERIFY: Everything is retrievable
        recall = await memory_tool.invoke(
            "recall",
            query="API gateway language Python Go",
            max_facts=10,
            max_experiences=5,
        )
        assert recall.success

        decisions = await memory_tool.invoke(
            "recall_decisions",
            query="API gateway",
        )
        assert decisions.success

    @pytest.mark.asyncio
    async def test_response_quality(self, silicon_memory, sample_source, llm):
        """Verify LLM produces quality decision analysis."""
        # Store beliefs with clear trade-offs
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="React has the largest community and job market", confidence=0.95, source=sample_source)
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Svelte has significantly better performance benchmarks", confidence=0.85, source=sample_source)
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="Our team has 3 years of React experience", confidence=0.9, source=sample_source)
        )
        await silicon_memory.commit_belief(
            Belief(id=uuid4(), content="The project has strict performance requirements", confidence=0.8, source=sample_source)
        )

        tool = DecisionTool(silicon_memory, llm_provider=llm)
        result = await tool.invoke(question="Should we use React or Svelte for the new frontend?")

        # Quality checks on the LLM output
        assert result["recommendation"]
        assert result["confidence_in_recommendation"] > 0
        assert result["confidence_in_recommendation"] <= 1.0

        # Should have options if LLM is working well
        if "options" in result and result["options"]:
            for opt in result["options"]:
                assert "title" in opt
                assert "description" in opt


# ============================================================================
# Email Ingestion with real LLM
# ============================================================================


class TestEmailIngestionWithLLM:
    """Test email ingestion with real SiliconDB + real LLM action item extraction."""

    @pytest.mark.asyncio
    async def test_ingest_raw_rfc2822_email(self, silicon_memory, llm):
        """Ingest a raw RFC 2822 email and verify experiences are created."""
        from silicon_memory.ingestion.email import EmailAdapter

        raw_email = (
            "From: alice@example.com\n"
            "To: bob@example.com, carol@example.com\n"
            "Subject: Q3 Planning - Action Items\n"
            "Date: Mon, 10 Feb 2025 09:30:00 +0000\n"
            "Message-ID: <abc123@example.com>\n"
            "\n"
            "Hi team,\n"
            "\n"
            "Following up on our planning session. Here are the key items:\n"
            "\n"
            "1. Bob, please finalize the API specification by Friday.\n"
            "2. Carol, can you set up the CI/CD pipeline for the new service?\n"
            "3. I'll schedule a review meeting for next Tuesday.\n"
            "\n"
            "Also, we decided to go with PostgreSQL for the new service based on\n"
            "the team's SQL experience and the need for complex joins.\n"
            "\n"
            "Best,\n"
            "Alice\n"
        )

        adapter = EmailAdapter()
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=raw_email,
            metadata={"email_id": "q3-planning-actions"},
            llm_provider=llm,
        )

        assert result.experiences_created > 0, "Should create experience from email"
        assert result.action_items_detected > 0, "LLM should extract action items"
        assert len(result.errors) == 0, f"No errors expected: {result.errors}"

        # Verify the action items make sense
        action_items = result.details.get("action_items", [])
        assert len(action_items) >= 2, "Should find at least 2 action items"

        # Recall the email content
        ctx = RecallContext(query="API specification PostgreSQL", max_experiences=10)
        response = await silicon_memory.recall(ctx)
        assert response.total_items > 0

    @pytest.mark.asyncio
    async def test_ingest_email_thread_with_replies(self, silicon_memory, llm):
        """Ingest an email with reply chain and verify thread splitting."""
        from silicon_memory.ingestion.email import EmailAdapter

        threaded_email = (
            "From: carol@example.com\n"
            "To: alice@example.com, bob@example.com\n"
            "Subject: Re: Database Migration Plan\n"
            "Date: Wed, 12 Feb 2025 14:00:00 +0000\n"
            "Message-ID: <thread3@example.com>\n"
            "In-Reply-To: <thread2@example.com>\n"
            "\n"
            "I've completed the migration script testing. All 500 tables\n"
            "migrated successfully with zero data loss. We're ready to\n"
            "schedule the production migration.\n"
            "\n"
            "Please review the runbook I attached and confirm the maintenance\n"
            "window works for everyone.\n"
            "\n"
            "On Mon, 10 Feb 2025, Bob Smith wrote:\n"
            "> I've started working on the migration scripts. The schema\n"
            "> changes look straightforward but we need to handle the\n"
            "> foreign key constraints carefully.\n"
            ">\n"
            "> Can you test the scripts against the staging database?\n"
            "\n"
            "On Fri, 7 Feb 2025, Alice Johnson wrote:\n"
            "> We need to plan the database migration from MySQL to PostgreSQL.\n"
            "> The main concerns are:\n"
            "> - Data integrity during migration\n"
            "> - Downtime minimization\n"
            "> - Rollback strategy\n"
        )

        adapter = EmailAdapter()
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=threaded_email,
            metadata={"thread_id": "db-migration-thread"},
            llm_provider=llm,
        )

        # Thread should produce multiple experiences (one per message in chain)
        assert result.experiences_created >= 2, (
            f"Thread should create multiple experiences, got {result.experiences_created}"
        )

        # Recall migration-related content
        ctx = RecallContext(query="database migration PostgreSQL", max_experiences=10)
        response = await silicon_memory.recall(ctx)
        assert response.total_items > 0

    @pytest.mark.asyncio
    async def test_ingest_json_email(self, silicon_memory, llm):
        """Ingest a pre-parsed JSON email with LLM action extraction."""
        from silicon_memory.ingestion.email import EmailAdapter

        email_json = json.dumps({
            "from_addr": "manager@company.com",
            "to": ["dev-team@company.com"],
            "subject": "Sprint Retrospective Follow-up",
            "body": (
                "Team,\n\n"
                "Great retrospective yesterday. Key takeaways:\n\n"
                "1. We need to improve our code review turnaround time. "
                "Dave, please set up automated review assignment by next sprint.\n\n"
                "2. The deployment pipeline is too slow. "
                "Sarah should investigate parallel test execution.\n\n"
                "3. Documentation is falling behind. Everyone needs to update "
                "their component docs before the end of the sprint.\n\n"
                "4. Consider adopting trunk-based development to reduce merge conflicts.\n\n"
                "Let's track these in our next standup.\n"
            ),
            "date": "2025-02-11",
            "message_id": "retro-followup-001",
        })

        adapter = EmailAdapter()
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=email_json,
            metadata={"category": "retrospective"},
            llm_provider=llm,
        )

        assert result.experiences_created > 0
        assert result.action_items_detected >= 2, (
            f"Should find multiple action items, got {result.action_items_detected}"
        )

        # Check action items have owners
        action_items = result.details.get("action_items", [])
        owners = [item.get("owner") for item in action_items if item.get("owner")]
        assert len(owners) >= 1, "At least one action item should have an owner"

    @pytest.mark.asyncio
    async def test_email_ingest_then_decision(self, silicon_memory, sample_source, llm):
        """Full workflow: ingest emails about a topic → add beliefs → synthesize decision."""
        from silicon_memory.ingestion.email import EmailAdapter

        # Step 1: Ingest emails discussing a technical choice
        emails = [
            json.dumps({
                "from_addr": "backend-lead@company.com",
                "to": ["architecture@company.com"],
                "subject": "Message Queue Evaluation",
                "body": (
                    "I've evaluated RabbitMQ and Kafka for our event system.\n\n"
                    "RabbitMQ pros: simpler to operate, good for task queues, "
                    "supports complex routing.\n"
                    "RabbitMQ cons: lower throughput for high-volume streams.\n\n"
                    "Kafka pros: excellent throughput, event sourcing support, "
                    "replay capability.\n"
                    "Kafka cons: operational complexity, needs ZooKeeper (or KRaft).\n\n"
                    "Please review and share your thoughts by Thursday."
                ),
                "message_id": "mq-eval-001",
            }),
            json.dumps({
                "from_addr": "devops-lead@company.com",
                "to": ["architecture@company.com"],
                "subject": "Re: Message Queue Evaluation",
                "body": (
                    "Thanks for the analysis. From an ops perspective:\n\n"
                    "We already run Kafka for logging, so we have operational "
                    "experience. Adding RabbitMQ would mean another system to maintain.\n\n"
                    "However, for the order processing use case, RabbitMQ's "
                    "acknowledgment model is more suitable.\n\n"
                    "Could we use Kafka for event streaming and RabbitMQ for "
                    "task queues? Hybrid approach might be best."
                ),
                "message_id": "mq-eval-002",
                "in_reply_to": "mq-eval-001",
            }),
        ]

        adapter = EmailAdapter()
        for email_content in emails:
            await silicon_memory.ingest_from(
                adapter=adapter,
                content=email_content,
                metadata={"thread_id": "mq-evaluation"},
                llm_provider=llm,
            )

        # Step 2: Add structured beliefs
        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                content="Our team has operational experience with Kafka",
                confidence=0.9,
                source=sample_source,
            )
        )
        await silicon_memory.commit_belief(
            Belief(
                id=uuid4(),
                content="Order processing requires reliable message acknowledgment",
                confidence=0.85,
                source=sample_source,
            )
        )

        # Step 3: Synthesize decision with LLM
        tool = DecisionTool(silicon_memory, llm_provider=llm)
        result = await tool.invoke(
            question="Should we use RabbitMQ, Kafka, or both for our messaging infrastructure?"
        )

        assert isinstance(result, dict)
        assert result["recommendation"], "LLM should recommend a messaging strategy"
        assert result["total_evidence"] > 0
        # Should be JSON-serializable
        json.dumps(result)

    @pytest.mark.asyncio
    async def test_email_heuristic_vs_llm_action_items(self, silicon_memory, llm):
        """Compare heuristic and LLM action item extraction on the same email."""
        from silicon_memory.ingestion.email import EmailAdapter, EmailConfig

        email_content = json.dumps({
            "from_addr": "pm@company.com",
            "to": ["team@company.com"],
            "subject": "Release Checklist",
            "body": (
                "Release v2.5 is scheduled for next Monday. Please complete:\n\n"
                "- Alex: update the changelog and release notes\n"
                "- Jordan: run the full regression test suite\n"
                "- Morgan: prepare the rollback procedure\n"
                "- Everyone: review the feature flag configuration\n\n"
                "Can you confirm completion by Friday EOD?\n"
                "Also, please tag your PRs with the release milestone.\n"
            ),
            "message_id": "release-checklist-001",
        })

        # Run with heuristic
        heuristic_adapter = EmailAdapter(config=EmailConfig(extract_action_items=True))
        heuristic_result = await silicon_memory.ingest_from(
            adapter=heuristic_adapter,
            content=email_content,
            metadata={"category": "release-heuristic"},
        )
        heuristic_items = heuristic_result.action_items_detected

        # Run with LLM
        llm_adapter = EmailAdapter(config=EmailConfig(extract_action_items=True))
        llm_result = await silicon_memory.ingest_from(
            adapter=llm_adapter,
            content=email_content,
            metadata={"category": "release-llm"},
            llm_provider=llm,
        )
        llm_items = llm_result.action_items_detected

        # LLM should generally find at least as many as heuristic
        assert llm_items >= 2, f"LLM should find action items, got {llm_items}"
        # Both should find something
        assert heuristic_items >= 1, f"Heuristic should find some items, got {heuristic_items}"


# ============================================================================
# Chat Ingestion with real LLM
# ============================================================================


class TestChatIngestionWithLLM:
    """Test chat platform ingestion with real SiliconDB + real LLM."""

    @pytest.mark.asyncio
    async def test_ingest_slack_channel(self, silicon_memory, llm):
        """Ingest a Slack channel export and verify experiences + knowledge extraction."""
        from silicon_memory.ingestion.slack import SlackAdapter, SlackConfig

        slack_export = json.dumps([
            {"user": "U1", "username": "alice", "text": "We need to decide on the caching strategy for the API.", "ts": "1700000001.000000", "thread_ts": "1700000001.000000"},
            {"user": "U2", "username": "bob", "text": "Redis seems like the best option. It supports pub/sub and data structures.", "ts": "1700000002.000000", "thread_ts": "1700000001.000000"},
            {"user": "U3", "username": "carol", "text": "I'll set up a Redis cluster in staging by Wednesday.", "ts": "1700000003.000000", "thread_ts": "1700000001.000000"},
            {"user": "U1", "username": "alice", "text": "We decided to use Redis for all caching needs. Deadline is end of sprint.", "ts": "1700000004.000000", "thread_ts": "1700000001.000000"},
        ])

        adapter = SlackAdapter(config=SlackConfig(
            user_map={"U1": "Alice", "U2": "Bob", "U3": "Carol"},
        ))
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=slack_export,
            metadata={"channel": "engineering"},
            llm_provider=llm,
        )

        assert result.experiences_created > 0, "Should create experiences from Slack"

        # Recall the content
        ctx = RecallContext(query="caching strategy Redis", max_experiences=10)
        response = await silicon_memory.recall(ctx)
        assert response.total_items > 0

    @pytest.mark.asyncio
    async def test_chat_action_items_with_llm(self, silicon_memory, llm):
        """Verify LLM extracts action items from chat conversations."""
        from silicon_memory.ingestion.slack import SlackAdapter, SlackConfig

        slack_export = json.dumps([
            {"user": "U1", "username": "alice", "text": "Sprint planning notes:", "ts": "1700000001.000000", "thread_ts": "1700000001.000000"},
            {"user": "U2", "username": "bob", "text": "I'll finish the authentication module by Friday.", "ts": "1700000002.000000", "thread_ts": "1700000001.000000"},
            {"user": "U3", "username": "carol", "text": "Can someone review the PR for the payment integration?", "ts": "1700000003.000000", "thread_ts": "1700000001.000000"},
            {"user": "U1", "username": "alice", "text": "TODO: Update the API documentation before release.", "ts": "1700000004.000000", "thread_ts": "1700000001.000000"},
        ])

        adapter = SlackAdapter(config=SlackConfig(extract_beliefs=False))
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=slack_export,
            metadata={"channel": "sprint-planning"},
            llm_provider=llm,
        )

        assert result.action_items_detected >= 2, (
            f"LLM should find multiple action items, got {result.action_items_detected}"
        )

    @pytest.mark.asyncio
    async def test_chat_belief_extraction_with_llm(self, silicon_memory, sample_source, llm):
        """Verify LLM extracts decisions and knowledge from chat."""
        from silicon_memory.ingestion.slack import SlackAdapter, SlackConfig

        slack_export = json.dumps([
            {"user": "U1", "username": "alice", "text": "After benchmarking, we decided to use FastAPI over Flask for the new service.", "ts": "1700000001.000000", "thread_ts": "1700000001.000000"},
            {"user": "U2", "username": "bob", "text": "Confirmed. FastAPI gives us 3x better throughput on our benchmarks.", "ts": "1700000002.000000", "thread_ts": "1700000001.000000"},
            {"user": "U3", "username": "carol", "text": "FYI: The migration deadline is March 15th.", "ts": "1700000003.000000", "thread_ts": "1700000001.000000"},
        ])

        adapter = SlackAdapter(config=SlackConfig(extract_action_items=False))
        result = await silicon_memory.ingest_from(
            adapter=adapter,
            content=slack_export,
            metadata={"channel": "architecture"},
            llm_provider=llm,
        )

        assert result.decisions_detected >= 1, (
            f"Should extract decisions/knowledge, got {result.decisions_detected}"
        )

    @pytest.mark.asyncio
    async def test_heuristic_vs_llm_chat_beliefs(self, silicon_memory, llm):
        """Compare heuristic and LLM belief extraction on the same chat data."""
        from silicon_memory.ingestion.slack import SlackAdapter, SlackConfig

        slack_export = json.dumps([
            {"user": "U1", "username": "alice", "text": "We decided to adopt trunk-based development.", "ts": "1700000001.000000", "thread_ts": "1700000001.000000"},
            {"user": "U2", "username": "bob", "text": "Agreed, let's go with feature flags instead of long-lived branches.", "ts": "1700000002.000000", "thread_ts": "1700000001.000000"},
            {"user": "U3", "username": "carol", "text": "FYI: CI pipeline now runs in under 5 minutes.", "ts": "1700000003.000000", "thread_ts": "1700000001.000000"},
            {"user": "U1", "username": "alice", "text": "Deadline is next Monday for the first trunk-based release.", "ts": "1700000004.000000", "thread_ts": "1700000001.000000"},
        ])

        # Heuristic
        h_adapter = SlackAdapter(config=SlackConfig(extract_action_items=False))
        h_result = await silicon_memory.ingest_from(
            adapter=h_adapter,
            content=slack_export,
            metadata={"channel": "heuristic-test"},
        )

        # LLM
        l_adapter = SlackAdapter(config=SlackConfig(extract_action_items=False))
        l_result = await silicon_memory.ingest_from(
            adapter=l_adapter,
            content=slack_export,
            metadata={"channel": "llm-test"},
            llm_provider=llm,
        )

        assert h_result.decisions_detected >= 1, "Heuristic should find decisions"
        assert l_result.decisions_detected >= 1, "LLM should find decisions"
