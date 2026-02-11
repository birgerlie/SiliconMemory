"""Basic usage example for Silicon Memory."""

import asyncio
from uuid import uuid4
from silicon_memory import (
    SiliconMemory,
    Belief,
    Experience,
    Procedure,
    Triplet,
    Source,
    RecallContext,
)


async def main():
    # Initialize memory system
    async with SiliconMemory("/tmp/example_memory") as memory:

        # =================================================================
        # STORING BELIEFS (Semantic Memory)
        # =================================================================

        # Create a source for attribution
        docs_source = Source(
            id="python-docs",
            type="documentation",
            name="Python Official Documentation",
            url="https://docs.python.org",
            reliability=0.95,
        )

        # Store a belief as a triplet (subject-predicate-object)
        belief1 = Belief(
            id=uuid4(),
            triplet=Triplet(
                subject="Python",
                predicate="is",
                object="programming language",
            ),
            confidence=0.95,
            source=docs_source,
            tags=["programming", "language"],
        )
        await memory.commit_belief(belief1)

        # Store a belief as free-form content
        belief2 = Belief(
            id=uuid4(),
            content="Python emphasizes code readability and simplicity.",
            confidence=0.9,
            source=docs_source,
            tags=["programming", "philosophy"],
        )
        await memory.commit_belief(belief2)

        print("Stored 2 beliefs about Python")

        # =================================================================
        # RECORDING EXPERIENCES (Episodic Memory)
        # =================================================================

        # Record an interaction experience
        experience = Experience(
            id=uuid4(),
            content="User asked about Python's history and creator",
            outcome="Explained that Guido van Rossum created Python in 1991",
            emotional_valence=0.6,  # Positive interaction
            importance=0.7,
            session_id="demo-session",
        )
        await memory.record_experience(experience)

        print("Recorded 1 experience")

        # =================================================================
        # STORING PROCEDURES (Procedural Memory)
        # =================================================================

        # Store a how-to procedure
        procedure = Procedure(
            id=uuid4(),
            name="Install Python Package",
            description="How to install a Python package using pip",
            steps=[
                "Open terminal or command prompt",
                "Ensure pip is installed: python -m pip --version",
                "Install package: pip install package-name",
                "Verify installation: python -c 'import package'",
            ],
            trigger="install python package pip",
            confidence=0.9,
        )
        await memory.commit_procedure(procedure)

        print("Stored 1 procedure")

        # =================================================================
        # WORKING MEMORY (Short-term Context)
        # =================================================================

        # Set working context (expires after TTL)
        await memory.set_context("current_topic", "Python basics", ttl_seconds=300)
        await memory.set_context("user_level", "beginner", ttl_seconds=300)

        # Retrieve context
        topic = await memory.get_context("current_topic")
        print(f"Working context set: topic={topic}")

        # =================================================================
        # RECALLING MEMORIES
        # =================================================================

        # Recall relevant memories across all types
        ctx = RecallContext(
            query="Python programming basics",
            max_facts=10,
            max_experiences=5,
            max_procedures=3,
            min_confidence=0.5,
        )

        response = await memory.recall(ctx)

        print(f"\nRecall Results for '{ctx.query}':")
        print(f"   Facts: {len(response.facts)}")
        print(f"   Experiences: {len(response.experiences)}")
        print(f"   Procedures: {len(response.procedures)}")
        print(f"   Working context: {response.working_context}")

        # =================================================================
        # KNOWLEDGE PROOFS ("What do you know?")
        # =================================================================

        # Ask what the system knows about a topic
        proof = await memory.what_do_you_know("Python")

        print(f"\nKnowledge Proof for 'Python':")
        print(f"   Total confidence: {proof.total_confidence:.0%}")
        print(f"   Beliefs found: {len(proof.beliefs)}")
        print(f"   Sources: {len(proof.sources)}")
        print(f"   Contradictions: {len(proof.contradictions)}")

        # Print the formatted report
        print("\n" + "=" * 50)
        print(proof.as_report())


if __name__ == "__main__":
    asyncio.run(main())
