"""Example: Building a knowledge base from documents."""

import asyncio
from uuid import uuid4
from silicon_memory import (
    SiliconMemory,
    Belief,
    Triplet,
    Source,
)


# Sample knowledge to ingest
PYTHON_FACTS = [
    ("Python", "was created by", "Guido van Rossum"),
    ("Python", "was first released in", "1991"),
    ("Python", "is known for", "readability"),
    ("Python", "supports", "multiple programming paradigms"),
    ("Python", "has", "dynamic typing"),
    ("Python", "uses", "indentation for code blocks"),
    ("Python 3", "was released in", "2008"),
    ("Python 2", "reached end of life in", "2020"),
    ("pip", "is", "Python package manager"),
    ("PyPI", "is", "Python Package Index"),
    ("Django", "is", "Python web framework"),
    ("Flask", "is", "Python micro web framework"),
    ("NumPy", "is used for", "numerical computing in Python"),
    ("Pandas", "is used for", "data analysis in Python"),
    ("TensorFlow", "is", "machine learning framework for Python"),
    ("PyTorch", "is", "machine learning framework for Python"),
]


async def main():
    async with SiliconMemory("/tmp/knowledge_base_example") as memory:

        # Create source
        source = Source(
            id="python-knowledge",
            type="curated",
            name="Python Knowledge Base",
            reliability=0.9,
        )

        # =================================================================
        # INGEST KNOWLEDGE
        # =================================================================

        print("Ingesting Python knowledge base...")

        for subject, predicate, obj in PYTHON_FACTS:
            belief = Belief(
                id=uuid4(),
                triplet=Triplet(subject=subject, predicate=predicate, object=obj),
                confidence=0.9,
                source=source,
                tags=["python"],
            )
            await memory.commit_belief(belief)

        print(f"Ingested {len(PYTHON_FACTS)} facts")

        # =================================================================
        # QUERY THE KNOWLEDGE BASE
        # =================================================================

        queries = [
            "Python history creator",
            "Python web frameworks",
            "machine learning Python",
            "Python package management",
        ]

        for query in queries:
            proof = await memory.what_do_you_know(query)

            print(f"\nQuery: '{query}'")
            print(f"   Found {len(proof.beliefs)} relevant beliefs")

            for belief in proof.beliefs[:3]:
                if belief.triplet:
                    print(
                        f"   - {belief.triplet.subject} "
                        f"{belief.triplet.predicate} "
                        f"{belief.triplet.object}"
                    )
                else:
                    print(f"   - {belief.content[:50]}...")

        # =================================================================
        # FIND CONTRADICTIONS
        # =================================================================

        # Add a potentially contradicting belief
        contradicting = Belief(
            id=uuid4(),
            triplet=Triplet("Python", "was first released in", "1989"),  # Wrong!
            confidence=0.5,
            source=Source(id="unreliable", type="web", name="Random Blog", reliability=0.3),
        )
        await memory.commit_belief(contradicting)

        # Check for contradictions
        proof = await memory.what_do_you_know("Python release date")

        print(f"\nContradiction check:")
        print(f"   Contradictions found: {len(proof.contradictions)}")

        for b1, b2 in proof.contradictions:
            t1 = b1.triplet
            t2 = b2.triplet
            if t1 and t2:
                print(
                    f"   - '{t1.object}' vs '{t2.object}' "
                    f"(confidence: {b1.confidence:.0%} vs {b2.confidence:.0%})"
                )


if __name__ == "__main__":
    asyncio.run(main())
