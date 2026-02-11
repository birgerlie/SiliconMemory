"""Example: Using Silicon Memory as LLM function tools."""

import asyncio
import json
from silicon_memory import SiliconMemory, MemoryTool, QueryTool


async def main():
    async with SiliconMemory("/tmp/llm_tools_example") as memory:

        # Create tools
        memory_tool = MemoryTool(memory)
        query_tool = QueryTool(memory)

        # =================================================================
        # GET OPENAI FUNCTION SCHEMA
        # =================================================================

        schema = MemoryTool.get_openai_schema()
        print("OpenAI Function Schema:")
        print(json.dumps(schema, indent=2))

        # =================================================================
        # SIMULATE LLM FUNCTION CALLS
        # =================================================================

        # LLM decides to store a fact
        response = await memory_tool.invoke(
            action="store_fact",
            subject="Machine Learning",
            predicate="is a subset of",
            object="Artificial Intelligence",
            confidence=0.95,
            tags=["ml", "ai"],
        )
        print(f"\nstore_fact: {response.data}")

        # LLM stores an experience
        response = await memory_tool.invoke(
            action="store_experience",
            content="Explained ML vs AI distinction to user",
            outcome="User understood the relationship",
            importance=0.7,
        )
        print(f"store_experience: {response.data}")

        # LLM recalls relevant context
        response = await memory_tool.invoke(
            action="recall",
            query="machine learning artificial intelligence",
            max_facts=5,
        )
        print(f"recall: found {response.data['total_items']} items")

        # LLM asks what it knows
        response = await memory_tool.invoke(
            action="what_do_you_know",
            query="AI and ML",
        )
        print(f"what_do_you_know: {response.data['belief_count']} beliefs")
        print(f"\nReport:\n{response.data['report']}")

        # =================================================================
        # USING QUERY TOOL FOR VERIFICATION
        # =================================================================

        # Verify a claim
        result = await query_tool.verify_claim(
            "Machine Learning is part of AI",
            min_confidence=0.5,
        )
        print(f"\nClaim verification:")
        print(f"   Status: {result['status']}")
        print(f"   Score: {result['verification_score']:.2f}")
        print(f"   Supporting beliefs: {result['supporting_beliefs']}")


if __name__ == "__main__":
    asyncio.run(main())
