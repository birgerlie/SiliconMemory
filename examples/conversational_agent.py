"""Example: Building a conversational agent with memory."""

import asyncio
from uuid import uuid4
from datetime import datetime
from silicon_memory import (
    SiliconMemory,
    Experience,
    RecallContext,
    MemoryTool,
)


class ConversationalAgent:
    """A simple conversational agent with memory.

    This demonstrates how to:
    1. Recall relevant context before responding
    2. Record interactions as experiences
    3. Build up knowledge over time
    """

    def __init__(self, memory: SiliconMemory):
        self.memory = memory
        self.tool = MemoryTool(memory)
        self.session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.turn_count = 0

    async def process_message(self, user_message: str) -> str:
        """Process a user message and generate a response."""
        self.turn_count += 1

        # =================================================================
        # STEP 1: Recall relevant context
        # =================================================================

        ctx = RecallContext(
            query=user_message,
            max_facts=5,
            max_experiences=3,
            max_procedures=2,
            min_confidence=0.5,
        )
        recall = await self.memory.recall(ctx)

        # Build context string for LLM (in real app, send to LLM)
        context_parts = []

        if recall.facts:
            context_parts.append("Relevant facts:")
            for fact in recall.facts:
                context_parts.append(f"  - {fact.content} ({fact.confidence:.0%})")

        if recall.experiences:
            context_parts.append("Previous interactions:")
            for exp in recall.experiences:
                context_parts.append(f"  - {exp.content}")

        if recall.procedures:
            context_parts.append("Relevant procedures:")
            for proc in recall.procedures:
                context_parts.append(f"  - {proc.content}")

        context = "\n".join(context_parts) if context_parts else "No relevant context found."

        # =================================================================
        # STEP 2: Generate response (placeholder - use LLM in real app)
        # =================================================================

        response = f"[Agent would respond based on context]\n\nContext used:\n{context}"

        # =================================================================
        # STEP 3: Record the interaction as an experience
        # =================================================================

        experience = Experience(
            id=uuid4(),
            content=f"User asked: {user_message[:100]}",
            outcome=f"Responded with context from {len(recall.facts)} facts",
            emotional_valence=0.5,
            importance=0.6,
            session_id=self.session_id,
            sequence_id=self.turn_count,
        )
        await self.memory.record_experience(experience)

        # =================================================================
        # STEP 4: Update working memory with conversation state
        # =================================================================

        await self.memory.set_context("last_query", user_message, ttl_seconds=600)
        await self.memory.set_context("turn_count", self.turn_count, ttl_seconds=600)

        return response

    async def learn_fact(
        self, subject: str, predicate: str, obj: str, confidence: float = 0.8
    ):
        """Learn a new fact from the conversation."""
        response = await self.tool.invoke(
            action="store_fact",
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
        )
        return response.success


async def main():
    async with SiliconMemory("/tmp/agent_example") as memory:

        agent = ConversationalAgent(memory)

        # Teach the agent some facts
        await agent.learn_fact("Python", "is great for", "beginners")
        await agent.learn_fact("Python", "has", "extensive libraries")

        # Simulate a conversation
        messages = [
            "What can you tell me about Python?",
            "Is it good for beginners?",
            "What about libraries?",
        ]

        for msg in messages:
            print(f"\nUser: {msg}")
            response = await agent.process_message(msg)
            print(f"\nAgent:\n{response}")
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
