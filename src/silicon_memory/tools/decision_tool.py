"""Decision support tool for LLM function calling integration."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from silicon_memory.decision.synthesis import DecisionBriefGenerator

if TYPE_CHECKING:
    from silicon_memory.memory.silicondb_router import SiliconMemory
    from silicon_memory.reflection.llm import LLMProvider


class DecisionTool:
    """LLM-callable tool for decision support.

    Provides a function-calling interface for LLMs to request
    structured decision briefs from memory.

    Example:
        >>> tool = DecisionTool(memory)
        >>> brief = await tool.invoke(question="Should we use PostgreSQL?")
    """

    def __init__(
        self,
        memory: "SiliconMemory",
        llm_provider: "LLMProvider | None" = None,
    ) -> None:
        self._generator = DecisionBriefGenerator(memory)
        self._llm_provider = llm_provider

    async def invoke(
        self,
        question: str,
        max_beliefs: int = 30,
        max_precedents: int = 5,
    ) -> dict[str, Any]:
        """Generate a decision brief.

        Args:
            question: The decision question to analyze
            max_beliefs: Maximum beliefs to retrieve
            max_precedents: Maximum past decisions to include

        Returns:
            Dictionary with the decision brief data
        """
        brief = await self._generator.generate(
            question=question,
            llm_provider=self._llm_provider,
            max_beliefs=max_beliefs,
            max_precedents=max_precedents,
        )
        return brief.to_dict()

    @staticmethod
    def get_openai_schema() -> dict[str, Any]:
        """Get OpenAI function calling schema for this tool."""
        return {
            "name": "decision_support",
            "description": (
                "Analyze a decision question using memory. Returns a structured "
                "brief with relevant evidence, options, risks, and recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The decision question to analyze",
                    },
                    "max_beliefs": {
                        "type": "integer",
                        "description": "Maximum beliefs to retrieve (default: 30)",
                    },
                    "max_precedents": {
                        "type": "integer",
                        "description": "Maximum past decisions to include (default: 5)",
                    },
                },
                "required": ["question"],
            },
        }
