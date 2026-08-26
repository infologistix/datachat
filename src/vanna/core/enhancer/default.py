"""
Default LLM context enhancer implementation using AgentMemory.

This implementation enriches the system prompt with relevant memories
based on the user's initial message.
"""

from typing import TYPE_CHECKING, List, Optional
from .base import LlmContextEnhancer

if TYPE_CHECKING:
    from ..user.models import User
    from ..llm.models import LlmMessage
    from ...capabilities.agent_memory import AgentMemory, TextMemorySearchResult


class DefaultLlmContextEnhancer(LlmContextEnhancer):
    """Default enhancer that uses AgentMemory to add relevant context.

    This enhancer searches the agent's memory for relevant examples and
    tool use patterns based on the user's message, and adds them to the
    system prompt.

    Example:
        agent = Agent(
            llm_service=...,
            agent_memory=agent_memory,
            llm_context_enhancer=DefaultLlmContextEnhancer(agent_memory)
        )
    """

    def __init__(self, agent_memory: Optional["AgentMemory"] = None):
        """Initialize with optional agent memory.

        Args:
            agent_memory: Optional AgentMemory instance. If not provided,
                         enhancement will be skipped.
        """
        self.agent_memory = agent_memory

    async def enhance_system_prompt(
        self, system_prompt: str, user_message: str, user: "User"
    ) -> str:
        """Enhance system prompt with relevant memories.

        Searches agent memory for relevant text memories based on the
        user's message and adds them to the system prompt.

        Args:
            system_prompt: The original system prompt
            user_message: The initial user message
            user: The user making the request

        Returns:
            Enhanced system prompt with relevant examples from memory
        """
        if not self.agent_memory:
            return system_prompt

        try:
            # Import here to avoid circular dependency
            from ..tool import ToolContext
            import uuid

            # Create a temporary context for memory search
            context = ToolContext(
                user=user,
                conversation_id="temp",
                request_id=str(uuid.uuid4()),
                agent_memory=self.agent_memory,
            )

            # Search for relevant text memories based on user message.
            #
            # similarity_threshold must be 0.0 here, not the 0.7 default.
            # ChromaAgentMemory computes similarity as max(0, 1 - distance),
            # which only maps to a [0,1] score for COSINE distance. The
            # collection is created with Chroma's default L2 (squared
            # euclidean) metric, where distances routinely exceed 1 - so
            # 1 - distance goes negative, clamps to 0, and every result is
            # rejected by any threshold above 0. Measured on the live store:
            # the best match for "what is top score of bbl last season" is
            # bronze.b_bbl_boxscore at L2 distance 1.31 -> score 0.0.
            #
            # The ranking itself is correct (Chroma orders by distance and
            # returns the right tables first); it was only the threshold
            # discarding them, leaving the agent with no schema context at
            # all. With 0.0 we keep Chroma's top-N ordering and filter
            # nothing.
            #
            # Note this means no relevance filtering: an off-topic question
            # still pulls in the 5 nearest tables. Fixing that properly needs
            # the collection created with hnsw:space=cosine, where relevant
            # matches score ~0.31-0.47 and unrelated ones ~0.15, making a
            # real threshold (~0.25) possible.
            memories: List[
                "TextMemorySearchResult"
            ] = await self.agent_memory.search_text_memories(
                query=user_message, context=context, limit=5, similarity_threshold=0.0
            )

            if not memories:
                return system_prompt

            # Format memories as context snippets to add to system prompt
            examples_section = "\n\n## Relevant Context from Memory\n\n"
            examples_section += "The following domain knowledge and context from prior interactions may be relevant:\n\n"

            for result in memories:
                memory = result.memory
                examples_section += f"• {memory.content}\n"

            # Append examples to system prompt
            return system_prompt + examples_section

        except Exception as e:
            # If memory search fails, return original prompt
            # Don't fail the entire request due to memory issues
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to enhance system prompt with memories: {e}")
            return system_prompt

    async def enhance_user_messages(
        self, messages: list["LlmMessage"], user: "User"
    ) -> list["LlmMessage"]:
        """Enhance user messages.

        The default implementation doesn't modify user messages.
        Override this to add context to user messages if needed.

        Args:
            messages: The list of messages
            user: The user making the request

        Returns:
            Original list of messages (unmodified)
        """
        return messages
