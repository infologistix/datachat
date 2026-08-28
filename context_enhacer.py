import asyncio
from typing import List, TYPE_CHECKING
from vanna.core.enhancer.base import LlmContextEnhancer
from vanna.integrations.chromadb.agent_memory import ChromaAgentMemory

if TYPE_CHECKING:
    from vanna.core.user.models import User
    from vanna.core.llm.models import LlmMessage


class TaggedLlmContextEnhancer(LlmContextEnhancer):
    def __init__(self, agent_memory: ChromaAgentMemory, rule_limit: int = 5, ddl_limit: int = 5):
        self.agent_memory = agent_memory
        self.rule_limit = rule_limit
        self.ddl_limit = ddl_limit

    def _query_by_type_sync(self, query: str, memory_type: str, limit: int) -> List[str]:
        # NOTE: relies on ChromaAgentMemory internals (_get_collection is private/Chroma-specific)
        collection = self.agent_memory._get_collection()
        results = collection.query(
            query_texts=[query],
            n_results=limit,
            where={"$and": [{"is_text_memory": True}, {"memory_type": memory_type}]},
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        return [meta.get("content", "") for meta in results["metadatas"][0]]

    async def _query_by_type(self, query: str, memory_type: str, limit: int) -> List[str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.agent_memory._executor, self._query_by_type_sync, query, memory_type, limit
        )

    async def enhance_system_prompt(self, system_prompt: str, user_message: str, user: "User") -> str:
        try:
            rules = await self._query_by_type(user_message, "rule", self.rule_limit)
            ddl = await self._query_by_type(user_message, "ddl", self.ddl_limit)

            if not rules and not ddl:
                return system_prompt

            section = "\n\n## Relevant Context from Memory\n\n"
            if rules:
                section += "### Business rules (always apply if relevant):\n"
                section += "\n".join(f"• {r}" for r in rules) + "\n\n"
            if ddl:
                section += "### Relevant schema:\n"
                section += "\n".join(f"• {d}" for d in ddl) + "\n"

            return system_prompt + section

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Context enhancement failed: {e}")
            return system_prompt

    async def enhance_user_messages(self, messages: List["LlmMessage"], user: "User") -> List["LlmMessage"]:
        return messages