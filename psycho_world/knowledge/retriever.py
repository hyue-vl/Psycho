"""
Retrieval helpers for the dual knowledge base design (DKB + PKB).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class DKBRetriever:
    """
    Vector-store retriever over the COKE-derived domain knowledge.
    """

    def __init__(self, persist_directory: str = "./data/coke_db"):
        self.embedding = OpenAIEmbeddings()
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding,
        )

    def retrieve_similar_chains(self, belief_state: Dict, k: int = 3) -> List[str]:
        """
        Query vector store for cognitively similar chains.
        """
        query = (
            f"Emotion {belief_state.get('emotion')} "
            f"with distortion {belief_state.get('cognitive_distortion')} "
            f"and intent {belief_state.get('intent')}"
        )
        docs = self.vectordb.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]


@dataclass
class PKBMemoryView:
    """
    Adapter that surfaces personalized user context from MemGPT or another memory backend.
    """

    memgpt_client: "MemGPTClient"
    user_id: str

    def fetch_profile(self) -> Dict:
        return self.memgpt_client.get_core_memory(self.user_id)

    def append_session_event(self, event: Dict) -> None:
        self.memgpt_client.append_event(self.user_id, event)


__all__ = ["DKBRetriever", "PKBMemoryView"]
