"""Knowledge base utilities for Psycho-World."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

from psycho_world.utils.schemas import BeliefState


@dataclass
class COKERow:
    node_id: str
    situation: str
    clue: str
    thought: str
    emotion: str

    def to_text(self) -> str:
        return (
            f"Situation: {self.situation}\n"
            f"Clue: {self.clue}\n"
            f"Thought: {self.thought}\n"
            f"Emotion: {self.emotion}"
        )


class DKBRetriever:
    """Hybrid retriever combining vector and graph lookups."""

    def __init__(
        self,
        persist_directory: str,
        graph_uri: str,
        graph_user: str,
        graph_password: str,
    ) -> None:
        self.embedding = OpenAIEmbeddings()
        self.vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding,
        )
        self.graph_driver = GraphDatabase.driver(graph_uri, auth=(graph_user, graph_password))

    def retrieve_similar_chains(self, belief_state: BeliefState, k: int = 3) -> List[str]:
        query = (
            "Psychological situation describing emotion "
            f"{belief_state.emotion} with distortion {belief_state.cognitive_distortion}"
        )
        docs = self.vectordb.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def expand_thought_causes(self, emotion: str) -> List[Dict[str, str]]:
        """Return graph neighbors that lead to a given emotion."""

        cypher = (
            "MATCH (s:Situation)-[:LEADS_TO]->(t:Thought)-[:EVOQUES]->(e:Emotion {name: $emotion}) "
            "RETURN s.description as situation, t.description as thought"
        )
        with self.graph_driver.session() as session:
            result = session.run(cypher, emotion=emotion)
            return [record.data() for record in result]

    def close(self) -> None:
        self.graph_driver.close()


class PKBRetriever:
    """Lightweight personalized memory fetcher (backed by MemGPT)."""

    def __init__(self, memgpt_client) -> None:
        self.client = memgpt_client

    def fetch_profile(self, user_id: str) -> Dict[str, str]:
        return self.client.get_core_memory(user_id)

    def update_profile(self, user_id: str, fields: Dict[str, str]) -> None:
        self.client.update_core_memory(user_id, fields)
