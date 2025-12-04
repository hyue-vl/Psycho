"""
ETL pipeline for transforming the COKE Situation->Thought->Emotion chains into
dual knowledge stores (vector + graph).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def load_coke_chains(json_path: Path) -> List[Dict]:
    """
    Load the raw COKE JSON file.
    """
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("COKE JSON must contain a list of chains.")
    return data


def _chain_to_document(chain: Dict) -> Document:
    """
    Convert a cognitive chain to a LangChain Document.
    """
    node_id = chain.get("id") or chain.get("chain_id")
    text = (
        f"Situation: {chain.get('situation')}\n"
        f"Thought: {chain.get('thought')} ({chain.get('distortion')})\n"
        f"Emotion: {chain.get('emotion')}"
    )
    metadata = {k: v for k, v in chain.items() if k not in {"situation", "thought", "emotion"}}
    metadata["node_id"] = node_id
    return Document(page_content=text, metadata=metadata)


def build_coke_vector_store(json_path: Path, persist_directory: Path) -> Chroma:
    """
    Build or update a Chroma vector index with COKE chains.
    """
    chains = load_coke_chains(json_path)
    documents = [_chain_to_document(chain) for chain in chains]

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )
    vectordb.persist()
    return vectordb


@dataclass
class Neo4jGraphBuilder:
    """
    Lightweight helper for inserting cognitive chains into Neo4j.
    """

    uri: str
    user: str
    password: str

    def __post_init__(self) -> None:
        from neo4j import GraphDatabase  # local import to keep optional

        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self._driver.close()

    def ingest_chains(self, chains: Iterable[Dict]) -> None:
        cypher = """
        MERGE (s:Situation {text: $situation})
        MERGE (t:Thought {text: $thought})
        MERGE (e:Emotion {name: $emotion})
        MERGE (s)-[:TRIGGERS {distortion: $distortion, node_id: $node_id}]->(t)
        MERGE (t)-[:EVOKES {intensity: $intensity}]->(e)
        """
        with self._driver.session() as session:
            for chain in chains:
                payload = {
                    "situation": chain.get("situation"),
                    "thought": chain.get("thought"),
                    "emotion": chain.get("emotion"),
                    "distortion": chain.get("distortion"),
                    "intensity": chain.get("intensity", 0),
                    "node_id": chain.get("id"),
                }
                session.run(cypher, **payload)


__all__ = ["build_coke_vector_store", "Neo4jGraphBuilder", "load_coke_chains"]
