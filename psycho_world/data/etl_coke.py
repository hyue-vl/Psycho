"""ETL helpers for converting the COKE cognitive chains into hybrid indices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

from psycho_world.configs import COKEPaths
from psycho_world.knowledge.retriever import COKERow


def load_coke_rows(filepath: Path) -> Iterable[COKERow]:
    data = json.loads(filepath.read_text())
    for idx, row in enumerate(data):
        yield COKERow(
            node_id=row.get("id", f"chain_{idx}"),
            situation=row["situation"],
            clue=row.get("clue", ""),
            thought=row["thought"],
            emotion=row["emotion"],
        )


def build_vector_index(rows: Iterable[COKERow], persist_directory: Path) -> None:
    texts = []
    metadatas = []
    ids = []
    for row in rows:
        texts.append(row.to_text())
        metadatas.append({"node_id": row.node_id, "emotion": row.emotion})
        ids.append(row.node_id)
    embeddings = OpenAIEmbeddings()
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        metadatas=metadatas,
        ids=ids,
    ).persist()


def ingest_graph(rows: Iterable[COKERow], paths: COKEPaths) -> None:
    driver = GraphDatabase.driver(paths.graph_uri, auth=(paths.graph_user, paths.graph_password))
    cypher = (
        "MERGE (s:Situation {id: $id}) SET s.description = $situation "
        "MERGE (t:Thought {id: $id}) SET t.description = $thought "
        "MERGE (e:Emotion {name: $emotion}) "
        "MERGE (s)-[:CLUE {text: $clue}]->(t) "
        "MERGE (t)-[:EVOQUES]->(e)"
    )
    with driver.session() as session:
        for row in rows:
            session.run(
                cypher,
                id=row.node_id,
                situation=row.situation,
                thought=row.thought,
                emotion=row.emotion,
                clue=row.clue,
            )
    driver.close()


def run_coke_etl(paths: COKEPaths | None = None) -> None:
    paths = paths or COKEPaths()

    rows = list(load_coke_rows(paths.raw_path))
    build_vector_index(rows, paths.vectordb_dir)
    ingest_graph(rows, paths)


if __name__ == "__main__":
    run_coke_etl()
