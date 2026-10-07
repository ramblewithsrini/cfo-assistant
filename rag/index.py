"""A tiny vector index: every chunk with its vector, saved to one JSON file.

Twelve policies need nothing more. At scale this becomes a vector database
(OpenSearch, pgvector, S3 Vectors), but the idea is identical.
"""
import json
from dataclasses import asdict
from pathlib import Path

from .chunk import Chunk, load_chunks
from .embed import cosine


class VectorIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list = []

    def build(self, policies_dir: Path) -> None:
        self.chunks = load_chunks(policies_dir)
        self.embedder.fit([c.text for c in self.chunks])
        self.vectors = [self.embedder.embed(c.text) for c in self.chunks]

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed(query)
        scored = [(c, cosine(q, v)) for c, v in zip(self.chunks, self.vectors)]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

    def save(self, path: Path) -> None:
        data = {
            "embedder": self.embedder.state(),
            "chunks": [asdict(c) for c in self.chunks],
            "vectors": self.vectors,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.embedder.load_state(data["embedder"])
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.vectors = data["vectors"]
