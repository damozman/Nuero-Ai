import threading
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, FilterSelector,
    PointIdsList, IsEmptyCondition, PayloadField
)
from sentence_transformers import SentenceTransformer

SHARED = "shared"  # partition every bot can search; matches core.bots.SHARED

_vault = None
_vault_lock = threading.Lock()


def get_vault():
    """Embedded Qdrant locks its folder, so every component shares one vault."""
    global _vault
    with _vault_lock:
        if _vault is None:
            _vault = VectorMemoryVault()
        return _vault


class VectorMemoryVault:
    def __init__(self, db_path="storage/memory_vault.db"):
        self.client = QdrantClient(path=db_path)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection_name = "nuero_memories"
        self.lock = threading.Lock()
        self._ensure_collection()
        # Memories from before bots existed belong to the shared partition
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"bot_id": SHARED},
            key="metadata",
            points=Filter(must=[IsEmptyCondition(is_empty=PayloadField(key="metadata.bot_id"))])
        )

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def add_memory(self, text: str, metadata: dict = None):
        """metadata may carry "source" (stable file identity) and "bot_id" (owner; defaults to shared)."""
        metadata = dict(metadata or {})
        metadata.setdefault("bot_id", SHARED)
        chunks = [text[i:i+800] for i in range(0, len(text), 600)]
        if not chunks:
            return
        vectors = self.encoder.encode(chunks)
        source = metadata.get("source")
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # Stable IDs per source so re-ingesting a file overwrites instead of duplicating
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{idx}")) if source else str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={"text": chunk, "metadata": metadata}
            ))
        with self.lock:
            if source:
                # Drop the previous version first, in case the file got shorter
                self._delete_source(source)
            self.client.upsert(collection_name=self.collection_name, points=points)

    def list_sources(self) -> set:
        """Every distinct metadata.source in the vault."""
        sources, offset = set(), None
        with self.lock:
            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name, limit=500, offset=offset,
                    with_payload=True, with_vectors=False
                )
                for rec in records:
                    source = rec.payload.get("metadata", {}).get("source")
                    if source:
                        sources.add(source)
                if offset is None:
                    return sources

    def remove_source(self, source: str):
        with self.lock:
            self._delete_source(source)

    def retag_source(self, old_source: str, new_source: str):
        """Moves every chunk from one source label to another, keeping the memories themselves."""
        with self.lock:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"source": new_source},
                key="metadata",
                points=self._source_filter(old_source)
            )

    def set_source_owner(self, source: str, bot_id: str):
        """Moves a file's memories into another partition without re-reading the file."""
        with self.lock:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"bot_id": bot_id},
                key="metadata",
                points=self._source_filter(source)
            )

    def remove_bot(self, bot_id: str):
        """Deletes every memory in a bot's private partition."""
        with self.lock:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=Filter(must=[
                    FieldCondition(key="metadata.bot_id", match=MatchValue(value=bot_id))
                ]))
            )

    def remove_point(self, point_id: str):
        with self.lock:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[point_id])
            )

    @staticmethod
    def _source_filter(source: str):
        return Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))])

    def _delete_source(self, source: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=self._source_filter(source))
        )

    def search_memories(self, query_text: str, limit=3, partitions=None):
        """partitions: bot ids (and/or SHARED) to search; None searches everything."""
        query_vector = self.encoder.encode(query_text).tolist()
        query_filter = None
        if partitions is not None:
            query_filter = Filter(must=[
                FieldCondition(key="metadata.bot_id", match=MatchAny(any=list(partitions)))
            ])
        with self.lock:
            return self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            ).points

    def query_memories(self, query_text: str, limit=3, partitions=None) -> str:
        results = self.search_memories(query_text, limit, partitions)
        return "\n---\n".join([res.payload["text"] for res in results])

    def fetch_all_nodes_and_edges(self):
        """Extracts database items to calculate clusters and neural link connections."""
        with self.lock:
            records, _ = self.client.scroll(collection_name=self.collection_name, limit=1000, with_vectors=True)
        nodes = []
        edges = []

        for rec in records:
            text_snippet = rec.payload.get("text", "")[:60] + "..."
            source = rec.payload.get("metadata", {}).get("source", "Manual Input")

            # Grouping colors dynamically by metadata contents
            source_str = str(source)
            if source_str.startswith("watcher:"):
                group = 1
            elif source_str.startswith("kept:"):
                group = 4  # kept after its folder was removed
            elif "code" in source_str.lower():
                group = 2
            else:
                group = 3

            nodes.append({
                "id": rec.id,
                "label": text_snippet,
                "val": 15,
                "group": group,
                "source": source_str.split(":", 1)[1] if group in (1, 4) else "Saved from chat",
                "bot": rec.payload.get("metadata", {}).get("bot_id", SHARED),
                "full_text": rec.payload.get("text", "")
            })

        # Synapses: link each node to its 2 nearest neighbours above 0.7 cosine similarity
        if len(records) > 1:
            matrix = np.array([rec.vector for rec in records], dtype=np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
            sims = matrix @ matrix.T
            np.fill_diagonal(sims, -1)
            seen = set()
            for i in range(len(records)):
                for j in np.argsort(sims[i])[::-1][:2]:
                    if sims[i, j] <= 0.7:
                        break
                    pair = (min(i, j), max(i, j))
                    if pair not in seen:
                        seen.add(pair)
                        edges.append({"source": records[i].id, "target": records[j].id})

        return {"nodes": nodes, "links": edges}
