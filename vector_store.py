
# CromeDB to build/ query/ save and load


from typing import List, Optional
import chromadb
from chromadb.config import Settings
import numpy as np
import uuid

class ChromaVectorStore:
    def __init__(self, emb_model, persist_directory: Optional[str] = None, collection_name: str = "docuchat"):
        self.emb_model = emb_model
        self.persist_directory = persist_directory
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory) if persist_directory else Settings()
        self.client = chromadb.Client(settings=settings)
        # ensure collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

    def build_from_texts(self, texts: List[str]):
        # use embedding engine to create vectors
        embs = self.emb_model.embed_texts(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        metadatas = [{"chunk_index": i} for i in range(len(texts))]
        # store documents & embeddings
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embs.tolist())

    def query(self, query_vec, top_k: int = 5):
        # query returns documents and distances
        res = self.collection.query(query_embeddings=[query_vec.tolist()], n_results=top_k, include=["documents", "metadatas", "distances"])
        if not res or "documents" not in res:
            return []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        results = []
        for doc, meta, dist in zip(docs, metas, dists):
            results.append({"text": doc, "meta": meta, "score": float(dist)})
        return results

    def get_all_texts(self):
        res = self.collection.get(include=["documents", "metadatas"])
        return res.get("documents", [])
