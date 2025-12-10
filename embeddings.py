
# embeddings loading and encoding

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32):
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs

    def embed_query(self, query: str):
        v = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        return v[0]
