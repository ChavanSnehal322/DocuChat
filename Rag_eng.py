
# rag_engine.py
import textwrap
from typing import List, Tuple

class RAGEngine:
    def __init__(self, emb_model, vector_store, llm, top_k: int = 5):
        self.emb_model = emb_model
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str) -> Tuple[str, List[str]]:
        q_vec = self.emb_model.embed_query(question)
        hits = self.vector_store.query(q_vec, top_k=self.top_k)
        if not hits:
            prompt = f"No relevant documents found. Answer concisely: {question}"
            reply = self.llm.generate(prompt)
            return reply, []
        # assemble context with indices
        assembled_parts = []
        sources = []
        for i, h in enumerate(hits):
            txt = h["text"]
            meta = h.get("meta", {})
            idx = meta.get("chunk_index", i)
            score = h.get("score", 0.0)
            snippet = txt if len(txt) < 800 else txt[:800] + "..."
            assembled_parts.append(f"[{idx}] (score:{score:.4f})\n{snippet}")
            sources.append(f"Chunk {idx}")
            
        context_block = "\n\n".join(assembled_parts)

        prompt = textwrap.dedent(f"""
            You are a helpful assistant. Use the following passages from documents to answer the question.
            If the answer is not contained, say you don't know. Cite chunk indices in your answer if applicable.

            Passages:
            {context_block}

            Question: {question}

            Answer concisely and cite chunk indices where applicable:
        """)
        reply = self.llm.generate(prompt, max_tokens=300)
        return reply, sources
