

import streamlit as st
from dotenv import load_dotenv
import os
from webpg import css, bot_template, user_template
from utils import extract_text_from_pdfs, chunk_text
from embeddings import EmbeddingEngine
from vector_store import ChromaVectorStore
from LLM_eng import HuggingFaceLLM
from Rag_eng  import RAGEngine

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="DocuChat 2.0", page_icon="📚", layout="wide")
st.write(css, unsafe_allow_html=True)
st.title(" DocuChat — Chroma + HF Inference")

# Sidebar controls
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs (multiple)", accept_multiple_files=True, type=["pdf"])
    st.markdown("---")
    st.header("Settings")
    embed_model = st.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2"], index=0)

    #  hf_model = st.text_input("HuggingFace model", value="mistralai/Mistral-7B-Instruct-v0.2")
    hf_model = st.text_input(
    "HuggingFace model",
    value="meta-llama/Llama-3.2-3B-Instruct"   # <— FREE + supported
    )


    chunk_size = st.number_input("Chunk size (chars)", value=800, step=100)

    chunk_overlap = st.number_input("Chunk overlap (chars)", value=200, step=50)
    
    #  find top k chunks
    top_k = st.slider("Top-k retrieval", 1, 10, 5)
    
    persist_dir = st.text_input("Chroma persist dir (empty => in-memory)", value="")
    
    process_btn = st.button("Process documents")

# Session state
if "texts" not in st.session_state:
    st.session_state.texts = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "emb" not in st.session_state:
    st.session_state.emb = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Process
if process_btn:
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF.")
    elif not HF_TOKEN:
        st.sidebar.error("Set HF_TOKEN in .env first.")
    else:
        with st.spinner("Extracting text and building index..."):
            text = extract_text_from_pdfs(uploaded_files)
            chunks = chunk_text(text, chunk_size=int(chunk_size), overlap=int(chunk_overlap))
            if not chunks:
                st.error("No text found in PDFs.")
            else:
                emb = EmbeddingEngine(model_name=embed_model)
                vs = ChromaVectorStore(emb_model=emb, persist_directory=persist_dir or None)
                vs.build_from_texts(chunks)

                
                llm = HuggingFaceLLM(hf_token=HF_TOKEN, model_name=hf_model)
                rag = RAGEngine(emb_model=emb, vector_store=vs, llm=llm, top_k=int(top_k))
                # store
                st.session_state.texts = chunks
                st.session_state.vector_store = vs
                st.session_state.emb = emb
                st.session_state.llm = llm
                st.session_state.rag = rag
                st.success(f"Indexed {len(chunks)} chunks.")

# Querying the UI
query = st.text_input("Ask a question about the uploaded documents:")
if st.button("Ask") or (query and st.session_state.get("auto_ask_on_enter", True) and query):
    if st.session_state.rag is None:
        st.error("Please process documents first.")
    else:
        q = query.strip()
        if not q:
            st.warning("Enter a question.")
        else:
            st.session_state.chat_history.append({"role": "user", "text": q})
            with st.spinner("Retrieving and generating..."):
                answer, sources = st.session_state.rag.answer(q)
            st.session_state.chat_history.append({"role": "ai", "text": answer, "evidence": sources})

# Rendering chat
for m in reversed(st.session_state.chat_history):
    if m["role"] == "user":
        st.write(user_template.replace("{{MSG}}", m["text"]), unsafe_allow_html=True)
    else:
        sources_html = "<br>".join(m.get("evidence", []))
        st.write(
            bot_template.replace("{{MSG}}", m["text"] + "<br><br><b>Sources:</b><br>" + sources_html),
            unsafe_allow_html=True
        )
        
