**DocuChat.git**

The project is a lightweight, production-ready RAG (Retrieval-Augmented Generation) app built with Streamlit, ChromaDB, Sentence-Transformers, and HuggingFace Router API.

DocuChat allows users to upload multiple PDF documents, automatically extract and chunk their text, build embeddings, store them in ChromaDB, 
and chat with the documents using a powerful open-source LLM (e.g., Llama-3-Instruct).

This project is compatible and designed for modern RAG pipelines.


****************************************************************************************
**Features** 

- Chat with multiple PDFs
- Uses ChromaDB (DuckDB + Parquet) vector storage
- Fast, lightweight Sentence-Transformers embeddings
- Powered by HuggingFace Router /chat/completions API
- Supports free models such as:
    meta-llama/Llama-3.2-3B-Instruct
    microsoft/Phi-3-mini-4k-instruct
    google/flan-t5-base

customizable models: Embedding model, LLM model, Chunking size / overlap, Top-k retrieval, Persistent or in-memory vector store

*************************************************************************************************************************************
**Tech Stack**

**Component      	                    Library**
Frontend UI	                        Streamlit
Text Extraction	                    pypdf
Embeddings	                        Sentence-Transformers
Vector Store	                      ChromaDB
LLM	                                HuggingFace Router /v1/chat/completions
RAG Pipeline	                      Custom Embedding + Chroma + Router
Env Config	                        python-dotenv

***********************************************************************************************************
**Installation steps:**

1] type cmds to clone repo 
    > git clone https://github.com/YourUsername/docuChat.git
    > cd docuChat

2] Creating virtual environment
    > python -m venv venv
    > source venv/bin/activate

3] Installing dependencies 
    > pip install -r requirements.txt

4] create a huggingface api token and store it in .env file
    > HF_TOKEN=your_huggingface_api_key_here

5] Run the app
    > streamlit run app.py

**********************************************************************************************************
**Working of projects:**

1. Upload PDFs: DocuChat extracts text using pypdf.

2. Chunking: Text is split into overlapping chunks for better retrieval.

3. Embedding: Chunks are converted into dense vectors:
                    sentence-transformers/all-MiniLM-L6-v2

4. Storage: Embeddings + text chunks stored in ChromaDB.

5. Retrieval: Top-k similar chunks retrieved using cosine similarity.

6. LLM Response: Prompts the HuggingFace Router API with extracted context:

                          POST https://router.huggingface.co/v1/chat/completions

7. Returns clean OpenAI-style responses.


******************************************************************************************************
**UI Features**

- Upload multiple PDFs
- Select any HuggingFace model
- Configure:
      - Chunk size
      - Chunk overlap
      - Top-k retrieval:
          - View chat history (latest messages appear on top)
          - Source chunk citations


******************************************************************************************************
**RAG pipeline**

PDFs → Extract Text → Chunk Text → Embed → Store in ChromaDB
Query → Embed → Retrieve Top-k Chunks → LLM → Answer + Sources

