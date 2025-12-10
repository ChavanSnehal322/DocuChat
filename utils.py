
# utils.py
from pypdf import PdfReader
from typing import List

def extract_text_from_pdfs(uploaded_files) -> str:
    texts = []
    for f in uploaded_files:
        try:
            reader = PdfReader(f)
            pages = []
            for p in reader.pages:
                txt = p.extract_text()
                if txt:
                    pages.append(txt)
            if pages:
                texts.append("\n".join(pages))
        except Exception as e:
            print(f"Failed to read {getattr(f, 'name', 'file')}: {e}")
    return "\n\n".join(t for t in texts if t)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, end - overlap)
    return [c for c in chunks if c]
