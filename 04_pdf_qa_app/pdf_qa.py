# Upload a PDF or text file and ask questions about it (RAG in a web app).
# Before running:  ollama pull nomic-embed-text   and   ollama pull llama3.2:1b
# Run with:        streamlit run pdf_qa.py

import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:1b"
CHUNK_WORDS = 150   # words per chunk
OVERLAP_WORDS = 30  # words shared between neighbouring chunks, so sentences aren't cut off
TOP_K = 3


def read_file(uploaded):
    """Return a list of (page_number, text). A .txt file counts as page 1."""
    if uploaded.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(1, uploaded.read().decode("utf-8", errors="ignore"))]


def make_chunks(pages):
    """Cut each page into overlapping pieces of CHUNK_WORDS words."""
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for page_no, text in pages:
        words = text.split()
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + CHUNK_WORDS])
            if piece:
                chunks.append({"page": page_no, "text": piece})
    return chunks


def embed(texts):
    return np.array(ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"])


def top_chunks(question, chunks, vectors):
    q = embed([question])[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] for i in best]


st.title("Chat with your Document")

uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Only rebuild the index when a NEW file is uploaded (embedding is slow)
if uploaded is not None and st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Reading and indexing the file..."):
        chunks = make_chunks(read_file(uploaded))
        st.session_state.chunks = chunks
        st.session_state.vectors = embed([c["text"] for c in chunks])
        st.session_state.file_name = uploaded.name
    st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")

if "chunks" in st.session_state:
    question = st.text_input("Your question about the document")
    if question:
        found = top_chunks(question, st.session_state.chunks, st.session_state.vectors)
        context = "\n\n".join(f"(page {c['page']}) {c['text']}" for c in found)
        prompt = (
            "Use only the context to answer. If the answer is not in it, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        with st.spinner("Thinking..."):
            reply = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
        st.write(reply["message"]["content"])

        with st.expander("Show the parts of the document that were used"):
            for c in found:
                st.markdown(f"**Page {c['page']}:** {c['text']}")
else:
    st.info("Upload a file to get started.")
