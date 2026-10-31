# RAG (Retrieval-Augmented Generation) over my C++ notes, in the terminal.
#
# Steps:
#   1. Read cpp_notes.txt and split it into paragraphs (chunks)
#   2. Turn every chunk into a vector of numbers (an "embedding")
#   3. For a question, embed it too and find the most similar chunks
#   4. Give those chunks + the question to the LLM and print its answer
#
# Before running:  ollama pull nomic-embed-text   and   ollama pull llama3.2:1b
# Run with:        python cpp_rag.py

from pathlib import Path

import numpy as np
import ollama

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:1b"
TOP_K = 3  # how many chunks to give the LLM

NOTES_FILE = Path(__file__).parent / "cpp_notes.txt"


def load_chunks(path):
    """Split the file on blank lines, so each paragraph is one chunk."""
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def embed(texts):
    """Return a 2D numpy array: one row (vector) per text."""
    result = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(result["embeddings"])


def cosine_similarity(question_vec, chunk_vecs):
    """How close the question is to each chunk: 1 = same direction, 0 = unrelated."""
    dot = chunk_vecs @ question_vec
    norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(question_vec)
    return dot / norms


def search(question, chunks, chunk_vecs, k=TOP_K):
    question_vec = embed([question])[0]
    scores = cosine_similarity(question_vec, chunk_vecs)
    best = np.argsort(scores)[::-1][:k]  # indexes of the k highest scores
    return [(chunks[i], scores[i]) for i in best]


def ask_llm(question, found_chunks):
    context = "\n\n".join(chunk for chunk, _ in found_chunks)
    prompt = (
        "Answer the question using only the notes below. "
        "If the notes do not contain the answer, say you don't know.\n\n"
        f"Notes:\n{context}\n\nQuestion: {question}"
    )
    response = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def main():
    chunks = load_chunks(NOTES_FILE)
    print(f"Loaded {len(chunks)} chunks. Creating embeddings...")
    chunk_vecs = embed(chunks)
    print("Ready! Type a C++ question, or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", ""):
            break

        found = search(question, chunks, chunk_vecs)
        print("\nMost relevant notes:")
        for chunk, score in found:
            print(f"  [{score:.2f}] {chunk[:80]}...")

        print("\nBot:", ask_llm(question, found), "\n")


if __name__ == "__main__":
    main()
