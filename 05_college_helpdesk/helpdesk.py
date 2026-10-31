# College helpdesk: answers questions by finding the best matching paragraph
# in the college documents using TF-IDF (no LLM needed).
#
# TF-IDF gives each word a weight: words that appear often in one paragraph but
# rarely in the others (like "hostel" or "fine") get a high weight.
# We turn every paragraph and the question into TF-IDF vectors and pick the
# paragraph with the highest cosine similarity.
#
# Run in the terminal with:  python helpdesk.py

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_FOLDER = Path(__file__).parent / "docs"
MIN_SCORE = 0.1  # below this we say "I don't know" instead of guessing


def load_paragraphs(folder=DOCS_FOLDER):
    """Read every .txt file and return a list of {"source", "text"} dicts."""
    paragraphs = []
    for file in sorted(folder.glob("*.txt")):
        for para in file.read_text(encoding="utf-8").split("\n\n"):
            if para.strip():
                paragraphs.append({"source": file.name, "text": para.strip()})
    return paragraphs


class Helpdesk:
    def __init__(self, folder=DOCS_FOLDER):
        self.paragraphs = load_paragraphs(folder)
        # stop_words="english" ignores common words like "the", "is", "what"
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([p["text"] for p in self.paragraphs])

    def answer(self, question):
        """Return a dict with the answer text, its source file and the match score."""
        question_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vec, self.matrix)[0]
        best = scores.argmax()

        if scores[best] < MIN_SCORE:
            return {"answer": "Sorry, I could not find that in the college documents.",
                    "source": None, "score": float(scores[best])}

        return {"answer": self.paragraphs[best]["text"],
                "source": self.paragraphs[best]["source"],
                "score": float(scores[best])}


if __name__ == "__main__":
    desk = Helpdesk()
    print(f"Loaded {len(desk.paragraphs)} paragraphs. Ask a question (or 'quit').")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("quit", "exit", ""):
            break
        result = desk.answer(q)
        print("Helpdesk:", result["answer"])
        if result["source"]:
            print(f"(source: {result['source']}, score {result['score']:.2f})")
