import os
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    CHUNK_MAX_WORDS = 200
    TOP_K = 3
    SIMILARITY_THRESHOLD = 0.1

    def __init__(self):
        self.chunks: list[dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.is_indexed = False

    def index_documents(self, docs_path: str) -> dict:
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Documents folder not found: {docs_path}")

        txt_files = sorted(docs_dir.glob("*.txt"))
        if not txt_files:
            raise ValueError("No .txt files found in the documents folder")

        self.chunks = []
        for filepath in txt_files:
            text = filepath.read_text(encoding="utf-8")
            file_chunks = self._split_into_chunks(text)
            for chunk_text in file_chunks:
                self.chunks.append({
                    "source": filepath.name,
                    "text": chunk_text,
                })

        chunk_texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
        self.is_indexed = True

        return {
            "documents": len(txt_files),
            "chunks": len(self.chunks),
        }

    def ask(self, question: str) -> dict:
        if not self.is_indexed:
            raise RuntimeError("No documents indexed yet. Call /index first.")

        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, self.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[::-1][:self.TOP_K]
        top_scores = similarities[top_indices]

        if top_scores[0] < self.SIMILARITY_THRESHOLD:
            return {
                "answer": "I could not find sufficient information in the documents to answer this question.",
                "sources": [],
                "confidence": "insufficient_context",
            }

        sources = []
        for idx, score in zip(top_indices, top_scores):
            if score >= self.SIMILARITY_THRESHOLD:
                sources.append({
                    "source": self.chunks[idx]["source"],
                    "text": self.chunks[idx]["text"],
                    "score": round(float(score), 4),
                })

        answer = self._extract_answer(question, sources)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": "answered_from_docs",
        }

    def _split_into_chunks(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        merged = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if merged and len(merged[-1].split()) < 15:
                merged[-1] = merged[-1] + "\n\n" + para
            else:
                merged.append(para)

        chunks = []
        for para in merged:
            words = para.split()
            if len(words) <= self.CHUNK_MAX_WORDS:
                chunks.append(para)
            else:
                for i in range(0, len(words), self.CHUNK_MAX_WORDS):
                    chunk = " ".join(words[i : i + self.CHUNK_MAX_WORDS])
                    chunks.append(chunk)
        return chunks

    def _extract_answer(self, question: str, sources: list[dict]) -> str:
        question_words = set(question.lower().split())
        stop_words = {"what", "is", "the", "a", "an", "how", "do", "does", "can",
                       "i", "my", "me", "to", "and", "or", "of", "in", "for", "on",
                       "are", "there", "this", "that", "it", "be", "was", "were",
                       "with", "from", "about", "which", "who", "when", "where", "why"}
        question_keywords = question_words - stop_words

        all_sentences = []
        for source in sources:
            lines = re.split(r"\n+", source["text"])
            for line in lines:
                line = line.strip()
                if len(line) < 10:
                    continue
                line_words = set(line.lower().split())
                overlap = len(question_keywords & line_words)
                combined_score = overlap + source["score"] * 3
                all_sentences.append((line, combined_score))

        all_sentences.sort(key=lambda x: x[1], reverse=True)

        selected = []
        total_words = 0
        for line, _score in all_sentences:
            if total_words > 150:
                break
            if line not in selected:
                selected.append(line)
                total_words += len(line.split())

        if not selected:
            return sources[0]["text"]

        return "\n".join(selected)
