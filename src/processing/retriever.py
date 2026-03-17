"""
retriever.py — Hybrid BM25 + Dense Embedding retriever with RRF merging and Cross-Encoder re-ranking.

Pipeline per query:
  1. BM25 sparse retrieval  → scored ranks
  2. Dense cosine-similarity → scored ranks
  3. Reciprocal Rank Fusion (RRF) to merge both rank lists → top_k candidates
  4. Cross-Encoder re-ranker on top_k candidates → best chunk + confidence
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from processing.chunker import Chunk
from utils import get_logger

log = get_logger(__name__)

# ── Default model names ────────────────────────────────────────────────────────

_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_RRF_K = 60  # Standard RRF constant


# ── Internal helpers ──────────────────────────────────────────────────────────


def _argsort_desc(scores: np.ndarray) -> np.ndarray:
    """Return indices sorted by score descending."""
    return np.argsort(scores)[::-1]


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load (and optionally cache) a SentenceTransformer embedding model."""
    log.info("📦 Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    log.info("📦 Embedding model ready.")
    return model


def load_reranker(model_name: str) -> CrossEncoder:
    """Load (and optionally cache) a CrossEncoder re-ranker model."""
    log.info("📦 Loading re-ranker model: %s", model_name)
    model = CrossEncoder(model_name, token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    log.info("📦 Re-ranker model ready.")
    return model


class ChunkRetriever:
    """
    Hybrid retriever that combines BM25 and dense embeddings via RRF,
    then re-ranks the top candidates with a CrossEncoder.

    Parameters
    ----------
    chunks : List[Chunk]
        The document chunks to search over.
    embedding_model : SentenceTransformer
        Pre-loaded embedding model (shared across calls for performance).
    reranker : CrossEncoder
        Pre-loaded cross-encoder re-ranker (shared across calls).
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
        reranker_model_name: str = _DEFAULT_RERANKER_MODEL,
    ) -> None:
        self.chunks = chunks
        self.embedding_model = load_embedding_model(embedding_model_name)
        self.reranker = load_reranker(reranker_model_name)

        texts = [c.text for c in chunks]
        tokenized = [t.lower().split() for t in texts]

        # ── BM25 index ────────────────────────────────────────────────────────
        self.bm25 = BM25Okapi(tokenized)

        # ── Dense embeddings index ────────────────────────────────────────────
        log.info("🔍 Encoding %d chunks for dense retrieval...", len(chunks))
        self.embeddings: np.ndarray = self.embedding_model.encode(
            sentences=texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        log.info("🔍 Chunk embeddings ready.")

    # ─── Public API ───────────────────────────────────────────────────────────

    def find_best_chunk(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[Chunk | None, float]:
        """
        Find the single best chunk for a query using:
          BM25 + cosine similarity → RRF → re-rank with CrossEncoder.

        Parameters
        ----------
        query : str
            Search query (field name + raw value gives best results).
        top_k : int
            Number of candidates to consider after RRF before re-ranking.

        Returns
        -------
        (best_chunk, confidence_0_to_1)
        """
        if not query or not self.chunks:
            return None, 0.0

        n = len(self.chunks)

        # ── Step 1 – BM25 ranks ───────────────────────────────────────────────
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_ranks = _argsort_desc(bm25_scores)

        # ── Step 2 – Dense cosine-similarity ranks ────────────────────────────
        query_emb = self.embedding_model.encode(
            query, normalize_embeddings=True, show_progress_bar=False
        )
        cos_scores = self.embeddings @ query_emb  # dot product = cosine (normalized)
        dense_ranks = _argsort_desc(cos_scores)

        # ── Step 3 – Reciprocal Rank Fusion ───────────────────────────────────
        rrf_scores = np.zeros(n, dtype=float)
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] += 1.0 / (_RRF_K + rank + 1)
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] += 1.0 / (_RRF_K + rank + 1)

        top_k_indices = _argsort_desc(rrf_scores)[:top_k]
        candidates = [self.chunks[i] for i in top_k_indices]

        # ── Step 4 – Cross-Encoder re-ranking ─────────────────────────────────
        pairs = [(query, c.text) for c in candidates]
        logits = self.reranker.predict(pairs, show_progress_bar=False)

        # Convert CrossEncoder logits to [0, 1] using softmax over candidates.
        # This provides a confidence score that accounts for how much better
        # the top chunk is compared to other candidates.
        exp_logits = np.exp(
            logits - np.max(logits)
        )  # Subtract max for numerical stability
        probs = exp_logits / np.sum(exp_logits)

        best_local_idx = int(np.argmax(logits))
        best_chunk = candidates[best_local_idx]
        confidence = float(probs[best_local_idx])

        log.debug(
            "🔍 Re-ranker picked chunk %s (logit=%.3f, confidence=%.1f%%)",
            best_chunk.id,
            float(logits[best_local_idx]),
            confidence * 100,
        )

        return best_chunk, confidence
