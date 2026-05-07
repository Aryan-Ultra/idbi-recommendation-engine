"""
embedding_model.py — Semantic embedding model for gift voucher items.
Uses TF-IDF vectorization + cosine similarity (offline substitute for sentence-transformers).
Architecturally identical: produces (n_items x n_dims) L2-normalised embedding matrix.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.config import EMBEDDING_MODEL_NAME, EMBEDDINGS_PKL

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """TF-IDF based semantic embeddings (drop-in for sentence-transformers)."""

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.embeddings: np.ndarray | None = None
        self.voucher_ids: list = []
        self._model_tag = f"TF-IDF (offline substitute for {EMBEDDING_MODEL_NAME})"

    def build_corpus_text(self, vouchers_df: pd.DataFrame) -> list:
        texts = []
        for _, row in vouchers_df.iterrows():
            text = (
                f"{row['name']} {row['category']} {row['brand']} {row['description']}"
            ).lower()
            texts.append(text)
        return texts

    def fit(self, vouchers_df: pd.DataFrame):
        """Encode all vouchers using TF-IDF and store L2-normalised embeddings."""
        logger.info("Fitting embedding model: %s", self._model_tag)
        self.voucher_ids = vouchers_df["voucher_id"].tolist()
        texts = self.build_corpus_text(vouchers_df)

        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=1024, sublinear_tf=True
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        self.embeddings = normalize(tfidf_matrix, norm="l2")
        logger.info("Embeddings shape: %s", self.embeddings.shape)

    def score_items_for_user(
        self, user_interacted_item_indices: list, candidate_item_indices: list
    ) -> dict:
        if not user_interacted_item_indices or self.embeddings is None:
            return {idx: 0.0 for idx in candidate_item_indices}
        user_profile = normalize(
            self.embeddings[user_interacted_item_indices].mean(axis=0, keepdims=True), norm="l2"
        )
        candidate_embeddings = self.embeddings[candidate_item_indices]
        scores = cosine_similarity(user_profile, candidate_embeddings)[0]
        return {idx: float(score) for idx, score in zip(candidate_item_indices, scores)}

    def get_similar_items(self, item_indices: list, n: int = 10) -> dict:
        if self.embeddings is None:
            raise RuntimeError("Model not fitted.")
        query = self.embeddings[item_indices]
        sim_matrix = cosine_similarity(query, self.embeddings)
        results = {}
        for i, item_idx in enumerate(item_indices):
            sims = sim_matrix[i].copy()
            sims[item_idx] = -1.0
            top_indices = np.argsort(sims)[::-1][:n]
            results[item_idx] = [(int(idx), float(sims[idx])) for idx in top_indices]
        return results

    def save(self):
        payload = {
            "embeddings": self.embeddings,
            "voucher_ids": self.voucher_ids,
            "vectorizer": self.vectorizer,
            "model_name": self._model_tag,
        }
        with open(EMBEDDINGS_PKL, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Embeddings saved to %s", EMBEDDINGS_PKL)

    @classmethod
    def load(cls) -> "EmbeddingModel":
        em = cls()
        with open(EMBEDDINGS_PKL, "rb") as f:
            payload = pickle.load(f)
        em.embeddings = payload["embeddings"]
        em.voucher_ids = payload["voucher_ids"]
        em.vectorizer = payload.get("vectorizer")
        logger.info("Embeddings loaded: shape=%s", em.embeddings.shape)
        return em
