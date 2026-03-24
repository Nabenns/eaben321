"""Vector Store — long-term memory menggunakan ChromaDB untuk similarity search."""

import json
import logging
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

COLLECTION_EPISODES = "trade_episodes"
COLLECTION_FORMULA = "formula_params"


class VectorStore:
    def __init__(self, persist_dir: str = "./data/chromadb"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.episodes = self.client.get_or_create_collection(
            name=COLLECTION_EPISODES,
            metadata={"hnsw:space": "cosine"},
        )
        self.formula_history = self.client.get_or_create_collection(
            name=COLLECTION_FORMULA,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorStore ready | episodes: %d", self.episodes.count())

    def save_episode(self, episode: dict) -> str:
        """Simpan satu trade episode ke vector DB. Return episode ID."""
        episode_id = f"ep_{episode['timestamp'].replace(':', '').replace('-', '').replace('.', '')}"

        # Buat teks yang akan di-embed: konteks + reasoning + outcome
        embed_text = (
            f"pair={episode.get('pair')} tf={episode.get('timeframe')} "
            f"context={episode.get('market_context', '')} "
            f"reasoning={episode.get('reasoning', '')[:500]} "
            f"action={episode.get('action', '')} "
            f"result={episode.get('outcome', {}).get('result', '')}"
        )

        self.episodes.add(
            ids=[episode_id],
            documents=[embed_text],
            metadatas=[{
                "timestamp": episode.get("timestamp", ""),
                "pair": episode.get("pair", ""),
                "timeframe": episode.get("timeframe", ""),
                "action": episode.get("action", ""),
                "pnl": float(episode.get("outcome", {}).get("pnl", 0)),
                "result": episode.get("outcome", {}).get("result", ""),
                "reasoning_snippet": episode.get("reasoning", "")[:300],
                "full_data": json.dumps(episode)[:2000],
            }],
        )
        return episode_id

    def search_similar(self, query: str, n_results: int = 3) -> list[dict]:
        """Cari episode serupa berdasarkan konteks market."""
        if self.episodes.count() == 0:
            return []

        results = self.episodes.query(
            query_texts=[query],
            n_results=min(n_results, self.episodes.count()),
        )

        episodes = []
        for i, meta in enumerate(results["metadatas"][0]):
            try:
                full = json.loads(meta.get("full_data", "{}"))
            except json.JSONDecodeError:
                full = meta
            full["_similarity_score"] = 1 - results["distances"][0][i]
            episodes.append(full)

        return episodes

    def save_formula_params(self, params: dict, reason: str) -> None:
        """Simpan versi parameter formula ke history."""
        ts = datetime.now().isoformat()
        doc_id = f"params_{ts.replace(':', '').replace('-', '').replace('.', '')}"
        self.formula_history.add(
            ids=[doc_id],
            documents=[f"params={json.dumps(params)} reason={reason}"],
            metadatas=[{
                "timestamp": ts,
                "params_json": json.dumps(params),
                "reason": reason,
            }],
        )

    def get_latest_formula_params(self) -> Optional[dict]:
        """Ambil versi parameter formula terbaru."""
        count = self.formula_history.count()
        if count == 0:
            return None
        results = self.formula_history.get(limit=count)
        # Sort by timestamp descending
        items = sorted(
            zip(results["metadatas"], results["ids"]),
            key=lambda x: x[0].get("timestamp", ""),
            reverse=True,
        )
        if not items:
            return None
        latest_meta = items[0][0]
        try:
            return json.loads(latest_meta.get("params_json", "{}"))
        except json.JSONDecodeError:
            return None
