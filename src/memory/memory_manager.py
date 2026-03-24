"""Memory Manager — koordinasi semua layer memory (vector DB + structured DB + formula)."""

import logging
from pathlib import Path

import yaml

from src.memory.vector_store import VectorStore
from src.memory.structured_db import StructuredDB

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(
        self,
        formula_path: str = "./config/formula.yaml",
        db_path: str = "./data/trades.db",
        vector_dir: str = "./data/chromadb",
    ):
        self.formula_path = formula_path
        self.vector = VectorStore(vector_dir)
        self.db = StructuredDB(db_path)
        self._formula_cache: dict = {}
        self._params_cache: dict = {}

    # ── Formula ───────────────────────────────────────────────────────────────

    def get_formula(self) -> str:
        """Load formula dari YAML dan return sebagai string terformat."""
        if not self._formula_cache:
            with open(self.formula_path, "r", encoding="utf-8") as f:
                self._formula_cache = yaml.safe_load(f)
        return yaml.dump(self._formula_cache, allow_unicode=True, default_flow_style=False)

    def get_current_formula_params(self) -> str:
        """Ambil parameter formula terkini (dari DB atau default dari YAML)."""
        params = self.db.get_latest_formula_params()
        if not params:
            # Fallback ke default params di YAML
            formula = self._formula_cache or yaml.safe_load(
                open(self.formula_path, encoding="utf-8")
            )
            params = {
                p["name"]: p["default"]
                for p in formula.get("adaptive_params", [])
            }
        self._params_cache = params
        lines = [f"  {k}: {v}" for k, v in params.items()]
        return "adaptive_params:\n" + "\n".join(lines) if lines else "(default)"

    def get_performance_metrics(self) -> str:
        """Return ringkasan performa sebagai string untuk system prompt."""
        m = self.db.get_performance_summary(n_trades=50)
        if m["total"] == 0:
            return "Belum ada data trade."
        return (
            f"Total: {m['total']} trade | Win rate: {m['win_rate']}% | "
            f"Profit factor: {m['profit_factor']} | Total PnL: {m['total_pnl']}"
        )

    # ── Episodes ──────────────────────────────────────────────────────────────

    def save_episode(self, episode: dict) -> None:
        """Simpan episode ke vector DB dan structured DB."""
        # Vector DB untuk similarity search
        self.vector.save_episode(episode)

        # Structured DB jika ada trade action (bukan HOLD)
        decision = episode.get("decision", "")
        if "EKSEKUSI" in decision or "CLOSE" in decision:
            action = "BUY" if "BUY" in decision else ("SELL" if "SELL" in decision else "CLOSE")
            self.db.save_trade({
                "timestamp": episode.get("timestamp"),
                "pair": episode.get("pair"),
                "timeframe": episode.get("timeframe"),
                "action": action,
                "reasoning": episode.get("reasoning", "")[:1000],
                "result": "pending",
            })

    def update_trade_outcome(self, ticket: int, pnl: float, result: str) -> None:
        """Update hasil trade setelah posisi ditutup."""
        with self.db._conn() as conn:
            conn.execute(
                "UPDATE trades SET pnl=?, result=?, close_price=? WHERE ticket=?",
                (pnl, result, 0, ticket),
            )

    def search_similar(self, context: str, n_results: int = 3) -> list[dict]:
        return self.vector.search_similar(context, n_results)

    # ── Formula Params ────────────────────────────────────────────────────────

    def save_formula_params(self, params: dict, reason: str) -> None:
        metrics = self.db.get_performance_summary()
        self.db.save_formula_params(params, reason, metrics)
        self.vector.save_formula_params(params, reason)
        self._params_cache = params
        logger.info("Formula params updated: %s | reason: %s", params, reason)
