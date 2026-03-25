"""Memory Manager — koordinasi semua layer memory (vector DB + structured DB + formula)."""

import json
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
        trades_json_path: str = "./data/trades.json",
    ):
        self.formula_path = formula_path
        self.trades_json_path = Path(trades_json_path)
        self.trades_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector = VectorStore(vector_dir)
        self.db = StructuredDB(db_path)
        self._formula_cache: dict = {}
        self._params_cache: dict = {}

    # ── Formula ───────────────────────────────────────────────────────────────

    def get_formula(self, compact: bool = False) -> str:
        """Load formula dari YAML dan return sebagai string terformat.
        compact=True: strip komentar dan field verbose untuk hemat token (pakai di Groq).
        """
        if not self._formula_cache:
            with open(self.formula_path, "r", encoding="utf-8") as f:
                self._formula_cache = yaml.safe_load(f)

        if compact:
            # Strip field deskriptif panjang, pertahankan rules & checklist
            _SKIP_KEYS = {
                "description", "analogy", "indicator", "warning", "note",
                "use_case", "when_to_use", "how_to_draw", "impulse_reference",
                "confluent_signal", "after_cisd_confirmed", "after_confirmed",
                "example", "example_valid", "example_invalid", "examples",
                "how_to_mark", "identification", "fibonacci_settings",
                "continuation_setup", "invalidation", "after_cisd_confirmed",
                "entry_execution", "fresh_criteria", "quality_criteria",
                "invalid_fvg", "fvg_context", "equilibrium_context",
                "macro_cycle_timing", "continuation_setup",
            }
            def _strip(obj):
                if isinstance(obj, dict):
                    return {k: _strip(v) for k, v in obj.items() if k not in _SKIP_KEYS}
                if isinstance(obj, list):
                    return [_strip(i) for i in obj]
                return obj
            data = _strip(self._formula_cache)
            return yaml.dump(data, allow_unicode=True, default_flow_style=False)

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
        """Simpan episode ke vector DB, structured DB, dan JSON log."""
        # Vector DB untuk similarity search
        self.vector.save_episode(episode)

        # Structured DB + JSON jika ada trade action (bukan HOLD)
        decision = episode.get("decision", "")
        if "EKSEKUSI" in decision or "CLOSE" in decision:
            action = "BUY" if "BUY" in decision else ("SELL" if "SELL" in decision else "CLOSE")
            trade = {
                "timestamp": episode.get("timestamp"),
                "pair": episode.get("pair"),
                "timeframe": episode.get("timeframe"),
                "action": action,
                "reasoning": episode.get("reasoning", "")[:1000],
                "result": "pending",
            }
            self.db.save_trade(trade)
            self._append_trade_json({
                "timestamp": episode.get("timestamp"),
                "pair": episode.get("pair"),
                "timeframe": episode.get("timeframe"),
                "decision": decision.strip(),
                "result": "pending",
                "pnl": None,
            })

    def _append_trade_json(self, trade: dict, max_entries: int = 50) -> None:
        """Append trade ke JSON log, simpan max 50 entry terakhir."""
        trades = []
        if self.trades_json_path.exists():
            try:
                trades = json.loads(self.trades_json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                trades = []
        trades.append(trade)
        trades = trades[-max_entries:]
        self.trades_json_path.write_text(
            json.dumps(trades, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get_recent_trades_str(self, n: int = 5) -> str:
        """Return N trade terakhir sebagai string untuk di-inject ke system prompt."""
        if not self.trades_json_path.exists():
            return ""
        try:
            trades = json.loads(self.trades_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return ""
        if not trades:
            return ""
        recent = trades[-n:]
        lines = []
        for t in recent:
            ts = (t.get("timestamp") or "")[:16].replace("T", " ")
            result = t.get("result", "pending")
            pnl = f" | PnL: {t['pnl']}" if t.get("pnl") is not None else ""
            lines.append(f"[{ts}] {t.get('decision', '')} | Status: {result}{pnl}")
        return "\n".join(lines)

    def update_trade_json_result(self, ticket: int, pnl: float, result: str) -> None:
        """Update result + pnl entry terakhir yang masih pending di JSON log."""
        if not self.trades_json_path.exists():
            return
        try:
            trades = json.loads(self.trades_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        # Update entry pending terbaru
        for t in reversed(trades):
            if t.get("result") == "pending":
                t["result"] = result
                t["pnl"] = pnl
                break
        self.trades_json_path.write_text(
            json.dumps(trades, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update_trade_outcome(self, ticket: int, pnl: float, result: str) -> None:
        """Update hasil trade setelah posisi ditutup."""
        with self.db._conn() as conn:
            conn.execute(
                "UPDATE trades SET pnl=?, result=?, close_price=? WHERE ticket=?",
                (pnl, result, 0, ticket),
            )
        self.update_trade_json_result(ticket, pnl, result)

    def search_similar(self, context: str, n_results: int = 3) -> list[dict]:
        return self.vector.search_similar(context, n_results)

    # ── Formula Params ────────────────────────────────────────────────────────

    def store_conversation(self, username: str, user_msg: str, ai_reply: str) -> None:
        """Simpan percakapan Telegram ke vector DB."""
        self.vector.save_conversation(username, user_msg, ai_reply)

    def get_recent_conversations(self, n: int = 15) -> str:
        """Return percakapan terakhir sebagai string untuk di-inject ke context."""
        items = self.vector.get_recent_conversations(n)
        if not items:
            return ""
        lines = []
        for item in reversed(items):  # oldest first
            ts = item.get("timestamp", "")[:16].replace("T", " ")
            user = item.get("username", "trader")
            msg = item.get("user_msg", "")
            reply = item.get("ai_reply", "")[:300]
            lines.append(f"[{ts}] {user}: {msg}")
            lines.append(f"[{ts}] AI: {reply}")
        return "\n".join(lines)

    def save_formula_params(self, params: dict, reason: str) -> None:
        metrics = self.db.get_performance_summary()
        self.db.save_formula_params(params, reason, metrics)
        self.vector.save_formula_params(params, reason)
        self._params_cache = params
        logger.info("Formula params updated: %s | reason: %s", params, reason)
