"""Structured DB — SQLite untuk trade logs, metrics, dan formula param history."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class StructuredDB:
    def __init__(self, db_path: str = "./data/trades.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    pair        TEXT NOT NULL,
                    timeframe   TEXT,
                    action      TEXT,
                    lot         REAL,
                    open_price  REAL,
                    close_price REAL,
                    sl          REAL,
                    tp          REAL,
                    pnl         REAL,
                    result      TEXT,
                    ticket      INTEGER,
                    reasoning   TEXT,
                    comment     TEXT
                );

                CREATE TABLE IF NOT EXISTS formula_params (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    reason      TEXT,
                    win_rate    REAL,
                    profit_factor REAL
                );

                CREATE TABLE IF NOT EXISTS daily_metrics (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    date            TEXT NOT NULL UNIQUE,
                    total_trades    INTEGER DEFAULT 0,
                    wins            INTEGER DEFAULT 0,
                    losses          INTEGER DEFAULT 0,
                    total_pnl       REAL DEFAULT 0,
                    max_drawdown    REAL DEFAULT 0,
                    profit_factor   REAL DEFAULT 0
                );
            """)
        logger.info("StructuredDB initialized: %s", self.db_path)

    # ── Trades ────────────────────────────────────────────────────────────────

    def save_trade(self, trade: dict) -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                """INSERT INTO trades
                   (timestamp, pair, timeframe, action, lot, open_price, close_price,
                    sl, tp, pnl, result, ticket, reasoning, comment)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade.get("timestamp", datetime.now().isoformat()),
                    trade.get("pair", ""),
                    trade.get("timeframe", ""),
                    trade.get("action", ""),
                    trade.get("lot", 0),
                    trade.get("open_price", 0),
                    trade.get("close_price", 0),
                    trade.get("sl", 0),
                    trade.get("tp", 0),
                    trade.get("pnl", 0),
                    trade.get("result", ""),
                    trade.get("ticket"),
                    trade.get("reasoning", "")[:1000],
                    trade.get("comment", ""),
                ),
            )
            return cursor.lastrowid

    def get_recent_trades(self, n: int = 20, pair: str = None) -> list[dict]:
        with self._conn() as conn:
            if pair:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE pair=? ORDER BY timestamp DESC LIMIT ?",
                    (pair, n),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (n,)
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Formula Params ────────────────────────────────────────────────────────

    def save_formula_params(self, params: dict, reason: str, metrics: dict = None) -> None:
        metrics = metrics or {}
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO formula_params (timestamp, params_json, reason, win_rate, profit_factor)
                   VALUES (?,?,?,?,?)""",
                (
                    datetime.now().isoformat(),
                    json.dumps(params),
                    reason,
                    metrics.get("win_rate", 0),
                    metrics.get("profit_factor", 0),
                ),
            )

    def get_latest_formula_params(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT params_json FROM formula_params ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row:
                return json.loads(row["params_json"])
            return {}

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_performance_summary(self, n_trades: int = 50) -> dict:
        """Hitung win rate, profit factor, total PnL dari N trade terakhir."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pnl, result FROM trades ORDER BY timestamp DESC LIMIT ?", (n_trades,)
            ).fetchall()

        if not rows:
            return {"total": 0, "win_rate": 0, "profit_factor": 0, "total_pnl": 0}

        total = len(rows)
        wins = sum(1 for r in rows if r["result"] == "win")
        gross_profit = sum(r["pnl"] for r in rows if r["pnl"] > 0)
        gross_loss = abs(sum(r["pnl"] for r in rows if r["pnl"] < 0))
        total_pnl = sum(r["pnl"] for r in rows)

        return {
            "total": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total * 100, 1) if total else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else 0,
            "total_pnl": round(total_pnl, 2),
        }
