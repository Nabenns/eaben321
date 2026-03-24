"""Tool Handler — eksekusi tool calls dari LLM dan return hasilnya."""

import logging
from typing import Any

from src.mt5.connector import MT5Connector
from src.mt5.data_formatter import (
    candles_to_toon,
    tick_to_toon,
    positions_to_toon,
    account_to_toon,
    trade_history_to_toon,
)
from src.mt5.executor import MT5Executor
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class ToolHandler:
    def __init__(
        self,
        connector: MT5Connector,
        executor: MT5Executor,
        memory: MemoryManager,
    ):
        self.connector = connector
        self.executor = executor
        self.memory = memory

    def handle(self, tool_name: str, tool_input: dict) -> Any:
        """Dispatch tool call dari LLM ke handler yang sesuai."""
        logger.info("Tool call: %s | input: %s", tool_name, tool_input)

        handlers = {
            "get_chart": self._get_chart,
            "get_tick": self._get_tick,
            "get_open_positions": self._get_open_positions,
            "get_account_info": self._get_account_info,
            "get_trade_history": self._get_trade_history,
            "query_memory": self._query_memory,
            "execute_trade": self._execute_trade,
            "close_position": self._close_position,
            "update_formula_params": self._update_formula_params,
            "get_session_info": self._get_session_info,
            "get_current_quarter": self._get_current_quarter,
            "get_ndog_nwog": self._get_ndog_nwog,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return f"ERROR: Tool '{tool_name}' tidak dikenal."

        try:
            return handler(**tool_input)
        except Exception as e:
            logger.exception("Tool %s error: %s", tool_name, e)
            return f"ERROR saat eksekusi {tool_name}: {e}"

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _get_chart(self, pair: str, timeframe: str, n_candles: int = 100) -> str:
        n_candles = min(n_candles, 500)
        df = self.connector.get_chart(pair, timeframe, n_candles)
        return candles_to_toon(pair, timeframe, df)

    def _get_tick(self, pair: str) -> str:
        tick = self.connector.get_tick(pair)
        return tick_to_toon(tick)

    def _get_open_positions(self, pair: str = None) -> str:
        positions = self.connector.get_open_positions(pair)
        return positions_to_toon(positions)

    def _get_account_info(self) -> str:
        account = self.connector.get_account_info()
        return account_to_toon(account)

    def _get_trade_history(self, n: int = 20, pair: str = None) -> str:
        history = self.connector.get_trade_history(n, pair)
        return trade_history_to_toon(history)

    def _query_memory(self, context: str, n_results: int = 3) -> str:
        results = self.memory.search_similar(context, n_results)
        if not results:
            return "Tidak ada situasi serupa ditemukan di memori."
        lines = ["Situasi serupa dari memori:"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"\n[{i}] {r.get('timestamp', '')} | {r.get('pair', '')} {r.get('timeframe', '')}\n"
                f"    Konteks: {r.get('market_context', '')}\n"
                f"    Aksi: {r.get('action', '')} | Hasil: {r.get('outcome', {}).get('result', '')} "
                f"PnL: {r.get('outcome', {}).get('pnl', 0)}\n"
                f"    Reasoning: {r.get('reasoning', '')}"
            )
        return "\n".join(lines)

    def _execute_trade(
        self,
        pair: str,
        action: str,
        lot: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "ai-trade",
    ) -> str:
        result = self.executor.execute_trade(pair, action, lot, sl, tp, comment)
        if result["success"]:
            return (
                f"Order berhasil. Ticket: {result['ticket']} | "
                f"{action} {pair} {lot} lot @ {result['price']} | SL: {sl} TP: {tp}"
            )
        return f"Order GAGAL: {result['error']}"

    def _close_position(self, ticket: int) -> str:
        result = self.executor.close_position(ticket)
        if result["success"]:
            return f"Posisi {ticket} berhasil ditutup @ {result['close_price']}"
        return f"Close GAGAL: {result['error']}"

    def _update_formula_params(self, params: dict, reason: str) -> str:
        self.memory.save_formula_params(params, reason)
        return f"Parameter formula diupdate: {params} | Alasan: {reason}"

    def _get_session_info(self) -> str:
        import json
        info = self.connector.get_session_info()
        lines = [
            f"Waktu UTC: {info['current_time_utc']}",
            f"Sesi aktif: {', '.join(info['active_sessions'])}",
            f"Asia bias: {info['asia_bias']} → Pattern: {info['amd_pattern']}",
            f"90m Quarter: {info['90min_quarter']['quarter']} "
            f"(pos: {info['90min_quarter']['position_in_cycle_min']:.1f}m dalam cycle "
            f"{info['90min_quarter']['cycle_start']}–{info['90min_quarter']['cycle_end']})",
            f"High-probability window: {info['is_high_probability_window']}",
            f"Entry allowed: {info['entry_allowed_by_session']}",
            f"Note: {info['note']}",
        ]
        return "\n".join(lines)

    def _get_current_quarter(self, cycle: str = "macro_90min") -> str:
        result = self.connector.get_current_quarter(cycle)
        if "error" in result:
            return f"ERROR: {result['error']}"
        if cycle == "macro_90min":
            return (
                f"Cycle: {cycle} | Quarter: {result['quarter']} | "
                f"Posisi dalam cycle: {result['position_in_cycle_min']:.1f} menit | "
                f"Cycle: {result['cycle_start']} – {result['cycle_end']}"
            )
        if cycle == "daily":
            return (
                f"Cycle: daily | Quarter: {result['quarter']} | "
                f"Jam UTC saat ini: {result['current_hour_utc']}:xx"
            )
        if cycle == "weekly":
            return f"Cycle: weekly | Quarter: {result['quarter']} | Hari: {result['weekday']}"
        return str(result)

    def _get_ndog_nwog(self, pair: str) -> str:
        data = self.connector.get_ndog_nwog(pair)
        ndog = data["ndog"]
        nwog = data["nwog"]

        lines = [f"NDOG/NWOG untuk {pair}:"]

        if ndog["exists"]:
            lines.append(
                f"NDOG: {ndog['direction']} | "
                f"Gap {ndog['yesterday_close']} → {ndog['today_open']} "
                f"(size: {ndog['gap_size']:+.5f}) | "
                f"SR zone: {ndog['sr_zone']['lower']}–{ndog['sr_zone']['upper']} | "
                f"Midpoint: {ndog['midpoint']}"
            )
        else:
            lines.append("NDOG: tidak ada gap (open == yesterday close)")

        if nwog.get("exists"):
            lines.append(
                f"NWOG: {nwog['direction']} | "
                f"Gap {nwog['friday_close']} → {nwog['monday_open']} "
                f"(size: {nwog['gap_size']:+.5f}) | "
                f"SR zone: {nwog['sr_zone']['lower']}–{nwog['sr_zone']['upper']} | "
                f"Midpoint: {nwog['midpoint']}"
            )
        else:
            lines.append(f"NWOG: tidak ada gap | {nwog.get('note', '')}")

        lines.append(data["usage_note"])
        return "\n".join(lines)
