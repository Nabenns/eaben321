"""Position Monitor — deteksi posisi yang tertutup dan record PnL ke DB."""

import logging
from datetime import datetime, timezone, timedelta

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class PositionMonitor:
    def __init__(self, memory, notifier=None, magic: int = 0):
        self.memory = memory
        self.notifier = notifier
        self.magic = magic
        # ticket -> snapshot posisi yang sedang open
        self._open_positions: dict[int, dict] = {}

    def sync(self) -> list[dict]:
        """
        Bandingkan snapshot open positions dengan kondisi MT5 sekarang.
        Posisi yang hilang dari MT5 = sudah closed → record outcome.
        Return list posisi yang baru ditutup.
        """
        # Ambil posisi open saat ini dari MT5
        raw = mt5.positions_get()
        current_tickets = set()

        if raw:
            for pos in raw:
                if self.magic and pos.magic != self.magic:
                    continue
                ticket = pos.ticket
                current_tickets.add(ticket)

                # Tambahkan ke snapshot jika belum ada
                if ticket not in self._open_positions:
                    self._open_positions[ticket] = {
                        "ticket": ticket,
                        "pair": pos.symbol,
                        "type": "BUY" if pos.type == 0 else "SELL",
                        "volume": pos.volume,
                        "open_price": pos.price_open,
                        "sl": pos.sl,
                        "tp": pos.tp,
                        "open_time": datetime.fromtimestamp(pos.time, tz=timezone.utc).isoformat(),
                    }
                    logger.info("[Monitor] New open position tracked: %s %s @ %.5f",
                                pos.symbol, "BUY" if pos.type == 0 else "SELL", pos.price_open)

        # Posisi yang ada di snapshot tapi sudah hilang dari MT5 = closed
        closed_tickets = set(self._open_positions.keys()) - current_tickets
        newly_closed = []

        for ticket in closed_tickets:
            snap = self._open_positions.pop(ticket)
            outcome = self._fetch_closed_outcome(ticket, snap)
            if outcome:
                self._record_outcome(outcome)
                newly_closed.append(outcome)

        return newly_closed

    def _fetch_closed_outcome(self, ticket: int, snap: dict) -> dict | None:
        """Ambil detail trade history dari MT5 untuk ticket yang baru ditutup."""
        # Cari di history 7 hari terakhir
        date_from = datetime.now(timezone.utc) - timedelta(days=7)
        deals = mt5.history_deals_get(date_from, datetime.now(timezone.utc))

        if deals is None:
            return None

        # Cari deal close yang terkait ticket ini
        close_deal = None
        for deal in deals:
            if deal.position_id == ticket and deal.entry == 1:  # entry=1 = close deal
                close_deal = deal
                break

        if not close_deal:
            # Fallback: pakai deal manapun dengan position_id ini
            for deal in deals:
                if deal.position_id == ticket:
                    close_deal = deal
                    break

        if not close_deal:
            logger.warning("[Monitor] Cannot find close deal for ticket %d", ticket)
            return None

        pnl = close_deal.profit
        close_price = close_deal.price
        result = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")

        return {
            "ticket": ticket,
            "pair": snap["pair"],
            "type": snap["type"],
            "volume": snap["volume"],
            "open_price": snap["open_price"],
            "close_price": close_price,
            "sl": snap["sl"],
            "tp": snap["tp"],
            "pnl": pnl,
            "result": result,
            "open_time": snap["open_time"],
            "close_time": datetime.fromtimestamp(close_deal.time, tz=timezone.utc).isoformat(),
        }

    def _record_outcome(self, outcome: dict) -> None:
        """Simpan hasil trade ke DB dan kirim notifikasi."""
        ticket = outcome["ticket"]
        pnl = outcome["pnl"]
        result = outcome["result"]

        # Update DB
        self.memory.update_trade_outcome(ticket, pnl, result)

        emoji = "✅" if result == "win" else ("❌" if result == "loss" else "➖")
        logger.info(
            "[Monitor] Trade closed: ticket=%d | %s %s | PnL=%.2f | %s",
            ticket, outcome["pair"], outcome["type"], pnl, result.upper()
        )

        # Telegram notifikasi
        if self.notifier:
            msg = (
                f"{emoji} <b>Trade Closed</b>\n"
                f"Ticket: {ticket} | {outcome['pair']} {outcome['type']}\n"
                f"Open: {outcome['open_price']:.5f} → Close: {outcome['close_price']:.5f}\n"
                f"PnL: <b>{pnl:+.2f}</b> | Result: {result.upper()}"
            )
            self.notifier.send(msg)
