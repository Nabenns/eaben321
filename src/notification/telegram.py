"""Telegram Notifier — kirim alert keputusan trading ke grup Telegram."""

import logging
import urllib.request
import urllib.parse
import json

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send(self, text: str) -> bool:
        """Kirim pesan ke grup. Return True jika berhasil."""
        payload = json.dumps({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")

        req = urllib.request.Request(
            self._base_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return False

    def notify_decision(self, pair: str, timeframe: str, decision: str, tool_calls: list) -> None:
        """Format dan kirim notifikasi keputusan LLM."""
        # Ekstrak baris terakhir yang berisi EKSEKUSI/HOLD/CLOSE
        lines = [l.strip() for l in decision.strip().splitlines() if l.strip()]
        verdict = next(
            (l for l in reversed(lines) if any(k in l for k in ("EKSEKUSI:", "HOLD:", "CLOSE:"))),
            lines[-1] if lines else "—"
        )

        emoji = "🟢" if "EKSEKUSI" in verdict and "BUY" in verdict else \
                "🔴" if "EKSEKUSI" in verdict and "SELL" in verdict else \
                "⚫" if "CLOSE" in verdict else "⏸️"

        tools_used = ", ".join(set(tc["tool"] for tc in tool_calls)) if tool_calls else "—"

        msg = (
            f"{emoji} <b>AI Trading Signal</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Pair:</b> {pair} | <b>TF:</b> {timeframe}\n"
            f"<b>Tools used:</b> {tools_used}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Keputusan:</b>\n{verdict}"
        )
        self.send(msg)

    def notify_error(self, error: str) -> None:
        """Kirim notifikasi error sistem."""
        self.send(f"⚠️ <b>System Error</b>\n{error[:300]}")

    def notify_startup(self, pair: str, mode: str) -> None:
        """Kirim notifikasi sistem start."""
        self.send(
            f"🚀 <b>AI Trading System Started</b>\n"
            f"Pair: {pair} | Mode: {mode}"
        )
