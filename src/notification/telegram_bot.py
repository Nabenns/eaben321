"""Telegram ChatBot — terima pesan dari grup dan balas via LLM."""

import json
import logging
import ssl
import threading
import urllib.request
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramChatBot:
    """
    Polling bot yang menerima pesan dari grup Telegram dan meneruskan ke LLM.
    Jalankan di background thread via start().
    """

    def __init__(self, bot_token: str, chat_id: str, engine, connector, notifier):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.engine = engine
        self.connector = connector
        self.notifier = notifier
        self._base = f"https://api.telegram.org/bot{bot_token}"
        self._offset = 0
        self._running = False
        self._ssl_ctx = ssl.create_default_context()
        self._ssl_ctx.check_hostname = False
        self._ssl_ctx.verify_mode = ssl.CERT_NONE

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, method: str, params: dict = None) -> dict:
        url = f"{self._base}/{method}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=35, context=self._ssl_ctx) as r:
                return json.loads(r.read())
        except Exception as e:
            logger.debug("Telegram GET %s error: %s", method, e)
            return {}

    def _post(self, method: str, payload: dict) -> dict:
        import urllib.parse
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base}/{method}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30, context=self._ssl_ctx) as r:
                return json.loads(r.read())
        except Exception as e:
            logger.warning("Telegram POST %s error: %s", method, e)
            return {}

    def _send(self, text: str, parse_mode: str = "HTML") -> None:
        self._post("sendMessage", {
            "chat_id": self.chat_id,
            "text": text[:4096],
            "parse_mode": parse_mode,
        })

    # ── Command handlers ──────────────────────────────────────────────────────

    def _handle_status(self) -> None:
        """Tampilkan posisi open + metrics."""
        try:
            positions = self.connector.get_open_positions()
            account = self.connector.get_account_info()
            metrics = self.engine.memory.get_performance_metrics()

            if positions:
                pos_lines = "\n".join(
                    f"  • {p['pair']} {p['type']} {p['volume']}lot @ {p['open_price']:.5f} | PnL: {p['profit']:+.2f}"
                    for p in positions
                )
            else:
                pos_lines = "  Tidak ada posisi terbuka."

            msg = (
                f"📊 <b>Status Sistem</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Balance:</b> {account.get('balance', 0):.2f} {account.get('currency', '')}\n"
                f"<b>Equity:</b> {account.get('equity', 0):.2f} | PnL: {account.get('profit', 0):+.2f}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Open Positions:</b>\n{pos_lines}\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"<b>Performa:</b> {metrics}"
            )
            self._send(msg)
        except Exception as e:
            self._send(f"Error: {e}")

    def _handle_analyze(self) -> None:
        """Trigger analisis manual sekarang."""
        import os
        self._send("🔄 Memulai analisis manual...")
        try:
            pair = os.environ.get("DEFAULT_PAIR", "XAUUSDm")
            tf = os.environ.get("DEFAULT_TF", "M15")
            n = int(os.environ.get("DEFAULT_CANDLES", 50))

            from src.mt5.data_formatter import build_context_toon
            df = self.connector.get_chart(pair, tf, n)
            tick = self.connector.get_tick(pair)
            positions = self.connector.get_open_positions()
            account = self.connector.get_account_info()
            context = build_context_toon(pair, tf, df, tick, positions, account)

            result = self.engine.analyze(context, pair, tf)
            decision = result["decision"]

            # Kirim hasil
            lines = [l.strip() for l in decision.strip().splitlines() if l.strip()]
            verdict = next(
                (l for l in reversed(lines) if any(k in l for k in ("EKSEKUSI:", "HOLD:", "CLOSE:"))),
                lines[-1] if lines else "—"
            )
            self._send(f"🎯 <b>Hasil Analisis Manual</b>\n\n{verdict}")
        except Exception as e:
            self._send(f"❌ Error analisis: {e}")

    def _handle_chat(self, user_text: str, username: str) -> None:
        """Forward pesan bebas ke LLM sebagai chat."""
        self._send("💭 Memproses...")
        try:
            import os
            pair = os.environ.get("DEFAULT_PAIR", "XAUUSDm")
            # Buat context sederhana — hanya tanya ke LLM
            context = (
                f"Pesan dari trader ({username}):\n{user_text}\n\n"
                f"Jawab sebagai AI trading assistant. Jika diminta analisis, gunakan tools yang tersedia."
            )
            result = self.engine.analyze(context, pair, "M15")
            reply = result["decision"] or result["reasoning"]
            # Potong jika terlalu panjang
            if len(reply) > 3000:
                reply = reply[:3000] + "\n...(terpotong)"
            self._send(reply)
        except Exception as e:
            self._send(f"❌ Error: {e}")

    # ── Polling loop ──────────────────────────────────────────────────────────

    def _process_update(self, update: dict) -> None:
        msg = update.get("message") or update.get("channel_post")
        if not msg:
            return

        # Hanya proses pesan dari chat_id yang dikonfigurasi
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != self.chat_id:
            return

        text = msg.get("text", "").strip()
        if not text:
            return

        username = msg.get("from", {}).get("username") or msg.get("from", {}).get("first_name", "user")
        logger.info("[TelegramBot] Message from %s: %s", username, text[:80])

        cmd = text.lower().split()[0]
        if cmd in ("/status", "/s"):
            self._handle_status()
        elif cmd in ("/analyze", "/a"):
            threading.Thread(target=self._handle_analyze, daemon=True).start()
        elif cmd == "/help":
            self._send(
                "🤖 <b>Perintah tersedia:</b>\n"
                "/status — lihat posisi & performa\n"
                "/analyze — trigger analisis sekarang\n"
                "/help — daftar perintah\n\n"
                "Atau kirim pesan bebas untuk chat dengan AI."
            )
        else:
            # Pesan bebas → forward ke LLM
            threading.Thread(target=self._handle_chat, args=(text, username), daemon=True).start()

    def _poll(self) -> None:
        import urllib.parse
        logger.info("[TelegramBot] Polling started")
        while self._running:
            result = self._get("getUpdates", {"offset": self._offset, "timeout": 30})
            updates = result.get("result", [])
            for upd in updates:
                self._offset = upd["update_id"] + 1
                try:
                    self._process_update(upd)
                except Exception as e:
                    logger.error("[TelegramBot] Error processing update: %s", e)

    def start(self) -> None:
        """Jalankan polling di background thread."""
        self._running = True
        t = threading.Thread(target=self._poll, daemon=True, name="TelegramBotPoller")
        t.start()
        logger.info("[TelegramBot] Started — send /help ke grup untuk mulai")

    def stop(self) -> None:
        self._running = False
