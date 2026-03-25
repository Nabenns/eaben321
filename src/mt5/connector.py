"""MT5 Connector — koneksi ke MetaTrader5 dan fetch data market."""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


class MT5Connector:
    def __init__(self, login: int, password: str, server: str, path: Optional[str] = None):
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self._connected = False

    def connect(self) -> bool:
        kwargs = {}
        if self.path:
            kwargs["path"] = self.path

        if not mt5.initialize(**kwargs):
            logger.error("MT5 initialize failed: %s", mt5.last_error())
            return False

        authorized = mt5.login(self.login, password=self.password, server=self.server)
        if not authorized:
            logger.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown()
            return False

        self._connected = True
        info = mt5.account_info()
        logger.info("Connected to MT5 — account %s, broker %s", info.login, info.company)
        return True

    def disconnect(self):
        mt5.shutdown()
        self._connected = False
        logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        return self._connected and mt5.terminal_info() is not None

    # ── Market Data ──────────────────────────────────────────────────────────

    def get_chart(self, pair: str, timeframe: str, n_candles: int = 100) -> pd.DataFrame:
        """Fetch OHLCV candles dari MT5."""
        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Timeframe tidak dikenal: {timeframe}")

        mt5.symbol_select(pair, True)  # pastikan symbol ada di Market Watch
        rates = mt5.copy_rates_from_pos(pair, tf, 0, n_candles)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Gagal fetch data {pair} {timeframe}: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df[["time", "open", "high", "low", "close", "volume"]]

    def get_tick(self, pair: str) -> dict:
        """Ambil bid/ask real-time dan spread."""
        mt5.symbol_select(pair, True)
        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            raise RuntimeError(f"Gagal ambil tick {pair}: {mt5.last_error()}")
        symbol_info = mt5.symbol_info(pair)
        spread_pips = (tick.ask - tick.bid) / symbol_info.point if symbol_info else None
        return {
            "pair": pair,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread_points": tick.ask - tick.bid,
            "spread_pips": spread_pips,
            "time": datetime.fromtimestamp(tick.time).isoformat(),
        }

    def get_open_positions(self, pair: Optional[str] = None) -> list[dict]:
        """Ambil semua posisi aktif, optional filter per pair."""
        positions = mt5.positions_get(symbol=pair) if pair else mt5.positions_get()
        if positions is None:
            return []
        return [
            {
                "ticket": p.ticket,
                "pair": p.symbol,
                "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": p.volume,
                "open_price": p.price_open,
                "current_price": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "swap": p.swap,
                "comment": p.comment,
                "open_time": datetime.fromtimestamp(p.time).isoformat(),
            }
            for p in positions
        ]

    def get_account_info(self) -> dict:
        """Ambil info akun: balance, equity, margin."""
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"Gagal ambil account info: {mt5.last_error()}")
        return {
            "login": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level,
            "profit": info.profit,
            "currency": info.currency,
            "leverage": info.leverage,
        }

    # ── Strategy Helpers ─────────────────────────────────────────────────────

    def get_session_info(self) -> dict:
        """Deteksi sesi trading aktif, AMDX/XAMD bias, dan high-probability window."""
        now_utc = datetime.now(timezone.utc)
        hour = now_utc.hour
        minute = now_utc.minute
        total_min = hour * 60 + minute

        # Sesi (UTC)
        # Asia:     00:00 – 09:00
        # London:   07:00 – 13:00
        # NY AM:    13:00 – 17:00
        # NY PM:    17:00 – 22:00
        sessions = []
        if 0 <= hour < 9:
            sessions.append("Asia")
        if 7 <= hour < 13:
            sessions.append("London")
        if 13 <= hour < 17:
            sessions.append("NY_AM")
        if 17 <= hour < 22:
            sessions.append("NY_PM")
        if not sessions:
            sessions.append("Off-Hours")

        # AMDX/XAMD bias — deteksi dari arah Asia session
        # Ambil data H1 pair default untuk cek Asia range
        asia_bias = "unknown"
        try:
            import os
            bias_pair = os.environ.get("DEFAULT_PAIR", "XAUUSDm")
            mt5.symbol_select(bias_pair, True)
            rates = mt5.copy_rates_from_pos(bias_pair, mt5.TIMEFRAME_H1, 0, 24)
            if rates is not None and len(rates) > 0:
                # Cari candle dari 00:00 UTC sampai 09:00 UTC hari ini
                today_date = now_utc.date()
                asia_candles = [
                    r for r in rates
                    if datetime.fromtimestamp(r["time"], tz=timezone.utc).date() == today_date
                    and datetime.fromtimestamp(r["time"], tz=timezone.utc).hour < 9
                ]
                if asia_candles:
                    asia_open = asia_candles[0]["open"]
                    asia_close = asia_candles[-1]["close"]
                    asia_high = max(r["high"] for r in asia_candles)
                    asia_low = min(r["low"] for r in asia_candles)
                    if asia_close > asia_open:
                        asia_bias = "A"   # Accumulation → AMDX pattern
                    else:
                        asia_bias = "X"   # Distribution/Expansion → XAMD pattern
        except Exception:
            pass

        # High-probability entry window: Q3/Q4 dari 90-min cycle di London atau NY AM
        quarter_info = self._get_90min_quarter(now_utc)
        is_high_prob = (
            any(s in sessions for s in ["London", "NY_AM"])
            and quarter_info["quarter"] in ["Q3", "Q4"]
        )

        # Apakah layak entry sesuai bias
        entry_allowed = False
        if asia_bias == "A" and "London" in sessions:
            entry_allowed = True
        elif asia_bias == "X" and "NY_AM" in sessions:
            entry_allowed = True
        elif asia_bias == "unknown":
            entry_allowed = any(s in sessions for s in ["London", "NY_AM"])

        return {
            "current_time_utc": now_utc.isoformat(),
            "active_sessions": sessions,
            "asia_bias": asia_bias,
            "amd_pattern": "AMDX" if asia_bias == "A" else ("XAMD" if asia_bias == "X" else "unknown"),
            "90min_quarter": quarter_info,
            "is_high_probability_window": is_high_prob,
            "entry_allowed_by_session": entry_allowed,
            "note": (
                "London entry OK (AMDX)" if (asia_bias == "A" and "London" in sessions)
                else "NY AM entry OK (XAMD, skip London)" if (asia_bias == "X" and "NY_AM" in sessions)
                else "Observe only" if not any(s in sessions for s in ["London", "NY_AM"])
                else "Active session — evaluate setup"
            ),
        }

    def _get_90min_quarter(self, dt: datetime) -> dict:
        """Hitung posisi kuartal saat ini dalam siklus 90 menit."""
        total_min = dt.hour * 60 + dt.minute
        cycle_start_min = (total_min // 90) * 90
        pos_in_cycle = total_min - cycle_start_min
        quarter_len = 90 / 4  # 22.5 menit

        q_num = int(pos_in_cycle // quarter_len) + 1
        q_num = min(q_num, 4)
        quarter = f"Q{q_num}"

        cycle_start_h = cycle_start_min // 60
        cycle_start_m = cycle_start_min % 60
        cycle_end_min = cycle_start_min + 90
        cycle_end_h = cycle_end_min // 60
        cycle_end_m = cycle_end_min % 60

        return {
            "quarter": quarter,
            "position_in_cycle_min": pos_in_cycle,
            "cycle_start": f"{cycle_start_h:02d}:{cycle_start_m:02d} UTC",
            "cycle_end": f"{cycle_end_h:02d}:{cycle_end_m:02d} UTC",
            "quarter_boundaries": {
                "Q1": f"{cycle_start_h:02d}:{cycle_start_m:02d}–+22.5m",
                "Q2": "+22.5m–+45m (True Open zone)",
                "Q3": "+45m–+67.5m (high-prob entry)",
                "Q4": "+67.5m–+90m (high-prob entry)",
            },
        }

    def get_current_quarter(self, cycle: str = "macro_90min") -> dict:
        """Kembalikan posisi kuartal saat ini untuk cycle tertentu."""
        now_utc = datetime.now(timezone.utc)

        if cycle == "macro_90min":
            return self._get_90min_quarter(now_utc)

        if cycle == "daily":
            # 4 kuartal per hari: Q1 00-06, Q2 06-12, Q3 12-18, Q4 18-24
            q_num = now_utc.hour // 6 + 1
            quarter = f"Q{q_num}"
            return {
                "quarter": quarter,
                "current_hour_utc": now_utc.hour,
                "boundaries": {
                    "Q1": "00:00–06:00 UTC (Asia accumulation)",
                    "Q2": "06:00–12:00 UTC (London open)",
                    "Q3": "12:00–18:00 UTC (NY AM)",
                    "Q4": "18:00–24:00 UTC (NY PM / late)",
                },
            }

        if cycle == "weekly":
            # Hari dalam minggu: Mon=Q1, Tue-Wed=Q2, Thu=Q3, Fri=Q4
            weekday = now_utc.weekday()  # 0=Mon
            mapping = {0: "Q1", 1: "Q2", 2: "Q2", 3: "Q3", 4: "Q4"}
            return {
                "quarter": mapping.get(weekday, "Weekend"),
                "weekday": now_utc.strftime("%A"),
            }

        return {"error": f"cycle tidak dikenal: {cycle}"}

    def get_ndog_nwog(self, pair: str) -> dict:
        """
        Hitung NDOG (New Day Opening Gap) dan NWOG (New Week Opening Gap).
        NDOG = gap antara close kemarin dan open hari ini (00:00 server).
        NWOG = gap antara close Jumat lalu dan open Senin ini.
        Level ini bertindak sebagai S/R kuat.
        """
        # Ambil data D1 untuk NDOG & NWOG
        rates_d1 = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_D1, 0, 10)
        if rates_d1 is None or len(rates_d1) < 2:
            raise RuntimeError(f"Gagal ambil data D1 untuk {pair}: {mt5.last_error()}")

        # NDOG: selisih antara close candle D1 kemarin dan open candle D1 hari ini
        today_candle = rates_d1[-1]
        yesterday_candle = rates_d1[-2]

        today_open = today_candle["open"]
        yesterday_close = yesterday_candle["close"]

        ndog_gap = today_open - yesterday_close
        ndog_midpoint = (today_open + yesterday_close) / 2
        ndog_exists = abs(ndog_gap) > 0.00001

        ndog = {
            "exists": ndog_exists,
            "yesterday_close": round(yesterday_close, 5),
            "today_open": round(today_open, 5),
            "gap_size": round(ndog_gap, 5),
            "midpoint": round(ndog_midpoint, 5),
            "direction": "bullish_gap" if ndog_gap > 0 else "bearish_gap" if ndog_gap < 0 else "no_gap",
            "sr_zone": {
                "upper": round(max(today_open, yesterday_close), 5),
                "lower": round(min(today_open, yesterday_close), 5),
            },
        }

        # NWOG: cari Monday open vs Friday close
        nwog = {"exists": False, "note": "Data tidak cukup untuk kalkulasi NWOG"}
        try:
            # Ambil data H1 untuk deteksi weekly open dengan lebih presisi
            rates_h1 = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_H1, 0, 168)
            if rates_h1 is not None and len(rates_h1) > 0:
                now_utc = datetime.now(timezone.utc)
                current_weekday = now_utc.weekday()  # 0=Mon

                # Cari open Senin minggu ini
                monday_candle = None
                friday_close_candle = None

                for r in rates_h1:
                    dt = datetime.fromtimestamp(r["time"], tz=timezone.utc)
                    if dt.weekday() == 0 and dt.hour == 0:  # Senin 00:00
                        if monday_candle is None:
                            monday_candle = r
                    if dt.weekday() == 4:  # Jumat
                        friday_close_candle = r  # ambil yang terakhir

                if monday_candle and friday_close_candle:
                    monday_open = monday_candle["open"]
                    friday_close = friday_close_candle["close"]
                    nwog_gap = monday_open - friday_close
                    nwog_mid = (monday_open + friday_close) / 2

                    nwog = {
                        "exists": abs(nwog_gap) > 0.00001,
                        "friday_close": round(friday_close, 5),
                        "monday_open": round(monday_open, 5),
                        "gap_size": round(nwog_gap, 5),
                        "midpoint": round(nwog_mid, 5),
                        "direction": "bullish_gap" if nwog_gap > 0 else "bearish_gap" if nwog_gap < 0 else "no_gap",
                        "sr_zone": {
                            "upper": round(max(monday_open, friday_close), 5),
                            "lower": round(min(monday_open, friday_close), 5),
                        },
                    }
        except Exception as e:
            nwog["note"] = f"Error kalkulasi NWOG: {e}"

        return {
            "pair": pair,
            "ndog": ndog,
            "nwog": nwog,
            "usage_note": (
                "NDOG/NWOG berfungsi sebagai S/R kuat. "
                "Valid sebagai Stage1 HTF setup jika dikonfirmasi SMT di level ini."
            ),
        }

    def get_trade_history(self, n: int = 20, pair: Optional[str] = None) -> list[dict]:
        """Ambil N trade terakhir dari history."""
        deals = mt5.history_deals_get(
            datetime(2000, 1, 1), datetime.now()
        )
        if deals is None:
            return []

        deals_list = [d for d in deals if d.entry == mt5.DEAL_ENTRY_OUT]
        if pair:
            deals_list = [d for d in deals_list if d.symbol == pair]

        deals_list = sorted(deals_list, key=lambda d: d.time, reverse=True)[:n]

        return [
            {
                "ticket": d.ticket,
                "pair": d.symbol,
                "type": "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL",
                "volume": d.volume,
                "price": d.price,
                "profit": d.profit,
                "swap": d.swap,
                "comment": d.comment,
                "time": datetime.fromtimestamp(d.time).isoformat(),
            }
            for d in deals_list
        ]
