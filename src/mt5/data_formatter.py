"""Data Formatter — konversi data MT5 ke format compact untuk efisiensi token LLM."""

import pandas as pd


def _fmt_num(v: float, decimals: int = 5) -> str:
    """Format angka — hapus trailing zeros."""
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".")


def _encode(data: dict) -> str:
    """
    Encoder compact custom — hemat token vs JSON.
    Format: KEY[N,]{field1,field2,...}: v1|v2|...; v1|v2|...
    """
    parts = []
    for key, rows in data.items():
        if not isinstance(rows, list):
            rows = [rows]
        if not rows:
            parts.append(f"{key}[0,]{{}}: (none)")
            continue
        fields = list(rows[0].keys())
        header = f"{key}[{len(rows)},]{{{','.join(fields)}}}"
        values = "; ".join("|".join(str(r.get(f, "")) for f in fields) for r in rows)
        parts.append(f"{header}: {values}")
    return "\n".join(parts)


def candles_to_toon(pair: str, timeframe: str, df: pd.DataFrame) -> str:
    records = [
        {
            "t": row["time"].strftime("%m%dT%H%M"),
            "o": _fmt_num(float(row["open"])),
            "h": _fmt_num(float(row["high"])),
            "l": _fmt_num(float(row["low"])),
            "c": _fmt_num(float(row["close"])),
            "v": int(row["volume"]),
        }
        for _, row in df.iterrows()
    ]
    return _encode({f"{pair}_{timeframe}": records})


def tick_to_toon(tick: dict) -> str:
    row = {
        "pair": tick["pair"],
        "bid": _fmt_num(tick["bid"]),
        "ask": _fmt_num(tick["ask"]),
        "spread_pts": _fmt_num(tick["spread_points"], 5),
        "time": tick["time"],
    }
    return _encode({"tick": [row]})


def positions_to_toon(positions: list[dict]) -> str:
    if not positions:
        return "positions[0,]{}: (none)"
    rows = [
        {
            "ticket": p["ticket"],
            "pair": p["pair"],
            "type": p["type"],
            "vol": p["volume"],
            "open_px": _fmt_num(p["open_price"]),
            "cur_px": _fmt_num(p["current_price"]),
            "sl": _fmt_num(p["sl"]),
            "tp": _fmt_num(p["tp"]),
            "pnl": round(p["profit"], 2),
        }
        for p in positions
    ]
    return _encode({"positions": rows})


def trade_history_to_toon(history: list[dict]) -> str:
    if not history:
        return "history[0,]{}: (none)"
    rows = [
        {
            "ticket": h["ticket"],
            "pair": h["pair"],
            "type": h["type"],
            "vol": h["volume"],
            "px": _fmt_num(h["price"]),
            "pnl": round(h["profit"], 2),
            "time": h["time"],
        }
        for h in history
    ]
    return _encode({"history": rows})


def account_to_toon(account: dict) -> str:
    row = {
        "bal": round(account["balance"], 2),
        "eq": round(account["equity"], 2),
        "margin": round(account["margin"], 2),
        "free": round(account["free_margin"], 2),
        "lvl%": round(account.get("margin_level", 0), 1),
        "pnl": round(account["profit"], 2),
        "cur": account["currency"],
        "lev": account["leverage"],
    }
    return _encode({"account": [row]})


def build_context_toon(
    pair: str,
    timeframe: str,
    candles_df: pd.DataFrame,
    tick: dict,
    positions: list[dict],
    account: dict,
) -> str:
    parts = [
        f"# Context: {pair} {timeframe}",
        tick_to_toon(tick),
        candles_to_toon(pair, timeframe, candles_df),
    ]
    if positions:
        parts.append(positions_to_toon(positions))
    parts.append(account_to_toon(account))
    return "\n\n".join(parts)
