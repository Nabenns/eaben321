"""MT5 Executor — eksekusi order ke MetaTrader 5."""

import logging
from typing import Literal

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)

OrderType = Literal["BUY", "SELL"]


class MT5Executor:
    def __init__(self, magic: int = 20260325, slippage: int = 10):
        self.magic = magic
        self.slippage = slippage

    def execute_trade(
        self,
        pair: str,
        action: OrderType,
        lot: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "ai-trade",
    ) -> dict:
        """Kirim order market ke MT5. Return result dict."""
        order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            return {"success": False, "error": f"Gagal ambil tick {pair}"}

        price = tick.ask if action == "BUY" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error() if result is None else result.comment
            logger.error("Order gagal: %s | request: %s", error, request)
            return {"success": False, "error": str(error), "retcode": getattr(result, "retcode", None)}

        logger.info("Order sukses: %s %s %.2f lot @ %.5f | ticket %s", action, pair, lot, price, result.order)
        return {
            "success": True,
            "ticket": result.order,
            "pair": pair,
            "action": action,
            "lot": lot,
            "price": price,
            "sl": sl,
            "tp": tp,
        }

    def close_position(self, ticket: int) -> dict:
        """Tutup posisi berdasarkan ticket."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Posisi ticket {ticket} tidak ditemukan"}

        pos = positions[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": "ai-trade close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error() if result is None else result.comment
            logger.error("Close gagal ticket %s: %s", ticket, error)
            return {"success": False, "error": str(error)}

        logger.info("Posisi %s ditutup @ %.5f", ticket, price)
        return {"success": True, "ticket": ticket, "close_price": price}

    def modify_position(self, ticket: int, sl: float, tp: float) -> dict:
        """Modify SL/TP posisi aktif."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error() if result is None else result.comment
            return {"success": False, "error": str(error)}
        return {"success": True, "ticket": ticket, "sl": sl, "tp": tp}
