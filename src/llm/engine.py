"""LLM Engine — otak sistem trading, provider-agnostic (OpenAI / Anthropic / Groq)."""

import logging
from datetime import datetime

from src.llm.tools import TOOLS
from src.llm.provider import LLMProvider, GroqProvider, create_provider
from src.llm.tool_handler import ToolHandler
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10


class LLMEngine:
    def __init__(
        self,
        tool_handler: ToolHandler,
        memory: MemoryManager,
        provider: LLMProvider = None,
        dry_run: bool = False,
        pairs_config: dict = None,
    ):
        self.provider = provider or create_provider()
        self.tool_handler = tool_handler
        self.memory = memory
        self.dry_run = dry_run
        self.pairs_config = pairs_config or {}

    def _build_correlation_section(self, pair: str) -> str:
        corr_map = self.pairs_config.get("correlation_map", {})
        corr_pairs = corr_map.get(pair, [])
        pair_settings = self.pairs_config.get("pair_settings", {})
        risk = self.pairs_config.get("risk", {})
        ps = pair_settings.get(pair, {})
        lines = [
            f"## Pair Aktif: {pair}",
            f"- Korelasi untuk SMT/SSMT/Fill/Mirror: {', '.join(corr_pairs) if corr_pairs else 'tidak dikonfigurasi'}",
            f"- Min FVG: {ps.get('min_fvg_pts', 'N/A')} pts | SL buffer: {ps.get('sl_buffer_pts', 'N/A')} pts",
            f"- Max spread: {ps.get('max_spread_pts', 'N/A')} pts | Default lot: {ps.get('default_lot', 'N/A')}",
            f"- Risk per trade: max {risk.get('max_risk_per_trade_pct', 2)}% equity | Min R:R = {risk.get('min_rr_ratio', 2)}",
            f"- Max open trades: {risk.get('max_open_trades', 3)} | Max daily DD: {risk.get('max_daily_drawdown_pct', 5)}%",
        ]
        return "\n".join(lines)

    def _build_system_prompt(self, pair: str = "") -> str:
        formula = self.memory.get_formula(compact=False)
        params = self.memory.get_current_formula_params()
        metrics = self.memory.get_performance_metrics()
        mode_note = "\n⚠️ DRY RUN MODE: Jangan panggil execute_trade." if self.dry_run else ""
        correlation_section = self._build_correlation_section(pair) if pair else ""
        conversations = self.memory.get_recent_conversations(n=15)
        conv_section = f"\n## Instruksi & Percakapan Trader (Memory)\n{conversations}\n" if conversations else ""

        return f"""Kamu adalah AI trader forex yang disiplin. Kamu HARUS mengikuti formula strategi di bawah secara ketat.{mode_note}
{conv_section}
{correlation_section}

## WORKFLOW WAJIB — ikuti urutan ini setiap siklus analisis:

### STEP 1 — Cek Session & Timing (WAJIB, call tools ini dulu)
- Panggil `get_session_info` → cek sesi aktif dan AMDX/XAMD bias
- Panggil `get_current_quarter` dengan cycle="macro_90min" → cek apakah Q3/Q4
- **RULE: Entry HANYA di London atau NY AM session, DAN hanya di Q3/Q4 dari 90m cycle**
- Jika Asia bias=A → entry di London session; jika Asia bias=X → skip London, entry di NY AM
- Jika kondisi timing tidak terpenuhi → LANGSUNG HOLD, tidak perlu analisis lanjut

### STEP 2 — Identifikasi Liquidity Sweep (Trigger)
- Dari chart yang diberikan (M15 default), identifikasi apakah ada sweep liquidity:
  - BSL/SSL, EQH/EQL, PWH/PWL, PDH/PDL, Session High/Low
- Jika tidak ada sweep → HOLD
- Jika ada sweep: tentukan bias (sweep high=SELL, sweep low=BUY)

### STEP 3 — Konfirmasi Multi-TF & Multi-Pair (gunakan tools)
- Panggil `get_chart` untuk TF yang lebih tinggi (misal H1 atau H4) untuk konfirmasi trend
- Panggil `get_chart` untuk pair korelasi yang tercantum di atas
- Panggil `get_ndog_nwog` untuk pair yang dianalisis → cek level gap sebagai S/R
- Panggil `get_tick` untuk pair korelasi → cek divergence harga

### STEP 4 — Verifikasi 2-Stages (WAJIB minimal 2 konfirmasi)
Cek apakah minimal 2 dari setup berikut terpenuhi di HTF (Stage 1):
- **SSMT**: Pair korelasi buat swing serupa di level sama, tapi harga berbeda (divergence), dalam SSMT zone (true open area)
- **SMT Fill**: FVG terbentuk di kedua pair korelasi (minimal 2 pair), ukuran gap harus proporsional (jika tidak proporsional = INVALID)
- **Mirror PD**: Struktur equilibrium (0.5 Fibo) terbentuk di kedua pair, struktur harus proporsional ukurannya
- **NDOG/NWOG**: Harga bereaksi di level gap, dikonfirmasi SMT

Kemudian konfirmasi Stage 2 di LTF (turun max 2 step dari Stage 1, misal H1→M15 atau M15→M5):
- Cari MSS (wick level) atau CISD (body close melewati level) untuk konfirmasi reversal
- Cari PSP (candle dengan wick proporsional di kedua pair) jika sekalian ada SSMT/Fill/Mirror

**Jika hanya 1 konfirmasi atau 0 → HOLD**

### STEP 5 — Entry Decision
Jika semua step di atas terpenuhi:
- Tentukan entry price (ASK untuk BUY, BID untuk SELL)
- SL: di swing low/high Stage 2 (wick PSP atau swing MSS)
- TP1: Internal liquidity terdekat yang fresh | TP2: External liquidity (BSL/SSL)
- Lot: sesuai risk management (default max 2% equity per trade)
- Panggil `query_memory` untuk cek situasi serupa di masa lalu
- Jika DRY RUN: tulis keputusan tanpa call execute_trade

## Formula Lengkap (referensi detail)
{formula}

## Parameter Adaptif Saat Ini
{params}

## Performa Historis
{metrics}

## FORMAT OUTPUT WAJIB
Tulis reasoning per-step, lalu akhiri dengan SALAH SATU:
```
EKSEKUSI: [BUY/SELL] [PAIR] [LOT] lot | Entry: [price] | SL: [price] | TP1: [price] | TP2: [price] | Alasan: [ringkas]
HOLD: [alasan spesifik — step mana yang tidak terpenuhi]
CLOSE: Ticket [ticket] | Alasan: [ringkas]
```
"""

    def analyze(self, initial_context: str, pair: str, timeframe: str) -> dict:
        """Jalankan satu siklus analisis lengkap. Return dict keputusan + reasoning."""
        system = self._build_system_prompt(pair)
        messages = [{"role": "user", "content": initial_context}]

        reasoning_log = []
        tool_calls_log = []
        decision = None

        for round_num in range(MAX_TOOL_ROUNDS):
            response = self.provider.chat(system=system, messages=messages, tools=TOOLS)

            if response.text:
                reasoning_log.append(response.text)

            if response.finished:
                decision = response.text
                break

            tool_results = []
            for tc in response.tool_calls:
                tool_calls_log.append({"tool": tc["name"], "input": tc["input"]})
                logger.info("[Round %d] Tool: %s | Input: %s", round_num + 1, tc["name"], tc["input"])

                if self.dry_run and tc["name"] == "execute_trade":
                    result_text = f"[DRY RUN] Simulasi {tc['input'].get('action')} {tc['input'].get('pair')} — tidak dieksekusi."
                else:
                    result_text = self.tool_handler.handle(tc["name"], tc["input"])

                tool_results.append({"id": tc["id"], "content": str(result_text)})

            assistant_msg, tool_msg = self.provider.build_tool_result_message(None, tool_results)
            messages.append(assistant_msg)
            if isinstance(tool_msg, list):
                messages.extend(tool_msg)
            else:
                messages.append(tool_msg)

        else:
            logger.warning("Mencapai batas maksimal tool rounds (%d)", MAX_TOOL_ROUNDS)
            decision = "MAX_ROUNDS_REACHED"

        result = {
            "timestamp": datetime.now().isoformat(),
            "pair": pair,
            "timeframe": timeframe,
            "reasoning": "\n".join(reasoning_log),
            "tool_calls": tool_calls_log,
            "decision": decision or "",
        }

        self.memory.save_episode(result)
        return result
