"""Adaptive Learning Engine — LLM menganalisis performa dan menyesuaikan parameter formula."""

import logging

from src.llm.provider import LLMProvider, create_provider
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

LEARNING_PROMPT = """Kamu adalah analis performa trading yang bertugas mengevaluasi hasil strategi dan mengoptimalkan parameter.

## Performa Terkini (50 trade terakhir)
{metrics}

## 10 Trade Terakhir (detail)
{recent_trades}

## Parameter Formula Saat Ini
{current_params}

## Parameter yang Bisa Disesuaikan
{adjustable_params}

## Tugasmu
1. Analisis pola kegagalan — kondisi apa yang sering menghasilkan loss?
2. Identifikasi parameter mana yang perlu diubah
3. Propose nilai baru untuk parameter tersebut (dalam batas min-max)
4. Berikan reasoning yang jelas

## Format Output (WAJIB)
Akhiri dengan JSON block seperti ini:
```json
{{
  "adjustments": {{
    "param_name": new_value,
    ...
  }},
  "reason": "penjelasan singkat kenapa adjustment ini dilakukan"
}}
```

Jika tidak perlu adjustment, return:
```json
{{
  "adjustments": {{}},
  "reason": "parameter sudah optimal berdasarkan data terkini"
}}
```
"""


class AdaptiveLearner:
    def __init__(
        self,
        memory: MemoryManager,
        provider: LLMProvider = None,
        trigger_every_n_trades: int = 10,
    ):
        self.provider = provider or create_provider()
        self.memory = memory
        self.trigger_every_n_trades = trigger_every_n_trades

    def should_learn(self) -> bool:
        """Cek apakah sudah saatnya run learning cycle."""
        metrics = self.memory.db.get_performance_summary(n_trades=self.trigger_every_n_trades)
        return metrics["total"] > 0 and metrics["total"] % self.trigger_every_n_trades == 0

    def run(self) -> dict:
        """Jalankan satu siklus adaptive learning. Return adjustment yang dilakukan."""
        import json
        import re
        import yaml

        logger.info("Menjalankan adaptive learning cycle...")

        # Kumpulkan data untuk analisis
        metrics = self.memory.get_performance_metrics()
        recent_trades = self.memory.db.get_recent_trades(n=10)
        current_params = self.memory.get_current_formula_params()

        # Load formula untuk tahu batas min-max parameter
        formula_raw = yaml.safe_load(open(self.memory.formula_path, encoding="utf-8"))
        adjustable_params = formula_raw.get("adaptive_params", [])

        prompt = LEARNING_PROMPT.format(
            metrics=metrics,
            recent_trades=json.dumps(recent_trades, indent=2, default=str),
            current_params=current_params,
            adjustable_params=yaml.dump(adjustable_params, allow_unicode=True),
        )

        response = self.provider.chat(
            system="Kamu adalah analis performa trading.",
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            max_tokens=2048,
        )

        response_text = response.text

        # Parse JSON dari response
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if not json_match:
            logger.warning("Tidak bisa parse JSON dari learning response")
            return {"adjustments": {}, "reason": "parse error"}

        result = json.loads(json_match.group(1))
        adjustments = result.get("adjustments", {})
        reason = result.get("reason", "")

        if adjustments:
            # Validasi nilai dalam batas min-max
            param_limits = {p["name"]: p for p in adjustable_params}
            validated = {}
            for param, value in adjustments.items():
                if param in param_limits:
                    limits = param_limits[param]
                    value = max(limits.get("min", value), min(limits.get("max", value), value))
                    validated[param] = value

            if validated:
                self.memory.save_formula_params(validated, reason)
                logger.info("Adaptive adjustment diterapkan: %s | %s", validated, reason)
            result["adjustments"] = validated

        return result
