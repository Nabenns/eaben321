# Product Requirements Document
# AI Forex Trading System

**Version:** 1.0
**Date:** 2026-03-25
**Status:** Draft

---

## 1. Overview

Sistem trading forex otomatis berbasis LLM yang mengambil data dari MetaTrader 5 (MT5), menganalisis pasar menggunakan formula strategi milik user, mengeksekusi trade, dan secara adaptif belajar dari performa historis untuk menyempurnakan parameter formula.

---

## 2. Problem Statement

Trading forex manual membutuhkan monitoring 24 jam, konsistensi emosional, dan kemampuan menganalisis multi-timeframe secara simultan. Sistem ini menggantikan proses manual tersebut dengan LLM yang mampu berpikir layaknya trader profesional — meminta data yang dibutuhkan, menganalisis, mengeksekusi, dan belajar dari setiap trade.

---

## 3. Goals

- Mengotomatiskan eksekusi strategi trading forex milik user
- LLM mampu request data multi-timeframe dan multi-pair secara dinamis
- Sistem memiliki persistent memory untuk belajar dari histori trade
- Formula user dapat disesuaikan parameternya secara adaptif berdasarkan performa

---

## 4. Non-Goals

- Tidak mendukung aset selain forex (untuk versi awal)
- Tidak membangun exchange atau broker sendiri
- Tidak menyediakan UI dashboard kompleks di versi pertama

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MT5 Platform                         │
│              (Data Source + Execution Engine)               │
└──────────────────────┬──────────────────▲───────────────────┘
                       │ OHLCV, Tick       │ Order
                       ▼                  │
┌─────────────────────────────────────────────────────────────┐
│                    Python Bridge Layer                      │
│         (MT5 Connector, Data Formatter, Order Handler)      │
└──────────────────────┬──────────────────▲───────────────────┘
                       │ Structured Data   │ Trade Signal
                       ▼                  │
┌─────────────────────────────────────────────────────────────┐
│                      LLM Core Engine                        │
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Tool Caller │  │ Formula      │  │ Decision Engine   │  │
│  │ (Data Req.) │  │ Interpreter  │  │ (BUY/SELL/HOLD)   │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ Read/Write
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      Memory System                          │
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ Short-term   │  │ Long-term     │  │ Structured DB   │  │
│  │ (Context     │  │ (Vector DB /  │  │ (Trade logs,    │  │
│  │  Window)     │  │  ChromaDB)    │  │  metrics, params│  │
│  └──────────────┘  └───────────────┘  └─────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Core Components

### 6.1 MT5 Connector (Python)

**Fungsi:** Jembatan antara MT5 dan sistem AI.

**Responsibilities:**
- Koneksi ke MT5 via library `MetaTrader5` Python
- Fetch data OHLCV berdasarkan request LLM (pair + timeframe)
- Hitung indikator teknikal jika diminta
- Kirim order eksekusi (BUY/SELL/CLOSE) ke MT5
- Ambil status posisi aktif dan histori trade

**Data yang di-fetch:**
```
- OHLCV candles (n candle terakhir)
- Bid/Ask price real-time
- Spread saat ini
- Open positions
- Account info (balance, equity, margin)
- Trade history
```

**Format pengiriman ke LLM — TOON:**
```toon
meta{pair,tf,spread,bid,ask}:
  EURUSD,M15,0.8,1.08501,1.08509

candles[100,]{time,open,high,low,close,vol}:
  20260325T0800,1.0832,1.0845,1.0828,1.0840,1520
  20260325T0815,1.0840,1.0858,1.0836,1.0855,1890
  ...
```
Hemat **40-52% token** dibanding JSON untuk setiap request data.

---

### 6.2 LLM Core Engine

**Model:** Claude (claude-sonnet atau claude-opus) via Anthropic API
**Mode:** Agentic dengan Tool Calling

**System Prompt berisi:**
- Formula strategi user (rules, kondisi entry/exit, risk management)
- Instruksi cara menggunakan memory
- Kepribadian sebagai trader yang disiplin dan adaptif
- Parameter formula terkini (hasil pembelajaran)

**Alur reasoning LLM:**
```
1. Terima data awal (default: pair + TF utama)
2. Analisis berdasarkan formula
3. Jika butuh konfirmasi → request data tambahan via tool call
4. Query memory → cari situasi serupa di masa lalu
5. Ambil keputusan: BUY / SELL / HOLD + parameter (lot, SL, TP)
6. Kirim keputusan ke Python Bridge
7. Catat reasoning + hasil ke memory
```

---

### 6.3 Tool Calling System

LLM dapat memanggil tools berikut secara dinamis:

| Tool | Parameter | Deskripsi |
|------|-----------|-----------|
| `get_chart` | `pair`, `timeframe`, `n_candles` | Ambil data OHLCV dari MT5 |
| `get_indicator` | `pair`, `tf`, `indicator`, `params` | Hitung indikator (MA, RSI, dll) |
| `get_open_positions` | - | Lihat semua posisi aktif |
| `get_account_info` | - | Balance, equity, margin level |
| `get_trade_history` | `n`, `pair` | N trade terakhir + hasil |
| `query_memory` | `context`, `n_results` | Cari situasi serupa di vector DB |
| `execute_trade` | `pair`, `action`, `lot`, `sl`, `tp`, `comment` | Eksekusi order di MT5 |
| `close_position` | `ticket` | Tutup posisi spesifik |
| `update_formula_params` | `params_json` | Simpan penyesuaian parameter formula |

**Contoh multi-step reasoning:**
```
LLM step 1: get_chart("EURUSD", "M15", 100)
LLM step 2: get_chart("EURUSD", "H1", 50)       ← konfirmasi trend
LLM step 3: get_chart("XAUUSD", "M15", 50)      ← cek korelasi
LLM step 4: query_memory("EURUSD bearish H1, M15 ranging")
LLM step 5: execute_trade("EURUSD", "SELL", 0.1, sl=1.0850, tp=1.0780)
```

---

### 6.4 Memory System

#### Short-term Memory
- Context window LLM aktif
- Berisi: data trade sesi ini, reasoning chain, tool call results

#### Long-term Memory (Vector DB — ChromaDB)
- Setiap trade disimpan sebagai "episode":
  ```json
  {
    "timestamp": "...",
    "pair": "EURUSD",
    "timeframe": "M15",
    "market_context": "...",
    "reasoning": "...",
    "action": "SELL",
    "params": {"lot": 0.1, "sl": 1.0850, "tp": 1.0780},
    "outcome": {"pnl": -15.5, "result": "loss", "reason": "news spike"}
  }
  ```
- Saat analisis baru, LLM query vector DB dengan embedding konteks market saat ini
- Retrieve top-N situasi paling mirip sebagai referensi

#### Structured DB (SQLite / PostgreSQL)
- Tabel `trades`: semua histori trade dengan P&L
- Tabel `formula_params`: versi parameter formula + tanggal update + performa
- Tabel `performance_metrics`: win rate, profit factor, max drawdown per periode
- Tabel `pair_stats`: performa per pair dan timeframe

---

### 6.5 Adaptive Learning Engine

**Mekanisme:** LLM menyesuaikan parameter formula berdasarkan analisis performa historis.

**Trigger pembelajaran:**
- Setiap N trade selesai (misalnya setiap 10 trade)
- Setiap akhir sesi trading harian
- Ketika drawdown melewati threshold tertentu

**Proses:**
```
1. Pull performa terbaru dari DB (win rate, avg RR, consecutive losses)
2. LLM analisis: "parameter mana yang berkontribusi pada loss?"
3. LLM propose adjustment parameter formula
4. Simpan parameter baru ke DB (versioned)
5. Update system prompt dengan parameter terbaru
6. Log reasoning di balik perubahan
```

**Formula parameter yang bisa disesuaikan:**
- Threshold kondisi entry/exit
- Filter waktu trading (jam/hari)
- Multiplier lot sizing
- SL/TP ratio
- Kondisi filter tambahan

---

## 7. Data Flow Detail

```
[Setiap N menit / event trigger]
        │
        ▼
Python fetch data default (pair utama, TF utama dari formula)
        │
        ▼
Data dikirim ke LLM sebagai initial context
        │
        ▼
LLM mulai reasoning loop:
  ├── Perlu data tambahan? → Tool call get_chart / get_indicator
  ├── Cek memory → Tool call query_memory
  ├── Cek posisi aktif → Tool call get_open_positions
  └── Keputusan final
        │
        ▼
BUY / SELL → Tool call execute_trade → MT5 eksekusi
HOLD → Log reasoning, tunggu trigger berikutnya
        │
        ▼
Simpan episode ke memory (vector DB + structured DB)
        │
        ▼
[Jika trigger learning] → Adaptive learning cycle
```

---

## 8. Formula Integration

User menyediakan formula trading dalam format terstruktur:

```yaml
formula:
  name: "Nama Strategi"
  version: "1.0"

  entry_conditions:
    long:
      - condition: "..."
      - condition: "..."
    short:
      - condition: "..."

  exit_conditions:
    take_profit: "..."
    stop_loss: "..."
    trailing: "..."

  filters:
    time: "..."
    spread_max: "..."

  risk_management:
    lot_method: "fixed | percent_balance | formula"
    lot_value: "..."
    max_open_positions: "..."
    max_daily_loss: "..."

  adaptive_params:
    - name: "param_name"
      default: value
      min: value
      max: value
      description: "apa yang dikontrol parameter ini"
```

LLM menerima formula ini di system prompt dan menggunakannya sebagai panduan keputusan, bukan aturan kaku — LLM boleh **tidak** eksekusi jika konteks market tidak mendukung.

---

## 9. Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Data Source & Execution | MetaTrader 5 |
| MT5 Python Bridge | `MetaTrader5` library |
| LLM | Claude (Anthropic API) via `anthropic` SDK |
| **Data Serialization** | **TOON (`toon-format`) — 40-52% token reduction vs JSON** |
| Orchestration | Python (async) |
| Vector DB | ChromaDB (lokal) |
| Structured DB | SQLite (dev) / PostgreSQL (prod) |
| Embedding | `sentence-transformers` atau Anthropic embeddings |
| Scheduler | APScheduler |
| Logging | Python `logging` + file rotation |

---

## 10. Project Structure

```
ai-trade/
├── PRD.md
├── README.md
├── .env                        # API keys, MT5 credentials
├── config/
│   └── formula.yaml            # Formula strategi user
├── src/
│   ├── mt5/
│   │   ├── connector.py        # Koneksi & fetch data MT5
│   │   ├── executor.py         # Eksekusi order
│   │   └── data_formatter.py   # Format data untuk LLM
│   ├── llm/
│   │   ├── engine.py           # LLM core, system prompt builder
│   │   ├── tools.py            # Definisi semua tools
│   │   └── tool_handler.py     # Handler eksekusi tool calls
│   ├── memory/
│   │   ├── vector_store.py     # ChromaDB interface
│   │   ├── structured_db.py    # SQLite/PostgreSQL interface
│   │   └── memory_manager.py   # Koordinasi semua layer memory
│   ├── learning/
│   │   └── adaptive.py         # Adaptive learning engine
│   └── main.py                 # Entry point, scheduler
└── tests/
    └── ...
```

---

## 11. Safety & Risk Controls

- **Hard stop loss:** Jika daily loss > X%, sistem berhenti otomatis hari itu
- **Max drawdown kill switch:** Jika drawdown > threshold, sistem pause dan alert user
- **Spread filter:** Tidak eksekusi jika spread terlalu lebar (news time)
- **Confirmation required mode:** Mode di mana LLM propose trade, user approve manual
- **Dry run / paper trading mode:** Simulasi tanpa eksekusi nyata
- **Rate limiting:** Batas maksimal order per jam/hari
- **Logging semua keputusan:** Setiap reasoning LLM disimpan untuk audit

---

## 12. Milestones

| Phase | Deliverable |
|-------|-------------|
| **Phase 1** | MT5 connector + fetch data + basic LLM call |
| **Phase 2** | Tool calling system (multi-TF, multi-pair request) |
| **Phase 3** | Memory system (vector DB + structured DB) |
| **Phase 4** | Formula integration + eksekusi trade |
| **Phase 5** | Adaptive learning engine |
| **Phase 6** | Safety controls + paper trading mode |
| **Phase 7** | Testing, tuning, live deployment |

---

## 13. Open Questions

- [ ] Formula strategi user (detail rules akan diberikan user)
- [ ] Pair utama yang ditarget
- [ ] Timeframe default sebagai trigger analisis
- [ ] Threshold risk management (max daily loss %, max drawdown %)
- [ ] Frekuensi analisis (setiap berapa menit sistem berjalan)
- [ ] Model Claude yang digunakan (Sonnet untuk speed vs Opus untuk reasoning)

---

*PRD ini akan diupdate setelah formula strategi user diberikan.*
