"""Tool definitions untuk LLM — semua tools yang bisa di-call LLM saat analisis."""

TOOLS = [
    {
        "name": "get_chart",
        "description": (
            "Ambil data OHLCV (candles) dari MT5 untuk pair dan timeframe tertentu. "
            "Gunakan untuk konfirmasi trend, cek multi-timeframe, atau analisis pair lain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "Symbol forex, contoh: EURUSD, XAUUSD, GBPUSD"},
                "timeframe": {
                    "type": "string",
                    "enum": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
                    "description": "Timeframe chart",
                },
                "n_candles": {
                    "type": "integer",
                    "description": "Jumlah candle yang diambil (default 100, max 500)",
                    "default": 100,
                },
            },
            "required": ["pair", "timeframe"],
        },
    },
    {
        "name": "get_tick",
        "description": "Ambil harga bid/ask real-time dan spread untuk suatu pair.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "Symbol forex"}
            },
            "required": ["pair"],
        },
    },
    {
        "name": "get_open_positions",
        "description": "Lihat semua posisi yang sedang terbuka. Optional filter per pair.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "Filter per pair (optional, kosongkan untuk semua)"}
            },
        },
    },
    {
        "name": "get_account_info",
        "description": "Ambil informasi akun: balance, equity, margin, free margin.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_trade_history",
        "description": "Ambil histori trade terakhir. Gunakan untuk evaluasi performa terkini.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Jumlah trade terakhir (default 20)", "default": 20},
                "pair": {"type": "string", "description": "Filter per pair (optional)"},
            },
        },
    },
    {
        "name": "query_memory",
        "description": (
            "Cari situasi market serupa dari memori historis. "
            "Gunakan untuk melihat apa yang terjadi di masa lalu pada kondisi market mirip."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Deskripsi kondisi market saat ini untuk dicari padanannya di memori",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Jumlah hasil yang dikembalikan (default 3)",
                    "default": 3,
                },
            },
            "required": ["context"],
        },
    },
    {
        "name": "execute_trade",
        "description": (
            "Eksekusi order BUY atau SELL di MT5. "
            "HANYA panggil setelah analisis selesai dan keputusan sudah final."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "Symbol forex"},
                "action": {"type": "string", "enum": ["BUY", "SELL"], "description": "Arah trade"},
                "lot": {"type": "number", "description": "Volume lot"},
                "sl": {"type": "number", "description": "Stop loss price (0 jika tidak pakai)"},
                "tp": {"type": "number", "description": "Take profit price (0 jika tidak pakai)"},
                "comment": {"type": "string", "description": "Komentar order (optional)"},
            },
            "required": ["pair", "action", "lot"],
        },
    },
    {
        "name": "close_position",
        "description": "Tutup posisi aktif berdasarkan ticket number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "integer", "description": "Ticket number posisi yang akan ditutup"}
            },
            "required": ["ticket"],
        },
    },
    {
        "name": "get_session_info",
        "description": (
            "Ambil informasi sesi trading saat ini: sesi aktif (Asia/London/NY AM/NY PM), "
            "AMDX/XAMD bias berdasarkan pergerakan Asia, dan apakah saat ini adalah waktu high-probability entry."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_current_quarter",
        "description": (
            "Cek posisi kuartal saat ini dalam siklus tertentu (daily atau macro_90min). "
            "Gunakan untuk menentukan apakah saat ini adalah Q3/Q4 dari 90m cycle (high-probability entry window)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cycle": {
                    "type": "string",
                    "enum": ["daily", "macro_90min", "weekly"],
                    "description": "Jenis siklus yang ingin dicek",
                }
            },
            "required": ["cycle"],
        },
    },
    {
        "name": "get_ndog_nwog",
        "description": (
            "Ambil level NDOG (New Day Opening Gap) dan NWOG (New Week Opening Gap) untuk suatu pair. "
            "Level ini berfungsi sebagai support/resistance kuat, terutama jika dikombinasikan dengan SMT."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pair": {"type": "string", "description": "Symbol forex, contoh: XAUUSD"}
            },
            "required": ["pair"],
        },
    },
    {
        "name": "update_formula_params",
        "description": (
            "Simpan penyesuaian parameter formula ke memori. "
            "Gunakan setelah analisis performa untuk update parameter yang perlu diubah."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": "Dict parameter yang diupdate beserta nilai barunya",
                },
                "reason": {
                    "type": "string",
                    "description": "Alasan melakukan adjustment ini",
                },
            },
            "required": ["params", "reason"],
        },
    },
]
