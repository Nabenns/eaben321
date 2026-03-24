"""Main entry point — AI Forex Trading System."""

import logging
import os
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Pastikan data dir ada SEBELUM logging setup
Path("./data").mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("./data/trading.log"),
    ],
)
logger = logging.getLogger("main")

from src.mt5.connector import MT5Connector
from src.mt5.executor import MT5Executor
from src.mt5.data_formatter import build_context_toon
from src.memory.memory_manager import MemoryManager
from src.llm.provider import create_provider
from src.llm.engine import LLMEngine
from src.llm.tool_handler import ToolHandler
from src.learning.adaptive import AdaptiveLearner


def build_system():
    """Inisialisasi semua komponen sistem."""
    # MT5
    connector = MT5Connector(
        login=int(os.environ["MT5_LOGIN"]),
        password=os.environ["MT5_PASSWORD"],
        server=os.environ["MT5_SERVER"],
        path=os.environ.get("MT5_PATH"),
    )
    executor = MT5Executor(magic=int(os.environ.get("MT5_MAGIC", 20260325)))

    # Memory
    memory = MemoryManager(
        formula_path=os.environ.get("FORMULA_PATH", "./config/formula.yaml"),
        db_path=os.environ.get("DB_PATH", "./data/trades.db"),
        vector_dir=os.environ.get("VECTOR_DIR", "./data/chromadb"),
    )

    # LLM
    provider = create_provider()
    tool_handler = ToolHandler(connector, executor, memory)
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    engine = LLMEngine(
        tool_handler=tool_handler,
        memory=memory,
        provider=provider,
        dry_run=dry_run,
    )

    # Adaptive learner
    learner = AdaptiveLearner(
        memory=memory,
        provider=provider,
        trigger_every_n_trades=int(os.environ.get("LEARN_EVERY_N", 10)),
    )

    return connector, engine, learner


def run_analysis_cycle(connector, engine, learner):
    """Satu siklus analisis: fetch data → LLM analyze → optional learning."""
    pair = os.environ.get("DEFAULT_PAIR", "EURUSD")
    timeframe = os.environ.get("DEFAULT_TF", "M15")
    n_candles = int(os.environ.get("DEFAULT_CANDLES", 100))

    if not connector.is_connected():
        logger.warning("MT5 tidak terkoneksi, mencoba reconnect...")
        if not connector.connect():
            logger.error("Reconnect gagal, skip siklus ini.")
            return

    try:
        # Fetch data awal
        df = connector.get_chart(pair, timeframe, n_candles)
        tick = connector.get_tick(pair)
        positions = connector.get_open_positions()
        account = connector.get_account_info()

        # Build context dalam TOON
        context = build_context_toon(pair, timeframe, df, tick, positions, account)

        # Jalankan analisis LLM
        result = engine.analyze(context, pair, timeframe)
        logger.info("Keputusan: %s", result["decision"][:200])

        # Cek apakah perlu learning cycle
        if learner.should_learn():
            logger.info("Menjalankan adaptive learning...")
            learning_result = learner.run()
            logger.info("Learning result: %s", learning_result)

    except Exception as e:
        logger.exception("Error dalam siklus analisis: %s", e)


def main():
    logger.info("=" * 60)
    logger.info("AI Forex Trading System starting...")
    dry_run = os.environ.get("DRY_RUN", "true").lower() == "true"
    if dry_run:
        logger.info("MODE: DRY RUN (tidak ada order nyata)")
    else:
        logger.warning("MODE: LIVE TRADING — order nyata aktif!")
    logger.info("=" * 60)

    connector, engine, learner = build_system()

    # Connect ke MT5
    if not connector.connect():
        logger.error("Gagal connect ke MT5. Periksa .env dan pastikan MT5 berjalan.")
        sys.exit(1)

    interval_minutes = int(os.environ.get("INTERVAL_MINUTES", 15))
    logger.info("Interval analisis: setiap %d menit", interval_minutes)

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_analysis_cycle,
        "interval",
        minutes=interval_minutes,
        args=[connector, engine, learner],
        id="analysis_cycle",
    )

    # Jalankan sekali langsung saat start
    run_analysis_cycle(connector, engine, learner)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Sistem dihentikan oleh user.")
        connector.disconnect()


if __name__ == "__main__":
    main()
