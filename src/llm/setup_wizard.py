"""
Browser Provider Setup Wizard.

Run standalone to configure and authenticate your browser provider:
    python -m src.llm.setup_wizard

This will:
  1. Show provider selection menu (ChatGPT / Claude)
  2. Open the browser visibly
  3. Let you log in manually
  4. Save the session to disk
  5. Confirm success

After running this wizard, the system will run fully headless on every
subsequent start — no browser window, no manual interaction needed.
"""

import asyncio
import os
import sys
from pathlib import Path

# Make sure project root is on sys.path when running as __main__
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SESSION_DIR = os.environ.get("BROWSER_SESSION_DIR", "data/sessions")
PROVIDER_CHOICE_FILE = os.path.join(SESSION_DIR, "provider_choice.txt")


def _print_banner():
    print()
    print("=" * 56)
    print("  🤖  AI FOREX TRADING — BROWSER PROVIDER SETUP WIZARD")
    print("=" * 56)
    print()
    print("Wizard ini akan membantu Anda login ke AI provider pilihan.")
    print("Setelah selesai, sistem akan berjalan HEADLESS secara otomatis.")
    print()


def _select_provider() -> str:
    """Prompt user to select a provider and persist the choice."""
    print("=" * 49)
    print("🤖 PILIH AI PROVIDER")
    print("=" * 49)
    print("1. ChatGPT (chat.openai.com)")
    print("2. Claude  (claude.ai)")
    print("=" * 49)

    while True:
        raw = input("Pilihan (1/2): ").strip()
        if raw == "1":
            provider = "chatgpt"
            break
        elif raw == "2":
            provider = "claude"
            break
        else:
            print("Masukkan 1 atau 2.")

    Path(SESSION_DIR).mkdir(parents=True, exist_ok=True)
    with open(PROVIDER_CHOICE_FILE, "w") as f:
        f.write(provider)

    print()
    print(f"✅ Provider dipilih: {provider.upper()}")
    return provider


async def _run_guided_login(provider: str):
    """Import BrowserProvider and trigger the guided login flow."""
    from src.llm.browser_provider import BrowserProvider

    # Pass provider explicitly — skip the auto-prompt since we already chose
    bp = BrowserProvider(provider=provider)

    print()
    print(f"🚀 Memulai guided login untuk {provider.upper()}...")
    await bp.guided_login()

    # Verify
    if await bp.is_logged_in():
        print()
        print("=" * 56)
        print("🎉 SETUP SELESAI!")
        print("=" * 56)
        print(f"✅ Session {provider.upper()} berhasil disimpan.")
        print()
        print("Sekarang Anda bisa menjalankan sistem utama.")
        print("Browser akan berjalan headless (tidak terlihat) secara otomatis.")
        print()
    else:
        print()
        print("⚠️  Login belum terkonfirmasi.")
        print("Silakan jalankan wizard ini lagi dan pastikan login berhasil")
        print("sebelum menekan Enter.")
        print()

    await bp.close()


def run_wizard():
    """Entry point for the setup wizard."""
    _print_banner()

    # Check if there's an existing session
    if os.path.exists(PROVIDER_CHOICE_FILE):
        with open(PROVIDER_CHOICE_FILE) as f:
            existing = f.read().strip()
        if existing in ("chatgpt", "claude"):
            print(f"ℹ️  Provider tersimpan: {existing.upper()}")
            reset = input("Reset dan pilih ulang? (y/N): ").strip().lower()
            if reset == "y":
                try:
                    os.remove(PROVIDER_CHOICE_FILE)
                    # Also remove old session so we re-login
                    session_file = os.path.join(SESSION_DIR, f"{existing}_session.json")
                    if os.path.exists(session_file):
                        os.remove(session_file)
                    print(f"🗑️  Session {existing.upper()} dihapus.\n")
                except Exception as e:
                    print(f"⚠️  Gagal menghapus session: {e}\n")
                provider = _select_provider()
            else:
                provider = existing
                # If session file is missing, still run login
                session_file = os.path.join(SESSION_DIR, f"{provider}_session.json")
                if not os.path.exists(session_file):
                    print(f"ℹ️  Session file tidak ditemukan. Memulai login ulang...")
                else:
                    print(f"ℹ️  Session sudah ada untuk {provider.upper()}.")
                    redo = input("Login ulang? (y/N): ").strip().lower()
                    if redo != "y":
                        print("Tidak ada perubahan. Keluar.")
                        return
    else:
        provider = _select_provider()

    asyncio.run(_run_guided_login(provider))


if __name__ == "__main__":
    run_wizard()
