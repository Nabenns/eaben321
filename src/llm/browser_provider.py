"""
Browser-based LLM Provider using Playwright.
Supports ChatGPT (chat.openai.com) and Claude (claude.ai) via browser automation.

No API key or email/password required — uses guided first-time manual login.

FLOW:
  First time (or session expired):
    1. User picks provider (ChatGPT or Claude)
    2. Browser opens (visible)
    3. User logs in manually — handles CAPTCHA, 2FA, magic link, anything
    4. User presses Enter in terminal
    5. Session saved to disk
    6. Browser closes & re-launches headless for the session

  Subsequent runs:
    1. Load saved session
    2. Run headless (invisible)
    3. If session expired → repeat guided login automatically
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Session storage ────────────────────────────────────────────────────────────
SESSION_DIR = os.environ.get("BROWSER_SESSION_DIR", "data/sessions")
PROVIDER_CHOICE_FILE = os.path.join(SESSION_DIR, "provider_choice.txt")

CHATGPT_SESSION_FILE = os.path.join(SESSION_DIR, "chatgpt_session.json")
CLAUDE_SESSION_FILE = os.path.join(SESSION_DIR, "claude_session.json")

CHATGPT_LOGIN_URL = "https://chat.openai.com/auth/login"
CLAUDE_LOGIN_URL = "https://claude.ai/login"
CHATGPT_CHAT_URL = "https://chat.openai.com"
CLAUDE_CHAT_URL = "https://claude.ai"

MAX_RETRIES = 3
RESPONSE_TIMEOUT_SEC = 120
POLL_INTERVAL = 1.5   # seconds between streaming polls
STABLE_ROUNDS = 2     # text must be unchanged this many polls to be "done"

# User-agent string for stealth
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Playwright launch args (common)
_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--disable-dev-shm-usage",
]

# JS snippet to hide automation markers
_STEALTH_SCRIPT = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"


# ── Helper: prompt provider selection ─────────────────────────────────────────

def _prompt_provider_choice() -> str:
    """Interactively ask user to pick a provider and persist the choice."""
    print()
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
    print(f"✅ Provider dipilih: {provider.upper()}")
    return provider


def _load_or_prompt_provider(override: Optional[str] = None) -> str:
    """
    Return provider name (chatgpt|claude).

    Priority:
      1. override argument (used when instantiated with explicit provider)
      2. saved choice file
      3. interactive menu
    """
    if override and override != "auto":
        return override

    if os.path.exists(PROVIDER_CHOICE_FILE):
        with open(PROVIDER_CHOICE_FILE) as f:
            choice = f.read().strip().lower()
        if choice in ("chatgpt", "claude"):
            return choice

    return _prompt_provider_choice()


# ── Main BrowserProvider class ─────────────────────────────────────────────────

class BrowserProvider:
    """
    Browser automation LLM provider with guided first-time login.

    provider: "chatgpt" | "claude" | "auto"
              "auto" reads from saved choice or prompts the user.
    """

    def __init__(self, provider: str = "auto"):
        # Resolve provider (may prompt user if "auto" and no saved choice)
        self.provider = _load_or_prompt_provider(provider)

        if self.provider == "chatgpt":
            self._login_url = CHATGPT_LOGIN_URL
            self._chat_url = CHATGPT_CHAT_URL
            self._session_file = CHATGPT_SESSION_FILE
        elif self.provider == "claude":
            self._login_url = CLAUDE_LOGIN_URL
            self._chat_url = CLAUDE_CHAT_URL
            self._session_file = CLAUDE_SESSION_FILE
        else:
            raise ValueError(
                f"Unknown browser provider '{self.provider}'. Use 'chatgpt', 'claude', or 'auto'."
            )

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self):
        """
        Launch browser and restore session if available.

        - If a saved session file exists: launch headless and verify login.
        - If no session (or session expired): run guided_login().
        """
        if self._initialized:
            return

        Path(SESSION_DIR).mkdir(parents=True, exist_ok=True)

        session_exists = Path(self._session_file).exists()

        if session_exists:
            logger.info(
                "[BrowserProvider] Found saved session for %s — launching headless.",
                self.provider,
            )
            await self._launch_headless(load_session=True)

            # Verify session is still valid
            await self._page.goto(self._chat_url, wait_until="networkidle")
            await asyncio.sleep(4)

            if await self.is_logged_in():
                logger.info("[BrowserProvider] Session valid — running headless.")
                self._initialized = True
                return
            else:
                logger.info("[BrowserProvider] Session expired — starting guided login.")
                await self._close_browser()
                await self.guided_login()
        else:
            logger.info(
                "[BrowserProvider] No saved session for %s — starting guided login.",
                self.provider,
            )
            await self.guided_login()

        self._initialized = True

    async def close(self):
        """Close browser gracefully."""
        await self._close_browser()
        self._initialized = False

    async def _close_browser(self):
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                logger.debug("[BrowserProvider] Browser close error: %s", e)
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.debug("[BrowserProvider] Playwright stop error: %s", e)
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None

    # ── Browser launch helpers ───────────────────────────────────────────────

    async def _launch_headless(self, load_session: bool = False):
        """Launch Chromium in headless mode, optionally loading a saved session."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=_LAUNCH_ARGS,
        )
        await self._create_context(load_session=load_session)

    async def _launch_visible(self, load_session: bool = False):
        """Launch Chromium in visible (headed) mode for manual login."""
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=False,
            args=_LAUNCH_ARGS,
        )
        await self._create_context(load_session=load_session)

    async def _create_context(self, load_session: bool = False):
        """Create browser context with stealth settings."""
        context_kwargs = {
            "viewport": {"width": 1280, "height": 900},
            "user_agent": _USER_AGENT,
        }
        if load_session and Path(self._session_file).exists():
            context_kwargs["storage_state"] = self._session_file

        self._context = await self._browser.new_context(**context_kwargs)
        self._page = await self._context.new_page()
        await self._page.add_init_script(_STEALTH_SCRIPT)

    # ── Guided login ─────────────────────────────────────────────────────────

    async def guided_login(self):
        """
        Open browser visibly, let user log in manually, then save session
        and re-launch headless for subsequent use.
        """
        provider_upper = self.provider.upper()

        print()
        print("=" * 50)
        print(f"🔐 LOGIN {provider_upper}")
        print("=" * 50)
        print(f"Browser akan dibuka. Silakan login di browser.")
        print(f"Setelah berhasil login, kembali ke sini dan tekan Enter.")
        print("=" * 50)
        print()

        # Open browser visibly
        await self._launch_visible()

        # Navigate to the login page
        await self._page.goto(self._login_url, wait_until="domcontentloaded")

        # Wait for user to finish logging in
        # Using asyncio.get_event_loop().run_in_executor so we don't block the loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: input("✅ Sudah login? Tekan Enter untuk melanjutkan..."),
        )

        # Quick verify before saving
        if not await self.is_logged_in():
            # Maybe they're on a different page — try navigating to chat
            await self._page.goto(self._chat_url, wait_until="domcontentloaded")
            await asyncio.sleep(2)

        # Save session
        await self._context.storage_state(path=self._session_file)
        print()
        print("💾 Session tersimpan! Sistem akan berjalan otomatis mulai sekarang.")
        print()

        # Close visible browser
        await self._close_browser()

        # Re-launch headless with the saved session
        logger.info("[BrowserProvider] Re-launching headless with saved session.")
        await self._launch_headless(load_session=True)
        await self._page.goto(self._chat_url, wait_until="networkidle")
        await asyncio.sleep(4)

    # ── Login check ──────────────────────────────────────────────────────────

    async def is_logged_in(self) -> bool:
        """Return True if the current page indicates an active session."""
        try:
            if self.provider == "chatgpt":
                return await self._chatgpt_is_logged_in()
            else:
                return await self._claude_is_logged_in()
        except Exception as e:
            logger.debug("[BrowserProvider] is_logged_in check error: %s", e)
            return False

    async def _chatgpt_is_logged_in(self) -> bool:
        url = self._page.url
        if "auth" in url or "login" in url:
            return False
        try:
            await self._page.wait_for_selector(
                'nav, textarea, #prompt-textarea, [data-testid="send-button"]',
                timeout=5000,
            )
            return True
        except Exception:
            return False

    async def _claude_is_logged_in(self) -> bool:
        url = self._page.url
        if "login" in url or "auth" in url:
            return False
        try:
            await self._page.wait_for_selector(
                '.ProseMirror, [data-testid="chat-input"], div[contenteditable="true"]',
                timeout=5000,
            )
            return True
        except Exception:
            return False

    # ── Conversation management ──────────────────────────────────────────────

    async def new_conversation(self):
        """Start a fresh conversation thread."""
        if self.provider == "chatgpt":
            await self._chatgpt_new_conversation()
        else:
            await self._claude_new_conversation()

    async def _chatgpt_new_conversation(self):
        try:
            btn = await self._page.query_selector(
                'a[href="/"], button[aria-label="New chat"], '
                'nav a:has-text("New chat"), [data-testid="create-new-chat-button"]'
            )
            if btn:
                await btn.click()
                await asyncio.sleep(1.5)
                return
        except Exception:
            pass
        # Fallback: navigate directly
        await self._page.goto(self._chat_url, wait_until="domcontentloaded")
        await asyncio.sleep(1.5)

    async def _claude_new_conversation(self):
        try:
            btn = await self._page.query_selector(
                'button[aria-label="New chat"], a:has-text("New chat"), '
                'button:has-text("New conversation")'
            )
            if btn:
                await btn.click()
                await asyncio.sleep(1.5)
                return
        except Exception:
            pass
        await self._page.goto(self._chat_url, wait_until="domcontentloaded")
        await asyncio.sleep(1.5)

    # ── Send message ─────────────────────────────────────────────────────────

    async def send_message(self, prompt: str) -> str:
        """
        Send a prompt and return the full response text.
        Retries up to MAX_RETRIES on transient errors.
        Automatically re-runs guided login if session has expired.
        """
        if not self._initialized:
            await self.initialize()

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Check session validity before each attempt
                if not await self.is_logged_in():
                    logger.info(
                        "[BrowserProvider] Session expired on attempt %d — re-login.", attempt
                    )
                    await self._close_browser()
                    await self.guided_login()

                if self.provider == "chatgpt":
                    return await self._chatgpt_send_message(prompt)
                else:
                    return await self._claude_send_message(prompt)

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[BrowserProvider] Attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(5 * attempt)
                    try:
                        await self._page.reload(wait_until="domcontentloaded")
                        await asyncio.sleep(2)
                    except Exception:
                        pass

        raise RuntimeError(
            f"[BrowserProvider] All {MAX_RETRIES} attempts failed. Last error: {last_error}"
        )

    # ── ChatGPT messaging ────────────────────────────────────────────────────

    async def _chatgpt_send_message(self, prompt: str) -> str:
        page = self._page

        # Ensure we're on the chat page
        if "chat.openai.com" not in page.url:
            await page.goto(self._chat_url, wait_until="domcontentloaded")
            await asyncio.sleep(2)

        # Wait for page to fully load first
        await asyncio.sleep(3)

        # Find input
        textarea_selector = '#prompt-textarea, textarea[data-id="root"], textarea'
        await page.wait_for_selector(textarea_selector, timeout=30000, state="attached")

        textarea = await page.query_selector(textarea_selector)
        if not textarea:
            raise RuntimeError("ChatGPT: input textarea not found")

        # Scroll into view and wait for it to be interactable
        await textarea.scroll_into_view_if_needed()
        await asyncio.sleep(0.5)
        await textarea.click()
        await asyncio.sleep(0.4)

        # Clear existing content and type the prompt
        await textarea.fill("")
        await page.keyboard.type(prompt, delay=30)
        await asyncio.sleep(0.5)

        # Click send or press Enter
        send_sent = False
        for send_sel in [
            'button[data-testid="send-button"]',
            'button[aria-label="Send message"]',
            'button[aria-label="Send prompt"]',
        ]:
            try:
                btn = await page.query_selector(send_sel)
                if btn and await btn.is_enabled():
                    await btn.click()
                    send_sent = True
                    break
            except Exception:
                pass
        if not send_sent:
            await page.keyboard.press("Enter")

        await asyncio.sleep(1.5)

        # Wait for streaming to start (stop-generating button appears)
        try:
            await page.wait_for_selector(
                'button[aria-label="Stop generating"], button[data-testid="stop-button"]',
                timeout=15000,
            )
        except Exception:
            logger.debug("[BrowserProvider] ChatGPT: stop button not detected — response may be instant.")

        # Wait for streaming to finish
        await self._wait_chatgpt_done(page)

        return await self._extract_chatgpt_response(page)

    async def _wait_chatgpt_done(self, page):
        """Poll until the 'Stop generating' button disappears."""
        start = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start > RESPONSE_TIMEOUT_SEC:
                logger.warning("[BrowserProvider] ChatGPT response timeout.")
                break
            stop = await page.query_selector(
                'button[aria-label="Stop generating"], button[data-testid="stop-button"]'
            )
            if not stop:
                break
            await asyncio.sleep(POLL_INTERVAL)
        await asyncio.sleep(1.0)   # settling

    async def _extract_chatgpt_response(self, page) -> str:
        """Extract the last assistant message from ChatGPT."""
        selectors = [
            '[data-message-author-role="assistant"]:last-of-type .markdown',
            '[data-message-author-role="assistant"]:last-of-type',
            '.group.w-full:last-child .markdown',
            'article:last-of-type .prose',
        ]
        for sel in selectors:
            try:
                els = await page.query_selector_all(sel)
                if els:
                    text = await els[-1].inner_text()
                    if text and text.strip():
                        return text.strip()
            except Exception:
                continue

        # JS fallback
        text = await page.evaluate("""
            () => {
                const msgs = document.querySelectorAll('[data-message-author-role="assistant"]');
                if (!msgs.length) return null;
                return msgs[msgs.length - 1].innerText;
            }
        """)
        if text:
            return text.strip()

        raise RuntimeError("ChatGPT: could not extract response text from page")

    # ── Claude messaging ─────────────────────────────────────────────────────

    async def _claude_send_message(self, prompt: str) -> str:
        page = self._page

        # Ensure we're on the chat page
        if "claude.ai" not in page.url:
            await page.goto(self._chat_url, wait_until="domcontentloaded")
            await asyncio.sleep(2)

        # Find input (ProseMirror contenteditable div)
        input_selector = '.ProseMirror, [data-testid="chat-input"], div[contenteditable="true"]'
        await page.wait_for_selector(input_selector, timeout=15000)

        input_el = await page.query_selector(input_selector)
        if not input_el:
            raise RuntimeError("Claude: input field not found")

        await input_el.click()
        await asyncio.sleep(0.4)

        # Type the prompt
        await page.keyboard.type(prompt, delay=30)
        await asyncio.sleep(0.5)

        # Click send
        send_sent = False
        for send_sel in [
            'button[aria-label="Send message"]',
            'button[data-testid="send-button"]',
            'button[type="submit"]',
        ]:
            try:
                btn = await page.query_selector(send_sel)
                if btn and await btn.is_enabled():
                    await btn.click()
                    send_sent = True
                    break
            except Exception:
                pass
        if not send_sent:
            await page.keyboard.press("Enter")

        await asyncio.sleep(1.5)

        # Wait for response to appear and finish streaming
        await self._wait_claude_done(page)

        return await self._extract_claude_response(page)

    async def _wait_claude_done(self, page):
        """Wait for Claude streaming to complete using text-stability polling."""
        start = asyncio.get_event_loop().time()

        # Wait for response to start
        try:
            await page.wait_for_selector(
                '.font-claude-message, [data-testid="assistant-message"], article',
                timeout=15000,
            )
        except Exception:
            logger.debug("[BrowserProvider] Claude: response element not visible immediately.")

        prev_text = ""
        stable_count = 0

        while True:
            if asyncio.get_event_loop().time() - start > RESPONSE_TIMEOUT_SEC:
                logger.warning("[BrowserProvider] Claude response timeout.")
                break

            # If a stop/loading button is visible, streaming is still in progress
            stop = await page.query_selector('button[aria-label="Stop"], button:has-text("Stop")')
            if stop:
                stable_count = 0
                await asyncio.sleep(POLL_INTERVAL)
                continue

            # Check text stability
            try:
                current = await self._extract_claude_response(page)
                if current and current == prev_text:
                    stable_count += 1
                    if stable_count >= STABLE_ROUNDS:
                        break
                else:
                    stable_count = 0
                    prev_text = current
            except Exception:
                pass

            await asyncio.sleep(POLL_INTERVAL)

        await asyncio.sleep(1.0)   # settling

    async def _extract_claude_response(self, page) -> str:
        """Extract the last assistant message from Claude."""
        selectors = [
            ".font-claude-message",
            '[data-testid="assistant-message"]',
            ".assistant-message",
            'article[data-author="claude"]',
        ]
        for sel in selectors:
            try:
                els = await page.query_selector_all(sel)
                if els:
                    text = await els[-1].inner_text()
                    if text and text.strip():
                        return text.strip()
            except Exception:
                continue

        # JS fallback
        text = await page.evaluate("""
            () => {
                const sels = ['.font-claude-message', '[data-testid="assistant-message"]',
                              '.assistant-message'];
                for (const s of sels) {
                    const els = document.querySelectorAll(s);
                    if (els.length) return els[els.length - 1].innerText;
                }
                return null;
            }
        """)
        if text:
            return text.strip()

        raise RuntimeError("Claude: could not extract response text from page")

    # ── Sync wrappers ─────────────────────────────────────────────────────────

    def send_message_sync(self, prompt: str) -> str:
        """Synchronous convenience wrapper."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._async_send_with_init(prompt))
        finally:
            loop.close()

    async def _async_send_with_init(self, prompt: str) -> str:
        await self.initialize()
        return await self.send_message(prompt)


# ── Sync context-manager wrapper ───────────────────────────────────────────────

class SyncBrowserProvider:
    """
    Thin synchronous wrapper over BrowserProvider.

    Usage:
        with SyncBrowserProvider("chatgpt") as bp:
            response = bp.send_message("Hello!")
    """

    def __init__(self, provider: str = "auto"):
        self._provider = BrowserProvider(provider)
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._provider.initialize())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._provider.close())
        self._loop.close()

    def send_message(self, prompt: str) -> str:
        return self._loop.run_until_complete(self._provider.send_message(prompt))

    def new_conversation(self):
        self._loop.run_until_complete(self._provider.new_conversation())
