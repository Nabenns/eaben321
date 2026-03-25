"""
Browser-based LLM Provider using Playwright.
Supports ChatGPT (chat.openai.com) and Claude (claude.ai) via browser automation.
No API key required — uses web UI login with email/password.
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Session storage paths
SESSION_DIR = os.environ.get("BROWSER_SESSION_DIR", "data/sessions")
CHATGPT_SESSION_FILE = os.path.join(SESSION_DIR, "chatgpt_session.json")
CLAUDE_SESSION_FILE = os.path.join(SESSION_DIR, "claude_session.json")

# Headless mode (default False so user can intervene for CAPTCHA)
BROWSER_HEADLESS = os.environ.get("BROWSER_HEADLESS", "false").lower() == "true"

# Credentials
CHATGPT_EMAIL = os.environ.get("CHATGPT_EMAIL", "")
CHATGPT_PASSWORD = os.environ.get("CHATGPT_PASSWORD", "")
CLAUDE_EMAIL = os.environ.get("CLAUDE_EMAIL", "")
CLAUDE_PASSWORD = os.environ.get("CLAUDE_PASSWORD", "")

MAX_RETRIES = 3
RESPONSE_TIMEOUT_MS = 120_000   # 2 minutes max wait for LLM response
POLL_INTERVAL = 1.5             # seconds between polling checks


async def _human_type(page, selector: str, text: str, delay_range=(40, 120)):
    """Type text with random per-character delay to simulate human typing."""
    await page.click(selector)
    for char in text:
        await page.keyboard.type(char)
        await asyncio.sleep(random.randint(*delay_range) / 1000)


async def _random_delay(min_ms: float = 500, max_ms: float = 1500):
    """Small random pause to avoid bot detection."""
    await asyncio.sleep(random.uniform(min_ms, max_ms) / 1000)


class BrowserProvider:
    """
    Browser automation LLM provider.

    provider: "chatgpt" | "claude"
    """

    def __init__(self, provider: str = "chatgpt"):
        if provider not in ("chatgpt", "claude"):
            raise ValueError(f"Unknown browser provider '{provider}'. Use 'chatgpt' or 'claude'.")
        self.provider = provider
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._initialized = False

        # Set provider-specific values
        if provider == "chatgpt":
            self._url = "https://chat.openai.com"
            self._session_file = CHATGPT_SESSION_FILE
            self._email = CHATGPT_EMAIL
            self._password = CHATGPT_PASSWORD
        else:
            self._url = "https://claude.ai"
            self._session_file = CLAUDE_SESSION_FILE
            self._email = CLAUDE_EMAIL
            self._password = CLAUDE_PASSWORD

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """Launch browser and restore session if available."""
        if self._initialized:
            return

        from playwright.async_api import async_playwright

        Path(SESSION_DIR).mkdir(parents=True, exist_ok=True)

        self._playwright = await async_playwright().start()

        launch_kwargs = {
            "headless": BROWSER_HEADLESS,
            "args": [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        }

        self._browser = await self._playwright.chromium.launch(**launch_kwargs)

        # Load saved session if it exists
        context_kwargs = {
            "viewport": {"width": 1280, "height": 900},
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        }
        if Path(self._session_file).exists():
            context_kwargs["storage_state"] = self._session_file
            logger.info("[BrowserProvider] Loaded saved session from %s", self._session_file)

        self._context = await self._browser.new_context(**context_kwargs)
        self._page = await self._context.new_page()

        # Inject stealth: hide navigator.webdriver
        await self._page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        self._initialized = True

        # Navigate to the site
        await self._page.goto(self._url, wait_until="domcontentloaded")
        await _random_delay(1000, 2000)

        # Login if needed
        if not await self.is_logged_in():
            logger.info("[BrowserProvider] Not logged in — attempting login.")
            await self.login()
        else:
            logger.info("[BrowserProvider] Session restored — already logged in.")

    async def close(self):
        """Save session and close browser."""
        if self._context:
            try:
                await self._context.storage_state(path=self._session_file)
                logger.info("[BrowserProvider] Session saved to %s", self._session_file)
            except Exception as e:
                logger.warning("[BrowserProvider] Could not save session: %s", e)
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._initialized = False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def is_logged_in(self) -> bool:
        """Check if the browser is currently authenticated."""
        try:
            if self.provider == "chatgpt":
                return await self._chatgpt_is_logged_in()
            else:
                return await self._claude_is_logged_in()
        except Exception as e:
            logger.debug("[BrowserProvider] is_logged_in check failed: %s", e)
            return False

    async def _chatgpt_is_logged_in(self) -> bool:
        """Check ChatGPT login state by looking for the textarea or sidebar."""
        try:
            # If we see the new-chat button or the message textarea, we're in
            await self._page.wait_for_selector(
                'nav, textarea, [data-testid="send-button"]',
                timeout=5000
            )
            # Verify we're not on a login page
            current_url = self._page.url
            if "auth" in current_url or "login" in current_url:
                return False
            return True
        except Exception:
            return False

    async def _claude_is_logged_in(self) -> bool:
        """Check Claude login state."""
        try:
            await self._page.wait_for_selector(
                '[data-testid="chat-input"], .ProseMirror, button[aria-label="New chat"]',
                timeout=5000
            )
            current_url = self._page.url
            if "login" in current_url or "auth" in current_url:
                return False
            return True
        except Exception:
            return False

    async def login(self):
        """Perform login flow for the configured provider."""
        if self.provider == "chatgpt":
            await self._chatgpt_login()
        else:
            await self._claude_login()

        # Save session after login
        try:
            await self._context.storage_state(path=self._session_file)
            logger.info("[BrowserProvider] Session saved after login.")
        except Exception as e:
            logger.warning("[BrowserProvider] Could not save session after login: %s", e)

    async def _chatgpt_login(self):
        """ChatGPT email + password login flow."""
        if not self._email or not self._password:
            raise RuntimeError("CHATGPT_EMAIL and CHATGPT_PASSWORD must be set in .env")

        logger.info("[BrowserProvider] Logging into ChatGPT...")

        # Navigate to login
        await self._page.goto("https://chat.openai.com/auth/login", wait_until="domcontentloaded")
        await _random_delay(1500, 2500)

        # Click "Log in" button
        try:
            await self._page.click('button:has-text("Log in")', timeout=10000)
            await _random_delay(1000, 2000)
        except Exception:
            logger.debug("[BrowserProvider] No 'Log in' button found, proceeding...")

        # Enter email
        try:
            await self._page.wait_for_selector('input[name="username"], input[type="email"]', timeout=15000)
            await _human_type(self._page, 'input[name="username"], input[type="email"]', self._email)
            await _random_delay(500, 1000)
        except Exception as e:
            raise RuntimeError(f"ChatGPT login: could not find email field: {e}")

        # Click Continue
        try:
            await self._page.click('button[type="submit"], button:has-text("Continue")', timeout=5000)
            await _random_delay(1500, 2500)
        except Exception:
            await self._page.keyboard.press("Enter")
            await _random_delay(1500, 2500)

        # Enter password
        try:
            await self._page.wait_for_selector('input[type="password"]', timeout=15000)
            await _human_type(self._page, 'input[type="password"]', self._password)
            await _random_delay(500, 1000)
        except Exception as e:
            raise RuntimeError(f"ChatGPT login: could not find password field: {e}")

        # Submit
        try:
            await self._page.click('button[type="submit"], button:has-text("Continue")', timeout=5000)
        except Exception:
            await self._page.keyboard.press("Enter")

        # Wait for redirect to chat
        try:
            await self._page.wait_for_url("**/chat.openai.com/**", timeout=30000)
        except Exception:
            pass

        await _random_delay(2000, 3000)

        # Handle any post-login popups/modals
        for _ in range(3):
            try:
                dismiss = await self._page.query_selector(
                    'button:has-text("OK"), button:has-text("Got it"), button:has-text("Dismiss")'
                )
                if dismiss:
                    await dismiss.click()
                    await _random_delay(500, 1000)
                else:
                    break
            except Exception:
                break

        if not await self.is_logged_in():
            raise RuntimeError(
                "ChatGPT login failed. Please check credentials or handle CAPTCHA manually "
                "(browser is running with headless=False)."
            )
        logger.info("[BrowserProvider] ChatGPT login successful.")

    async def _claude_login(self):
        """Claude email + password / magic link login flow."""
        if not self._email or not self._password:
            raise RuntimeError("CLAUDE_EMAIL and CLAUDE_PASSWORD must be set in .env")

        logger.info("[BrowserProvider] Logging into Claude...")

        await self._page.goto("https://claude.ai/login", wait_until="domcontentloaded")
        await _random_delay(1500, 2500)

        # Enter email
        try:
            await self._page.wait_for_selector(
                'input[type="email"], input[name="email"]',
                timeout=15000
            )
            await _human_type(self._page, 'input[type="email"], input[name="email"]', self._email)
            await _random_delay(500, 1000)
        except Exception as e:
            raise RuntimeError(f"Claude login: could not find email field: {e}")

        # Click Continue
        try:
            await self._page.click(
                'button[type="submit"], button:has-text("Continue"), button:has-text("Sign in")',
                timeout=5000
            )
            await _random_delay(1500, 2500)
        except Exception:
            await self._page.keyboard.press("Enter")
            await _random_delay(1500, 2500)

        # Check if password field appeared (vs magic link)
        try:
            password_field = await self._page.wait_for_selector(
                'input[type="password"]',
                timeout=5000
            )
            if password_field:
                await _human_type(self._page, 'input[type="password"]', self._password)
                await _random_delay(500, 1000)
                try:
                    await self._page.click(
                        'button[type="submit"], button:has-text("Sign in"), button:has-text("Continue")',
                        timeout=5000
                    )
                except Exception:
                    await self._page.keyboard.press("Enter")
        except Exception:
            # Magic link flow — inform user
            logger.warning(
                "[BrowserProvider] Claude may have sent a magic link to %s. "
                "Please check your email and click the link in the browser window.",
                self._email
            )
            # Wait up to 2 minutes for user to click the magic link
            try:
                await self._page.wait_for_url("**/claude.ai/**", timeout=120000)
            except Exception:
                pass

        await _random_delay(2000, 3000)

        if not await self.is_logged_in():
            raise RuntimeError(
                "Claude login failed. Please check credentials, handle magic link, "
                "or resolve CAPTCHA in the browser window."
            )
        logger.info("[BrowserProvider] Claude login successful.")

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    async def new_conversation(self):
        """Start a fresh conversation thread."""
        if self.provider == "chatgpt":
            await self._chatgpt_new_conversation()
        else:
            await self._claude_new_conversation()

    async def _chatgpt_new_conversation(self):
        """Click 'New chat' in ChatGPT sidebar."""
        try:
            # Try the sidebar button
            new_chat_btn = await self._page.query_selector(
                'a[href="/"], button[aria-label="New chat"], '
                'nav a:has-text("New chat"), [data-testid="create-new-chat-button"]'
            )
            if new_chat_btn:
                await new_chat_btn.click()
                await _random_delay(1000, 1500)
            else:
                # Fall back: navigate directly
                await self._page.goto(self._url, wait_until="domcontentloaded")
                await _random_delay(1000, 1500)
        except Exception as e:
            logger.warning("[BrowserProvider] Could not start new ChatGPT conversation: %s", e)
            await self._page.goto(self._url, wait_until="domcontentloaded")

    async def _claude_new_conversation(self):
        """Click 'New chat' in Claude sidebar."""
        try:
            new_chat_btn = await self._page.query_selector(
                'button[aria-label="New chat"], a:has-text("New chat"), '
                'button:has-text("New conversation")'
            )
            if new_chat_btn:
                await new_chat_btn.click()
                await _random_delay(1000, 1500)
            else:
                await self._page.goto(self._url, wait_until="domcontentloaded")
                await _random_delay(1000, 1500)
        except Exception as e:
            logger.warning("[BrowserProvider] Could not start new Claude conversation: %s", e)
            await self._page.goto(self._url, wait_until="domcontentloaded")

    # ------------------------------------------------------------------
    # Sending messages & getting responses
    # ------------------------------------------------------------------

    async def send_message(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the full response text.
        Retries up to MAX_RETRIES on transient errors.
        """
        if not self._initialized:
            await self.initialize()

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Re-check login each attempt
                if not await self.is_logged_in():
                    logger.info("[BrowserProvider] Session expired — re-logging in.")
                    await self.login()

                if self.provider == "chatgpt":
                    return await self._chatgpt_send_message(prompt)
                else:
                    return await self._claude_send_message(prompt)

            except Exception as e:
                last_error = e
                logger.warning(
                    "[BrowserProvider] Attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, e
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(5 * attempt)   # back-off
                    # Try refreshing the page before retry
                    try:
                        await self._page.reload(wait_until="domcontentloaded")
                        await _random_delay(2000, 3000)
                    except Exception:
                        pass

        raise RuntimeError(
            f"[BrowserProvider] All {MAX_RETRIES} attempts failed. Last error: {last_error}"
        )

    async def _chatgpt_send_message(self, prompt: str) -> str:
        """Type and send a message in ChatGPT, then extract the response."""
        page = self._page

        # Find the input textarea
        textarea_selector = 'textarea, [contenteditable="true"], #prompt-textarea'
        await page.wait_for_selector(textarea_selector, timeout=15000)

        # Fill the textarea
        textarea = await page.query_selector(textarea_selector)
        if not textarea:
            raise RuntimeError("ChatGPT: textarea not found")

        await textarea.click()
        await _random_delay(300, 600)

        # Use fill for speed, then simulate Enter
        await textarea.fill(prompt)
        await _random_delay(400, 800)

        # Send: click send button or press Enter
        try:
            send_btn = await page.query_selector(
                'button[data-testid="send-button"], button[aria-label="Send message"]'
            )
            if send_btn and await send_btn.is_enabled():
                await send_btn.click()
            else:
                await page.keyboard.press("Enter")
        except Exception:
            await page.keyboard.press("Enter")

        await _random_delay(1000, 2000)

        # Wait for response to start (stop-generating button appears)
        try:
            await page.wait_for_selector(
                'button[aria-label="Stop generating"], button[data-testid="stop-button"]',
                timeout=15000
            )
        except Exception:
            logger.debug("[BrowserProvider] Stop button not detected — response may be instant.")

        # Wait for response to finish (stop button disappears)
        await self._wait_for_chatgpt_response_complete(page)

        # Extract last assistant message
        return await self._extract_chatgpt_response(page)

    async def _wait_for_chatgpt_response_complete(self, page):
        """Poll until the 'Stop generating' button is gone."""
        start = asyncio.get_event_loop().time()
        timeout = RESPONSE_TIMEOUT_MS / 1000

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                logger.warning("[BrowserProvider] ChatGPT response timeout after %.0fs", elapsed)
                break

            stop_btn = await page.query_selector(
                'button[aria-label="Stop generating"], button[data-testid="stop-button"]'
            )
            if not stop_btn:
                # Button gone → response complete
                break

            await asyncio.sleep(POLL_INTERVAL)

        # Extra settling delay
        await _random_delay(800, 1200)

    async def _extract_chatgpt_response(self, page) -> str:
        """Extract the last assistant message from ChatGPT."""
        # Try multiple selectors for different ChatGPT UI versions
        selectors = [
            '[data-message-author-role="assistant"]:last-of-type .markdown',
            '[data-message-author-role="assistant"]:last-of-type',
            '.group.w-full:last-child .markdown',
            'article:last-of-type .prose',
            '.message:last-child',
        ]
        for sel in selectors:
            try:
                elements = await page.query_selector_all(sel)
                if elements:
                    last = elements[-1]
                    text = await last.inner_text()
                    if text and text.strip():
                        return text.strip()
            except Exception:
                continue

        # Fallback: JS evaluation to get last message
        try:
            text = await page.evaluate("""
                () => {
                    const msgs = document.querySelectorAll('[data-message-author-role="assistant"]');
                    if (msgs.length === 0) return null;
                    return msgs[msgs.length - 1].innerText;
                }
            """)
            if text:
                return text.strip()
        except Exception:
            pass

        raise RuntimeError("ChatGPT: could not extract response text from page")

    async def _claude_send_message(self, prompt: str) -> str:
        """Type and send a message in Claude, then extract the response."""
        page = self._page

        # Claude uses a contenteditable ProseMirror div
        input_selector = (
            '.ProseMirror, [data-testid="chat-input"], '
            'div[contenteditable="true"], textarea'
        )
        await page.wait_for_selector(input_selector, timeout=15000)

        # Click the input area
        input_el = await page.query_selector(input_selector)
        if not input_el:
            raise RuntimeError("Claude: input field not found")

        await input_el.click()
        await _random_delay(300, 600)

        # Type the prompt
        # For ProseMirror, use keyboard type for best compatibility
        await page.keyboard.type(prompt, delay=random.randint(30, 80))
        await _random_delay(400, 800)

        # Send: click send button or Shift+Enter
        try:
            send_btn = await page.query_selector(
                'button[aria-label="Send message"], button[data-testid="send-button"], '
                'button[type="submit"]'
            )
            if send_btn and await send_btn.is_enabled():
                await send_btn.click()
            else:
                await page.keyboard.press("Enter")
        except Exception:
            await page.keyboard.press("Enter")

        await _random_delay(1000, 2000)

        # Wait for response to finish streaming
        await self._wait_for_claude_response_complete(page)

        # Extract last assistant message
        return await self._extract_claude_response(page)

    async def _wait_for_claude_response_complete(self, page):
        """Wait for Claude response to finish streaming."""
        start = asyncio.get_event_loop().time()
        timeout = RESPONSE_TIMEOUT_MS / 1000

        # Wait for response to start appearing
        try:
            await page.wait_for_selector(
                '.font-claude-message, [data-testid="assistant-message"], '
                '.assistant-message, article',
                timeout=15000
            )
        except Exception:
            logger.debug("[BrowserProvider] Claude: response element not detected immediately.")

        # Poll for streaming completion by checking if a "stop" indicator is gone
        # or if the response div has stabilized
        prev_text = ""
        stable_count = 0
        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                logger.warning("[BrowserProvider] Claude response timeout after %.0fs", elapsed)
                break

            # Check for stop/loading indicators
            stop_btn = await page.query_selector(
                'button[aria-label="Stop"], button:has-text("Stop")'
            )
            if stop_btn:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            # Text stability check: if text hasn't changed for 2 checks, we're done
            try:
                current_text = await self._extract_claude_response(page)
                if current_text == prev_text and current_text:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0
                    prev_text = current_text
            except Exception:
                pass

            await asyncio.sleep(POLL_INTERVAL)

        await _random_delay(800, 1200)

    async def _extract_claude_response(self, page) -> str:
        """Extract the last assistant message from Claude."""
        selectors = [
            '.font-claude-message',
            '[data-testid="assistant-message"]',
            '.assistant-message',
            'article[data-author="claude"]',
            '.message-content:last-child',
        ]
        for sel in selectors:
            try:
                elements = await page.query_selector_all(sel)
                if elements:
                    last = elements[-1]
                    text = await last.inner_text()
                    if text and text.strip():
                        return text.strip()
            except Exception:
                continue

        # Fallback: JS evaluation
        try:
            text = await page.evaluate("""
                () => {
                    // Claude uses various class names — try common ones
                    const selectors = [
                        '.font-claude-message',
                        '[data-testid="assistant-message"]',
                        '.assistant-message'
                    ];
                    for (const sel of selectors) {
                        const els = document.querySelectorAll(sel);
                        if (els.length > 0) {
                            return els[els.length - 1].innerText;
                        }
                    }
                    return null;
                }
            """)
            if text:
                return text.strip()
        except Exception:
            pass

        raise RuntimeError("Claude: could not extract response text from page")

    # ------------------------------------------------------------------
    # Synchronous wrapper (for non-async callers)
    # ------------------------------------------------------------------

    def send_message_sync(self, prompt: str) -> str:
        """Synchronous wrapper around send_message for use in non-async code."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._async_send_with_init(prompt))
        finally:
            loop.close()

    async def _async_send_with_init(self, prompt: str) -> str:
        await self.initialize()
        return await self.send_message(prompt)


# ---------------------------------------------------------------------------
# Convenience: sync context manager for use in sync code
# ---------------------------------------------------------------------------

class SyncBrowserProvider:
    """
    Thin synchronous wrapper over BrowserProvider.
    Usage:
        with SyncBrowserProvider("chatgpt") as bp:
            response = bp.send_message("Hello!")
    """
    def __init__(self, provider: str = "chatgpt"):
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
