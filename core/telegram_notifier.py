import os
import requests
import logging
from typing import Optional

class TelegramNotifier:
    """
    Handles broadcasting active updates to a Telegram Chat via a Bot Token.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            self.logger.warning("TelegramNotifier disabled. TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.")
        else:
            self.logger.info("📱 Telegram Notifier Online")
            
    def _send_message(self, text: str) -> bool:
        """Sends a raw text message to the configured Telegram chat."""
        if not self.enabled:
            return False
            
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        
        try:
            # Short timeout so it doesn't block the main trading thread
            response = requests.post(url, json=payload, timeout=3.0)
            if response.status_code != 200:
                self.logger.error(f"Telegram API Error: {response.text}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Telegram connection failed: {e}")
            return False

    def send_trade_alert(self, symbol: str, action: str, quantity: float, price: float, strategy: str = "Unknown", reason: str = ""):
        """Broadcasts a trade execution alert."""
        if not self.enabled: return
        
        emoji = "🟢" if action.upper() == "BUY" else "🔴"
        text = (
            f"<b>{emoji} TRADE EXECUTED</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Action:</b> {action} {quantity}\n"
            f"<b>Price:</b> ${price:.2f}\n"
            f"<b>Strategy:</b> {strategy}\n"
            f"<b>Reason:</b> {reason}"
        )
        self._send_message(text)

    def send_regime_shift(self, new_regime: str):
        """Broadcasts a macro regime shift."""
        if not self.enabled: return
        
        text = (
            f"<b>🚨 MARKET REGIME SHIFT DETECTED 🚨</b>\n\n"
            f"Orchestrator has shifted to: <b>{new_regime}</b>\n"
            f"Strategy presets are being dynamically reloaded."
        )
        self._send_message(text)
        
    def send_error_alert(self, component: str, error_msg: str):
        """Broadcasts a critical system error."""
        if not self.enabled: return
        
        text = (
            f"<b>⚠️ SYSTEM ALERT ⚠️</b>\n\n"
            f"<b>Component:</b> {component}\n"
            f"<b>Error:</b>\n<pre>{error_msg}</pre>"
        )
        self._send_message(text)
