import os
import json
import logging
import time
import requests
from typing import Dict, Any, Optional

class LLMSupervisor:
    """
    The Strategic Layer (CEO) of the Financial Platform.
    Calls Google Gemini REST API directly (no SDK needed) to avoid
    dependency conflicts with alpaca-trade-api.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            self.logger.warning("LLMSupervisor disabled. Missing GEMINI_API_KEY.")
        else:
            self.logger.info("🧠 LLM Supervisor (Google Gemini REST) Online")
            
    def analyze_market_context(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Takes the current context and generates a strategic response
        by calling the Gemini REST API directly.
        """
        if not self.enabled:
            return None
            
        system_prompt = (
            "You are the Chief Investment Officer (CIO) of a quantitative trading firm.\n"
            "Your job is to read the macro market context, the current algorithmic regime, "
            "and portfolio performance, and decide if the tactical trading algorithms (RL) "
            "need their risk limits adjusted.\n\n"
            "You MUST return ONLY a valid JSON object with strictly these keys:\n"
            "- \"reasoning\": A 1-2 sentence explanation of your market read.\n"
            "- \"executive_summary\": A short message to the user summarizing your view.\n"
            "- \"max_leverage\": Float (e.g., 0.5 to 5.0). Lower this if risk is high.\n"
            "- \"max_position_size_pct\": Float (e.g., 0.05 to 0.20). Lower this to reduce exposure.\n"
            "- \"emergency_stop\": Boolean (true/false). Set to true ONLY if a catastrophic crash is evident.\n\n"
            "If the market is normal, keep constraints loose (e.g., leverage: 2.0-5.0, pos_size: 0.15). "
            "If news is severely bearish or the portfolio is burning, tighten constraints drastically.\n"
            "Return ONLY valid JSON. No markdown, no code fences."
        )
        
        user_prompt = f"Current Market Context:\n{json.dumps(context, indent=2)}\n\nWhat are your new constraints?"
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 500,
                "responseMimeType": "application/json"
            }
        }
        
        for attempt in range(2):
            try:
                response = requests.post(url, json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    result = json.loads(text)
                    self.logger.info(f"LLM Supervisor Assessment: {result.get('reasoning')}")
                    return result
                elif response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    self.logger.warning(f"Gemini quota hit (attempt {attempt+1}/2). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Gemini API error {response.status_code}: {response.text[:200]}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"LLM Supervisor failed: {e}")
                return None
        
        self.logger.warning("LLM Supervisor: Quota exhausted. Skipping this cycle (non-critical).")
        return None

    def _call_gemini_raw(self, prompt: str) -> str:
        """
        Calls Gemini to get a raw text/markdown response instead of JSON.
        Used for generating the nightly journal.
        """
        if not self.enabled:
            return "LLM Supervisor disabled (No API Key). Cannot generate text."
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1000,
                "responseMimeType": "text/plain"
            }
        }
        
        for attempt in range(2):
            try:
                response = requests.post(url, json=payload, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                elif response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    return f"Error: Gemini API returned {response.status_code}"
            except Exception as e:
                return f"Error communicating with Gemini: {e}"
                
        return "Error: Gemini quota exhausted."
