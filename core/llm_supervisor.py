import os
import json
import logging
import requests
from typing import Dict, Any, Optional

class LLMSupervisor:
    """
    The Strategic Layer (CIO) of the Financial Platform.
    Now powered by the local Sentience Core model.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.server_url = os.getenv("SENTIENCE_SERVER_URL", "http://localhost:8000/v1/chat/completions")
        self.enabled = self._check_server_health()
        
        if self.enabled:
            self.logger.info(f"🧠 Sentience Core CONNECTED at {self.server_url}")
        else:
            self.logger.warning("🧠 Sentience Core DISCONNECTED (Model server not reachable)")

    def _check_server_health(self) -> bool:
        """Check if the model server is available."""
        try:
            # Simple check to see if we can reach the server
            response = requests.get(self.server_url.replace("/v1/chat/completions", "/docs"), timeout=2)
            return response.status_code == 200
        except:
            return False

    def analyze_market_context(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Takes the current context and calculates strategic decisions 
        using the local Sentience Core model.
        """
        if not self.enabled:
            # Re-check health just in case it was started late
            if self._check_server_health():
                self.enabled = True
            else:
                return None
        
        return self._call_model(context)

    def _call_model(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call the local inference server for a structured response."""
        prompt = f"Analyze this market context and provide an executive summary, max leverage (1.0-5.0), max position size (0.05-0.20), and emergency stop status: {json.dumps(context)}"
        
        try:
            response_text = self._call_model_raw(prompt)
            # Find the JSON part of the response if the model outputs text + JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return None
        except Exception as e:
            self.logger.error(f"LLM Supervisor Inference Error: {e}")
            return None

    def _call_model_raw(self, prompt: str) -> str:
        """Helper to call the raw chat completions endpoint."""
        payload = {
            "messages": [
                {"role": "system", "content": "You are the CIO of a quantitative trading firm. Follow SOUL.md rules strictly."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.server_url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: Server returned {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
