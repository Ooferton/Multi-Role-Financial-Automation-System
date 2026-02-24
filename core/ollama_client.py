"""
Ollama LLM Client for the Financial AI.

Connects to the local Ollama server and provides tool-calling
capabilities so the LLM can control all financial agents.
"""
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

OLLAMA_BASE = "http://127.0.0.1:11434"

@dataclass
class Tool:
    """A callable tool the LLM can invoke."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters
    handler: Callable           # The actual function to call

class OllamaClient:
    """
    Client for local Ollama LLM with tool-calling support.
    """
    def __init__(self, model: str = "llama2"):
        self.model = model
        self.logger = logging.getLogger("OllamaClient")
        self.tools: Dict[str, Tool] = {}
        self.conversation: List[Dict] = []
        
        # System prompt
        self.system_prompt = ""
    
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
    
    def register_tool(self, tool: Tool):
        """Register a tool the LLM can call."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def _build_tool_descriptions(self) -> str:
        """Build a text block describing available tools for the LLM."""
        if not self.tools:
            return ""
        
        desc = "\n\n## Available Tools\nYou can call tools by responding with a JSON block in this exact format:\n```json\n{\"tool\": \"tool_name\", \"args\": {\"param1\": \"value1\"}}\n```\n\nAvailable tools:\n"
        
        for name, tool in self.tools.items():
            params_desc = ""
            for pname, pinfo in tool.parameters.items():
                req = " (required)" if pinfo.get('required', False) else " (optional)"
                params_desc += f"    - {pname}: {pinfo.get('description', '')}{req}\n"
            
            desc += f"\n### {name}\n{tool.description}\nParameters:\n{params_desc}"
        
        return desc
    
    def _extract_tool_call(self, response: str) -> Optional[Dict]:
        """Try to extract a tool call from the LLM response."""
        # Look for JSON blocks
        import re
        
        # Try ```json blocks first
        json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        for block in json_blocks:
            try:
                parsed = json.loads(block.strip())
                if 'tool' in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Try inline JSON
        json_matches = re.findall(r'\{[^{}]*"tool"[^{}]*\}', response)
        for match in json_matches:
            try:
                parsed = json.loads(match)
                if 'tool' in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _execute_tool(self, tool_call: Dict) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call.get('tool')
        args = tool_call.get('args', {})
        
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
        
        tool = self.tools[tool_name]
        try:
            result = tool.handler(**args)
            self.logger.info(f"Tool '{tool_name}' executed successfully")
            return str(result)
        except Exception as e:
            self.logger.error(f"Tool '{tool_name}' failed: {e}")
            return f"Error executing {tool_name}: {e}"
    
    def chat(self, user_message: str, max_tool_rounds: int = 3) -> str:
        """
        Send a message to Ollama with tool-calling support.
        The LLM can call tools, get results, and respond.
        """
        # Build messages
        messages = []
        
        # System prompt with tool descriptions
        system = self.system_prompt + self._build_tool_descriptions()
        messages.append({"role": "system", "content": system})
        
        # Conversation history (keep last 10 turns)
        for msg in self.conversation[-20:]:
            messages.append(msg)
        
        # Current message
        messages.append({"role": "user", "content": user_message})
        self.conversation.append({"role": "user", "content": user_message})
        
        # Tool-calling loop
        for round_num in range(max_tool_rounds + 1):
            try:
                response_text = self._call_ollama(messages)
            except Exception as e:
                self.logger.error(f"Ollama call failed: {e}")
                return f"⚠️ LLM connection error: {e}\nFalling back to rule-based mode."
            
            # Check for tool calls
            tool_call = self._extract_tool_call(response_text)
            
            if tool_call and round_num < max_tool_rounds:
                # Execute the tool
                tool_result = self._execute_tool(tool_call)
                
                # Add assistant response and tool result to context
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Tool result for {tool_call['tool']}:\n{tool_result}\n\nNow respond to the user based on this result."})
            else:
                # Final response (no more tool calls)
                self.conversation.append({"role": "assistant", "content": response_text})
                return response_text
        
        return response_text
    
    def _call_ollama(self, messages: List[Dict]) -> str:
        """Make an API call to the local Ollama server."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
            }
        }
        
        response = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        data = response.json()
        return data.get("message", {}).get("content", "")
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            return resp.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            data = resp.json()
            return [m['name'] for m in data.get('models', [])]
        except:
            return []
