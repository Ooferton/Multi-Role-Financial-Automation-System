import logging
import json
import re
from typing import Dict, Any, List, Optional
from core.tool_registry import ToolRegistry
from core.llm_supervisor import LLMSupervisor

logger = logging.getLogger("AgentRouter")

class AgentRouter:
    """
    The main decision loop of Sentience Core.
    Coordinates between the LLM, the Tool Registry, and System Memory.
    """
    def __init__(self, llm_supervisor: LLMSupervisor, tool_registry: ToolRegistry):
        self.llm = llm_supervisor
        self.tool_registry = tool_registry
        self.max_iterations = 5

    def chat(self, user_msg: str, context: Optional[Dict] = None) -> str:
        """
        Main entry point for user interaction.
        Supports multi-step reasoning and tool chaining.
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt(context)},
            {"role": "user", "content": user_msg}
        ]

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Agent Router: Iteration {iteration}")
            
            # 1. Call LLM
            response_text = self.llm._call_model_raw(self._format_prompt(messages))
            
            # 2. Parse for tool calls (Format: name(arg1=val1, ...))
            tool_call = self._parse_tool_call(response_text)
            
            if tool_call:
                name = tool_call["name"]
                args = tool_call["args"]
                
                # 3. Execute Tool
                result = self.tool_registry.call_tool(name, args)
                
                # 4. Feed result back to messages
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Tool Result ({name}): {json.dumps(result)}"})
                continue # Next loop to let model synthesize result
            else:
                # No more tool calls, return final response
                return response_text

        return "Error: Reached maximum reasoning steps."

    def _build_system_prompt(self, context: Optional[Dict] = None) -> str:
        tools_list = self.tool_registry.get_tool_schemas()
        # Cleaned up schema for the prompt
        tools_str = ""
        for t in tools_list:
            tools_str += f"- {t['name']}: {t['description']} (Params: {list(t['parameters'].get('properties', {}).keys())})\n"

        prompt = f"""You are Sentience Core, the autonomous brain of a financial system.
Your mission is to manage trades, risk, and research proactively.
You have access to the following tools:
{tools_str}

To call a tool, use ONLY the format: name(arg1=val1, ...) 
Example: execute_trade(symbol="AAPL", side="BUY", qty=10, reason="Momentum bounce")

Current Context:
{json.dumps(context or {}, indent=2)}

Strictly follow SOUL.md rules and always prioritize risk management.
"""
        return prompt

    def _format_prompt(self, messages: List[Dict]) -> str:
        """Formats standard chat messages into the model's expected prompt structure."""
        formatted = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            formatted += f"<|{role}|>\n{content}<|end|>\n"
        formatted += "<|assistant|>\n"
        return formatted

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        """Looks for tool call format in the text."""
        # Regex to find: name(arg1="val", arg2=123)
        pattern = r"(\w+)\((.*)\)"
        match = re.search(pattern, text)
        if not match:
            return None
            
        name = match.group(1)
        args_str = match.group(2)
        
        args = {}
        # Parse simple arguments
        # Example: symbol="AAPL", side="BUY", qty=10
        arg_pairs = re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|(\d+\.?\d*)|(\[.*?\]))', args_str)
        for k, v_str, v_num, v_list in arg_pairs:
            if v_str:
                args[k] = v_str
            elif v_num:
                args[k] = float(v_num) if "." in v_num else int(v_num)
            elif v_list:
                try:
                    args[k] = json.loads(v_list.replace("'", '"'))
                except:
                    args[k] = v_list
                    
        return {"name": name, "args": args}
