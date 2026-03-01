import os
import re
import logging
from typing import Dict

class SoulParser:
    """
    Reads the plain-English SOUL.md master configuration file and 
    extracts quantitative constraints for the system to obey.
    """
    def __init__(self, soul_path: str = "SOUL.md"):
        self.logger = logging.getLogger(__name__)
        self.soul_path = soul_path
        
    def parse_constraints(self) -> Dict[str, float]:
        """
        Parses the markdown file for specific key-value constraints.
        Default values are returned if the file is missing or values aren't found.
        """
        constraints = {
            "max_leverage": 1.0,
            "max_position_size_pct": 0.10, # 10%
            "max_daily_loss": 500.0,
            "max_open_positions": 5
        }
        
        if not os.path.exists(self.soul_path):
            self.logger.warning(f"Could not find {self.soul_path}. Using default system constraints.")
            return constraints
            
        try:
            with open(self.soul_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Regex patterns for natural language parsing
            # E.g. "Max Leverage: 5x" or "- **Max Leverage:** 5x"
            
            leverage_match = re.search(r'(?i)max(?:\s+)leverage.*?(?:[:\* ]+)(\d+(?:\.\d+)?)x?', content)
            if leverage_match:
                constraints['max_leverage'] = float(leverage_match.group(1))

            pos_size_match = re.search(r'(?i)max(?:\s+)position(?:\s+)size.*?(?:[:\* ]+)(\d+(?:\.\d+)?)%?', content)
            if pos_size_match:
                constraints['max_position_size_pct'] = float(pos_size_match.group(1)) / 100.0

            daily_loss_match = re.search(r'(?i)max(?:\s+)daily(?:\s+)loss.*?(?:[:\*\$ ]+)(\d+(?:\.\d+)?)', content)
            if daily_loss_match:
                constraints['max_daily_loss'] = float(daily_loss_match.group(1))
                
            open_pos_match = re.search(r'(?i)max(?:\s+)open(?:\s+)positions.*?(?:[:\* ]+)(\d+)', content)
            if open_pos_match:
                constraints['max_open_positions'] = int(open_pos_match.group(1))
                
            self.logger.info(f"🧠 SOUL.md Parsed Successfully: {constraints}")
            return constraints
            
        except Exception as e:
            self.logger.error(f"Failed to parse SOUL.md: {e}")
            return constraints
