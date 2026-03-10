import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

logger = logging.getLogger("ExperienceBuffer")

class ExperienceBuffer:
    """
    Records LLM interactions and their outcomes (rewards).
    Used as the data source for nightly micro-LoRA updates.
    """
    def __init__(self, db_path: str = "data/experience_buffer.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Primary buffer table
        c.execute('''CREATE TABLE IF NOT EXISTS experiences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        prompt TEXT,
                        response TEXT,
                        tools_called TEXT,
                        outcome TEXT,
                        reward_signal REAL,
                        is_processed BOOLEAN DEFAULT 0
                    )''')
                    
        conn.commit()
        conn.close()

    def log_interaction(self, prompt: str, response: str, tools: List[str]):
        """Logs a new interaction before the outcome is known."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""INSERT INTO experiences (timestamp, prompt, response, tools_called) 
                     VALUES (?, ?, ?, ?)""",
                  (datetime.now(), prompt, response, json.dumps(tools)))
        last_id = c.lastrowid
        conn.commit()
        conn.close()
        return last_id

    def backfill_outcome(self, interaction_id: int, outcome: str, reward: float):
        """Updates an interaction with its real-world result and signal (-1.0 to 1.0)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("UPDATE experiences SET outcome = ?, reward_signal = ? WHERE id = ?",
                  (outcome, reward, interaction_id))
        conn.commit()
        conn.close()
        logger.info(f"Experience {interaction_id} updated: Reward {reward}")

    def get_unprocessed_experiences(self, min_reward: float = 0.5) -> List[Dict]:
        """Retrieves high-quality experiences for fine-tuning."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # We only want experiences with a known reward signal that haven't been trained on yet
        c.execute("""SELECT id, prompt, response, reward_signal FROM experiences 
                     WHERE is_processed = 0 AND reward_signal IS NOT NULL""")
        rows = c.fetchall()
        
        experiences = []
        for r in rows:
            experiences.append({
                "id": r[0],
                "prompt": r[1],
                "response": r[2],
                "reward": r[3]
            })
            
        conn.close()
        return experiences

    def mark_as_processed(self, ids: List[int]):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executemany("UPDATE experiences SET is_processed = 1 WHERE id = ?", [(idx,) for idx in ids])
        conn.commit()
        conn.close()
