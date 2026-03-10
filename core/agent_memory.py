import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger("AgentMemory")

class AgentMemory:
    """
    Persistent memory for Sentience Core.
    Stores session history and long-term facts in SQLite.
    """
    def __init__(self, db_path: str = "data/sentience_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Sessions table for conversation history
        c.execute('''CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT,
                        timestamp DATETIME,
                        role TEXT,
                        content TEXT
                    )''')
        
        # Knowledge table for long-term facts
        c.execute('''CREATE TABLE IF NOT EXISTS knowledge (
                        fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT,
                        content TEXT,
                        confidence REAL,
                        created_at DATETIME
                    )''')
                    
        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO sessions (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
                  (session_id, datetime.now(), role, content))
        conn.commit()
        conn.close()

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT role, content FROM sessions WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", 
                  (session_id, limit))
        rows = c.fetchall()
        conn.close()
        
        # Convert to list of dicts and reverse to chronological order
        history = [{"role": r, "content": c} for r, c in rows]
        return history[::-1]

    def store_fact(self, category: str, content: str, confidence: float = 1.0):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO knowledge (category, content, confidence, created_at) VALUES (?, ?, ?, ?)",
                  (category, content, confidence, datetime.now()))
        conn.commit()
        conn.close()
        logger.info(f"New fact stored in {category}: {content[:50]}...")

    def search_knowledge(self, query: str) -> List[str]:
        """Simple keyword search for now. Could be upgraded to vector search later."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # Simple LIKE search for prototype
        c.execute("SELECT content FROM knowledge WHERE content LIKE ? LIMIT 5", (f"%{query}%",))
        results = [row[0] for row in c.fetchall()]
        conn.close()
        return results

    def clear_session(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
