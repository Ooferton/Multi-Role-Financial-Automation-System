import os
import json
import pandas as pd
from datetime import datetime
import logging
from core.llm_supervisor import LLMSupervisor
from core.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NightlyJournal")

class NightlyJournal:
    """
    Tier 1 AI Process: Analyzes daily trades and writes a LEARNINGS.md summary using the LLM.
    """
    def __init__(self, trades_path="data/trades.csv", output_dir="logs/learnings"):
        self.trades_path = trades_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # We reuse the LLM Supervisor for the journaling logic
        self.llm = LLMSupervisor()

    def generate_journal(self):
        logger.info("Starting Nightly Journal Generation...")
        
        if not os.path.exists(self.trades_path):
            logger.warning(f"No trades file found at {self.trades_path}. Skipping journal.")
            return

        # Load today's trades
        df = pd.read_csv(self.trades_path)
        if df.empty:
            logger.info("No trades executed today. Skipping journal.")
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today_date = datetime.now().date()
        today_trades = df[df['timestamp'].dt.date == today_date]

        if today_trades.empty:
            logger.info("No trades executed TODAY. Skipping journal.")
            return

        # Summarize the day's activity
        total_trades = len(today_trades)
        symbols_traded = today_trades['symbol'].unique().tolist()
        
        # Prepare context for the LLM
        context = {
            "date": str(today_date),
            "total_trades": total_trades,
            "symbols_traded": symbols_traded,
            "trade_log": today_trades.to_dict(orient="records")
        }

        # Query the LLM Supervisor to generate structured learnings
        prompt = f"""
        You are the Tier 1 CEO of the Sentience Trading System.
        Review the following trades executed today by your RL agents:
        {json.dumps(context, indent=2)}

        Write a 'Nightly Journal' summarizing:
        1. Market conditions today (based on the trades and reasoning provided).
        2. What strategies worked well.
        3. What strategies failed or lost money.
        4. Proposed rule changes for the SOUL.md constraints list (do not write SOUL.md, just propose ideas).

        Format your response in Markdown.
        """

        try:
            # Call the LLM supervisor for a raw text response
            response = self.llm._call_model_raw(prompt)
            
            # Save the learning log
            file_name = f"{self.output_dir}/learnings_{today_date}.md"
            with open(file_name, "w") as f:
                f.write(response)

            logger.info(f"Nightly Journal written to {file_name} successfully.")
        except Exception as e:
            logger.error(f"Failed to generate nightly journal: {e}")

if __name__ == "__main__":
    journal = NightlyJournal()
    journal.generate_journal()
