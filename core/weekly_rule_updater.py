import os
import glob
import logging
from core.llm_supervisor import LLMSupervisor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WeeklyUpdater")

class WeeklyRuleUpdater:
    """
    Tier 1 AI Process: Analyzes accumulated LEARNINGS.md files to autonomously update
    the SOUL.md master rules file, completing the V3 self-improvement loop.
    """
    def __init__(self, learnings_dir="logs/learnings", soul_file="SOUL.md"):
        self.learnings_dir = learnings_dir
        self.soul_file = soul_file
        self.llm = LLMSupervisor()

    def update_rules(self):
        logger.info("Starting Weekly Autonomous Rule Update...")

        if not os.path.exists(self.learnings_dir):
            logger.warning("No learnings directory found. Skipping update.")
            return

        # Gather the last 7 learning files
        learning_files = sorted(glob.glob(os.path.join(self.learnings_dir, "*.md")))[-7:]
        
        if not learning_files:
            logger.info("No learning files found to analyze. Skipping update.")
            return

        accumulated_learnings = ""
        for file in learning_files:
            try:
                with open(file, "r") as f:
                    accumulated_learnings += f"\n--- {os.path.basename(file)} ---\n"
                    accumulated_learnings += f.read()
            except Exception as e:
                logger.error(f"Error reading learning file {file}: {e}")

        # Read current SOUL.md
        current_soul = "No SOUL.md exists yet."
        if os.path.exists(self.soul_file):
            try:
                with open(self.soul_file, "r") as f:
                    current_soul = f.read()
            except Exception as e:
                logger.error(f"Error reading {self.soul_file}: {e}")

        prompt = f"""
        You are the Tier 1 CEO of the Sentience Trading System.
        Your job is to read the accumulated 'Nightly Journals' from the past week
        and rewrite the master 'SOUL.md' file to adapt to new market realities.

        CURRENT SOUL.md:
        {current_soul}

        PAST WEEK'S LEARNINGS:
        {accumulated_learnings}

        INSTRUCTIONS:
        1. Synthesize the learnings. Identify what is causing losses and what is generating profit.
        2. Output a newly written SOUL.md. 
        3. You may adjust `MAX_RISK_PER_TRADE`, `MAX_LEVERAGE`, or add specific string constraints
           under `RESTRICTED_ASSETS` or new structural rules.
        4. Maintain the plain-English instructional format of SOUL.md.
        5. DO NOT output any other text besides the exact Markdown content for the new SOUL.md. 
           Do not use markdown code blocks (```) around the entire output, just raw text ready to save.
        """

        try:
            logger.info("Querying LLM Supervisor for new SOUL.md synthesis...")
            new_soul = self.llm._call_model_raw(prompt)
            
            # Clean up potential markdown formatting wrapping the whole file
            if new_soul.startswith("```markdown\n"):
                new_soul = new_soul[12:]
            if new_soul.startswith("```\n"):
                new_soul = new_soul[4:]
            if new_soul.endswith("\n```"):
                new_soul = new_soul[:-4]

            # Save the new version
            with open(self.soul_file, "w") as f:
                f.write(new_soul)

            logger.info("Successfully autonomously updated SOUL.md based on active learnings. Phase C Self-Learning Loop engaged.")
        except Exception as e:
            logger.error(f"Failed to perform weekly rule update: {e}")

if __name__ == "__main__":
    updater = WeeklyRuleUpdater()
    updater.update_rules()
