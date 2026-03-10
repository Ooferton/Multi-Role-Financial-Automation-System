import os
import json
import logging
import requests
from typing import List, Dict
from core.experience_buffer import ExperienceBuffer

logger = logging.getLogger("ContinuousLearner")

class ContinuousLearner:
    """
    Orchestrates the nightly learning flywheel.
    Pulls data from the experience buffer and triggers a micro-LoRA fine-tuning run.
    """
    def __init__(self, buffer: ExperienceBuffer):
        self.buffer = buffer
        self.training_script = "training/train_qlora.py"
        self.temp_dataset = "training/micro_dataset.jsonl"
        self.server_url = os.getenv("SENTIENCE_SERVER_URL", "http://localhost:8000")

    def run_nightly_update(self):
        """Main entry point for the nightly update."""
        logger.info("🌙 Starting nightly continuous learning update...")
        
        # 1. Collect Unprocessed Experiences
        experiences = self.buffer.get_unprocessed_experiences()
        if not experiences:
            logger.info("No new high-quality experiences found. Skipping update.")
            return
            
        logger.info(f"Found {len(experiences)} new experiences to learn from.")
        
        # 2. Format as Instruction Pairs
        # Positive rewards: Reinforce the behavior
        # Negative rewards: (In a more advanced version, we'd generate a correction)
        # For now, we'll focus on positive reinforcement (reward > 0.5)
        training_pairs = []
        ids_processed = []
        
        for exp in experiences:
            if exp["reward"] > 0.5:
                training_pairs.append({
                    "instruction": exp["prompt"],
                    "input": "",
                    "output": exp["response"]
                })
                ids_processed.append(exp["id"])
        
        if not training_pairs:
            logger.info("No positive reward experiences found for reinforcement.")
            return

        # 3. Save Temporary Dataset
        with open(self.temp_dataset, "w") as f:
            for pair in training_pairs:
                f.write(json.dumps(pair) + "\n")
        
        # 4. Trigger Training Script
        # We'll use a subprocess to run the training script with the micro-dataset
        logger.info(f"Triggering micro-LoRA update on {len(training_pairs)} examples.")
        import subprocess
        try:
            # Overwrite the default dataset path for the script temporarily or pass as arg
            # For this prototype, we'll assume the script checks for micro_dataset.jsonl
            # if it exists, otherwise use standard.
            result = subprocess.run(["python", self.training_script], capture_all=True)
            if result.return_code == 0:
                logger.info("✅ Micro-LoRA update successful.")
                self.buffer.mark_as_processed(ids_processed)
                
                # 5. Notify Model Server to reload (if it supports hot-reloading)
                self._notify_server_reload()
            else:
                logger.error(f"❌ Training script failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to run nightly update: {e}")

    def _notify_server_reload(self):
        """Tells the FastAPI server to reload the LoRA adapter."""
        try:
            # We'll add this endpoint to model_server.py next
            requests.post(f"{self.server_url}/v1/model/reload", timeout=5)
            logger.info("Model server notified to reload adapter.")
        except:
            logger.warning("Could not notify model server to reload.")
