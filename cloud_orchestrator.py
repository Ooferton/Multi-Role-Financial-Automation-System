
import subprocess
import time
import sys
import os
import signal
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ORCHESTRATOR] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/orchestrator.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Orchestrator")

def main():
    logger.info("Starting Sentience Cloud Orchestrator...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    processes = []

    try:
        # 1. Start Portfolio Runner (Standard Polling)
        logger.info("🚀 Launching Portfolio Runner...")
        p_live = subprocess.Popen(
            [sys.executable, "live_runner.py", "--skip-dashboard"],
            stdout=open("logs/live_runner_stdout.log", "w"),
            stderr=open("logs/live_runner_stderr.log", "w")
        )
        processes.append(p_live)

        # 2. Stagger and Start Bitcoin Runner (High-Speed)
        logger.info("⏳ Waiting 5s for warmup staggering...")
        time.sleep(5)
        
        logger.info("₿ Launching Bitcoin Runner...")
        p_bit = subprocess.Popen(
            [sys.executable, "bitcoin_runner.py", "--skip-dashboard"],
            stdout=open("logs/bitcoin_runner_stdout.log", "w"),
            stderr=open("logs/bitcoin_runner_stderr.log", "w")
        )
        processes.append(p_bit)

        # 3. Start Dashboard (Foreground)
        logger.info("📊 Launching Sentience Dashboard (Foreground)...")
        # HF expects streamlit on port 7860
        cmd_dash = [
            sys.executable, "-m", "streamlit", "run", "dashboard.py", 
            "--server.port", "7860", 
            "--server.address", "0.0.0.0", 
            "--server.headless", "true"
        ]
        
        p_dash = subprocess.Popen(cmd_dash)
        processes.append(p_dash)

        logger.info("✅ All systems online. Monitoring processes...")

        # Monitor loop
        while True:
            for p in processes:
                if p.poll() is not None:
                    name = "Portfolio" if p == p_live else "Bitcoin" if p == p_bit else "Dashboard"
                    logger.error(f"❌ Process {name} (PID: {p.pid}) terminated unexpectedly with code {p.returncode}")
                    
                    # If dashboard dies, we should probably exit the whole container
                    if p == p_dash:
                        raise KeyboardInterrupt
            
            time.sleep(10)

    except KeyboardInterrupt:
        logger.warning("Stopping Sentience System...")
    finally:
        for p in processes:
            try:
                logger.info(f"Terminating PID: {p.pid}")
                p.terminate()
            except: pass
        logger.info("Cleanup complete. Goodbye.")

if __name__ == "__main__":
    main()
