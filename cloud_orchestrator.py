
import subprocess
import time
import sys
import os
import signal
import logging
import threading
import datetime

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

def _stream_subprocess_stderr(proc, name):
    """Thread target: forward subprocess stderr lines to the orchestrator logger."""
    try:
        for line in iter(proc.stderr.readline, b''):
            decoded = line.decode('utf-8', errors='replace').rstrip()
            if decoded:
                logger.error(f"[{name}] {decoded}")
    except Exception:
        pass

def _learning_loop():
    """Background thread to run the Phase C Self-Learning processes."""
    logger.info("Autonomous Learning Loop starting...")
    while True:
        now = datetime.datetime.now()
        # Run Nightly Journal at 23:55 local time
        if now.hour == 23 and now.minute == 55:
            logger.info("Executing Nightly Journal (Trade Reflection)...")
            subprocess.run([sys.executable, "core/nightly_journal.py"])
            
            # If Friday, also run Weekly Rule Updater
            if now.weekday() == 4:
                logger.info("Executing Weekly Rule Updater (SOUL.md Rewrite)...")
                subprocess.run([sys.executable, "core/weekly_rule_updater.py"])
                
            # Sleep to avoid multi-triggering in the same minute
            time.sleep(120)
        else:
            time.sleep(30)

def main():
    logger.info("Starting Sentience Cloud Orchestrator...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 0. Run BIOS
    logger.info("Running System BIOS...")
    try:
        from bios import run_bios
        if not run_bios():
            logger.error("BIOS Check Failed. Aborting cloud startup.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to run BIOS: {e}")
        #sys.exit(1) # Continue for now if import fails

    processes = []

    try:
        # 1. Start CEO Manager (Supervisor)
        logger.info("Launching CEO Manager (Supervisor)...")
        p_manager = subprocess.Popen(
            [sys.executable, "manager_service.py"],
            stdout=open("logs/manager_stdout.log", "w"),
            stderr=subprocess.PIPE
        )
        threading.Thread(target=_stream_subprocess_stderr, args=(p_manager, "Manager"), daemon=True).start()
        processes.append(p_manager)

        # 2. Start Portfolio Runner
        logger.info("Launching Portfolio Runner...")
        p_live = subprocess.Popen(
            [sys.executable, "live_runner.py", "--skip-dashboard"],
            stdout=open("logs/live_runner_stdout.log", "w"),
            stderr=subprocess.PIPE
        )
        threading.Thread(target=_stream_subprocess_stderr, args=(p_live, "Portfolio"), daemon=True).start()
        processes.append(p_live)

        # 3. Stagger and Start Bitcoin Runner
        logger.info("Staggering Bitcoin Runner (10s)...")
        time.sleep(10)
        
        logger.info("Launching Bitcoin Runner...")
        p_bit = subprocess.Popen(
            [sys.executable, "bitcoin_runner.py", "--skip-dashboard"],
            stdout=open("logs/bitcoin_runner_stdout.log", "w"),
            stderr=subprocess.PIPE
        )
        threading.Thread(target=_stream_subprocess_stderr, args=(p_bit, "Bitcoin"), daemon=True).start()
        processes.append(p_bit)

        # 3.5 Start Learning Loop daemon
        threading.Thread(target=_learning_loop, daemon=True).start()

        # 4. Start Dashboard (Foreground)
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

        logger.info("✅ All cloud systems online. Monitoring processes...")

        # Monitor loop
        while True:
            for p in processes:
                if p.poll() is not None:
                    name = "Manager" if p == p_manager else \
                           "Portfolio" if p == p_live else \
                           "Bitcoin" if p == p_bit else \
                           "Dashboard"
                    logger.error(f"❌ Process {name} (PID: {p.pid}) terminated with code {p.returncode}")
                    
                    # If dashboard or manager dies, suggest critical failure
                    if p in [p_dash, p_manager]:
                        raise KeyboardInterrupt
            
            time.sleep(15)

    except KeyboardInterrupt:
        logger.warning("Stopping Sentience Cloud Systems...")
    finally:
        for p in processes:
            try:
                logger.info(f"Terminating PID: {p.pid}")
                p.terminate()
            except: pass
        logger.info("Cleanup complete. Goodbye.")

if __name__ == "__main__":
    main()
