
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
            
            # Phase C: Continuous Learning Update
            logger.info("Executing Continuous Learning (Micro-LoRA Update)...")
            from core.experience_buffer import ExperienceBuffer
            from core.continuous_learner import ContinuousLearner
            buffer = ExperienceBuffer()
            learner = ContinuousLearner(buffer)
            learner.run_nightly_update()
            
            # If Friday, also run Weekly Rule Updater
            if now.weekday() == 4:
                logger.info("Executing Weekly Rule Updater (SOUL.md Rewrite)...")
                subprocess.run([sys.executable, "core/weekly_rule_updater.py"])
            
            # Sleep to avoid multi-triggering in the same minute
            time.sleep(120)
            
        # Run Daily Researcher at 03:00 AM (Pre-Market)
        elif now.hour == 3 and now.minute == 0:
            logger.info("Executing Daily Autonomous Research Scan...")
            subprocess.run([sys.executable, "-m", "agents.researcher_agent"])
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

    # 0.5 Run Initial Research Scan (Ensure fresh targets on boot)
    logger.info("Running Initial Autonomous Research Scan...")
    try:
        subprocess.run([sys.executable, "-m", "agents.researcher_agent"], check=False)
    except Exception as e:
        logger.error(f"Initial research scan failed: {e}")

    processes = []

    try:
        # 1. Start Sentience Model Server (Local Inference) - DISABLED FOR FREE TIER (OOM KILL)
        # logger.info("Launch Sentience Core Model Server...")
        # p_model = subprocess.Popen(
        #     [sys.executable, "core/model_server.py"],
        #     stdout=open("logs/sentience_model_stdout.log", "a"),
        #     stderr=subprocess.PIPE
        # )
        # threading.Thread(target=_stream_subprocess_stderr, args=(p_model, "ModelServer"), daemon=True).start()
        # processes.append(p_model)
        
        # # Wait for model to load (can take 30-60s)
        # logger.info("Waiting for Sentience Core to wake up (30s)...")
        # time.sleep(30)

        # 2. Start Portfolio Runner
        logger.info("Launching Portfolio Runner...")
        p_live = subprocess.Popen(
            [sys.executable, "live_runner.py", "--skip-dashboard"],
            stdout=open("logs/live_runner_stdout.log", "a"),
            stderr=subprocess.PIPE
        )
        threading.Thread(target=_stream_subprocess_stderr, args=(p_live, "Portfolio"), daemon=True).start()
        processes.append(p_live)

        # 3. Start Sentience Autonomous Service (Heartbeat / Oversight)
        logger.info("Launching Sentience Autonomous Service (Oversight)...")
        p_sentience = subprocess.Popen(
            [sys.executable, "core/sentience_service.py"],
            stdout=open("logs/sentience_service_stdout.log", "a"),
            stderr=subprocess.PIPE
        )
        threading.Thread(target=_stream_subprocess_stderr, args=(p_sentience, "SentienceService"), daemon=True).start()
        processes.append(p_sentience)

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
                    if p == p_live: name = "Portfolio"
                    elif p == p_dash: name = "Dashboard"
                    # elif p == p_model: name = "ModelServer" # Disabled for free tier
                    elif p == p_sentience: name = "SentienceService"
                    else: name = "UnknownProcess"
                    
                    logger.error(f"❌ Process {name} (PID: {p.pid}) terminated with code {p.returncode}")
                    
                    # If dashboard dies, suggest critical failure
                    if p == p_dash:
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
