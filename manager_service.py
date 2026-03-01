import time
import subprocess
import os
import logging
import json

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - MANAGER - %(levelname)s - %(message)s')
logger = logging.getLogger("OpenClawManager")

def run_management_cycle():
    logger.info("Starting Managerial Review Cycle...")
    
    # 1. Collect Status Data
    status_path = os.path.join(os.getcwd(), 'data', 'sentience_status.json')
    status_data = {}
    if os.path.exists(status_path):
        try:
            with open(status_path, 'r') as f:
                status_data = json.load(f)
        except: pass

    veto_path = os.path.join(os.getcwd(), 'data', 'openclaw_veto.lock')
    veto_status = "ACTIVE" if os.path.exists(veto_path) else "INACTIVE"

    # 2. Present to CEO (OpenClaw)
    # Injecting system instruction directly into the message to bypass loading issues
    mandate = (
        "CRITICAL: You are the CEO and RISK MANAGER. NOT an AI assistant. DO NOT APOLOGIZE.\n"
        "You have control of the trading system. MANDATORY DECISION REQUIRED.\n\n"
    )
    msg = (
        f"CRITICAL: You are the CEO and RISK MANAGER. NOT an AI assistant. DO NOT APOLOGIZE. "
        f"MANDATORY DECISION REQUIRED. "
        f"STATUS: MODE={status_data.get('mode', 'UNKNOWN')}, VIX={status_data.get('vix', 'N/A')}, VIBE={status_data.get('vibe', 'N/A')}, VETO={veto_status}. "
        f"OFFERED COMMANDS: 'COMMAND: VETO', 'COMMAND: RESUME', 'COMMAND: STATUS_QUO', 'COMMAND: SWITCH_STRATEGY [conservative|aggressive|hft_only]'. "
        f"Choose based on the VIX (High VIX = VETO or Conservative, High Vibe = Aggressive). Reasoning first."
    )
    
    try:
        # Determine openclaw command based on OS
        if os.name == 'nt':
            openclaw_path = r"C:\Users\carst\AppData\Roaming\npm\openclaw.cmd"
            if not os.path.exists(openclaw_path):
                openclaw_path = "openclaw" # Fallback to path
        else:
            openclaw_path = "openclaw"
            
        cmd_str = f'"{openclaw_path}" agent --message "{msg}" --agent main --session-id CEO_REVIEW --thinking low'
        
        # Capture output
        res = subprocess.run(cmd_str, capture_output=True, text=True, timeout=60, shell=True)
        response_text = (res.stdout + res.stderr).strip()
        logger.info(f"CEO Turn Completed.")
        
        with open("logs/manager_decisions.log", "a", encoding="utf-8") as f:
            f.write(f"--- {time.ctime()} ---\n{response_text}\n\n")

        # 3. Execute Command (Fuzzy Match)
        response_upper = response_text.upper()
        if "COMMAND: VETO" in response_upper and veto_status == "INACTIVE":
            logger.warning("CEO ORDERED VETO!")
            subprocess.run(["python", "skills/sentience/scripts/veto.py", "halt"])
        elif "COMMAND: RESUME" in response_upper and veto_status == "ACTIVE":
            logger.info("CEO ORDERED RESUME!")
            subprocess.run(["python", "skills/sentience/scripts/veto.py", "resume"])
        elif "COMMAND: SWITCH_STRATEGY" in response_upper:
            # Extract the preset
            preset = "conservative"
            if "AGGRESSIVE" in response_upper: preset = "aggressive"
            elif "HFT_ONLY" in response_upper: preset = "hft_only"
            
            logger.warning(f"CEO ORDERED STRATEGY SWITCH TO: {preset.upper()}")
            subprocess.run(["python", "skills/sentience/scripts/switch_strategy.py", preset])
        else:
            logger.info(f"CEO Action: {response_text[:100]}...")

    except Exception as e:
        logger.error(f"Failed to trigger OpenClaw review: {e}")

if __name__ == "__main__":
    logger.info("OpenClaw Autonomous Manager Service STARTED (Interval: 5 minutes)")
    while True:
        run_management_cycle()
        time.sleep(300)
