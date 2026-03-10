import time
import os
import logging
import json
from core.llm_supervisor import LLMSupervisor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - MANAGER - %(levelname)s - %(message)s')
logger = logging.getLogger("AutonomousManager")

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

    veto_path = os.path.join(os.getcwd(), 'data', 'system_veto.lock')
    veto_status = "ACTIVE" if os.path.exists(veto_path) else "INACTIVE"

    # 2. Present context to LLM Supervisor
    context = {
        "mode": status_data.get('mode', 'UNKNOWN'),
        "vix": status_data.get('vix', 'N/A'),
        "vibe": status_data.get('vibe', 'N/A'),
        "veto_status": veto_status,
        "macro_sentiment": status_data.get('vibe', 'NEUTRAL'),
        "regime": status_data.get('mode', 'UNKNOWN'),
        "daily_pnl": status_data.get('daily_pnl', 0),
        "global_drawdown": status_data.get('drawdown', 0),
        "current_constraints": {
            "max_leverage": status_data.get('max_leverage', 2.0),
            "max_position_size_pct": status_data.get('max_pos_pct', 0.15)
        }
    }
    
    try:
        supervisor = LLMSupervisor()
        decision = supervisor.analyze_market_context(context)
        
        if decision:
            reasoning = decision.get("reasoning", "No reasoning provided.")
            summary = decision.get("executive_summary", "")
            emergency = decision.get("emergency_stop", False)
            
            logger.info(f"CEO Decision: {reasoning}")
            
            # Log decision
            with open("logs/manager_decisions.log", "a", encoding="utf-8") as f:
                f.write(f"--- {time.ctime()} ---\n")
                f.write(f"Reasoning: {reasoning}\n")
                f.write(f"Summary: {summary}\n")
                f.write(f"Emergency Stop: {emergency}\n\n")
            
            # Execute emergency stop if needed
            if emergency:
                logger.warning("CEO ORDERED EMERGENCY STOP!")
                with open(veto_path, "w") as f:
                    f.write(f"Emergency stop ordered at {time.ctime()}")
            elif os.path.exists(veto_path) and not emergency:
                # If veto is active but CEO says resume
                os.remove(veto_path)
                logger.info("CEO cleared the emergency stop. Resuming operations.")
        else:
            logger.info("CEO cycle completed (LLM unavailable or returned no decision).")

    except Exception as e:
        logger.error(f"Failed to run management cycle: {e}")

if __name__ == "__main__":
    logger.info("Autonomous Manager Service STARTED (Interval: 5 minutes)")
    while True:
        run_management_cycle()
        time.sleep(300)
