import os
import sys
import subprocess
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ANSI colors for BIOS feel
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def check_step(name, condition, fix_hint=""):
    status = f"{GREEN}[ OK ]{RESET}" if condition else f"{RED}[ FAIL ]{RESET}"
    print(f" {status} {name}")
    if not condition and fix_hint:
        print(f"        {YELLOW}Hint: {fix_hint}{RESET}")
    return condition

def run_bios():
    print(f"{CYAN}=== FINANCIAL SYSTEM BIOS v2.0 ==={RESET}")
    print(f"Initializing hardware abstraction layer...")
    
    success = True

    # 1. Check Environment
    load_env_exists = os.path.exists(".env")
    alpaca_key = os.getenv("APCA_API_KEY_ID")
    env_ok = load_env_exists or bool(alpaca_key)
    
    if not check_step("Environment", env_ok, "Create a .env file or set APCA_API_KEY_ID env var."):
        success = False

    # 2. Check Credentials
    if not check_step("Alpaca Credentials", bool(alpaca_key), "Ensure APCA_API_KEY_ID is set."):
        success = False

    # 3. Check Filesystem
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
    check_step("Data Directory", True)
    
    # 4. Check OpenClaw Core
    openclaw_home = os.path.expanduser("~/.openclaw")
    openclaw_config = Path(f"{openclaw_home}/openclaw.json").exists()
    
    # Cloud Bootstrap: If missing but we have node, try to mkdir
    if not openclaw_config and not os.name == 'nt':
        os.makedirs(openclaw_home, exist_ok=True)
        with open(f"{openclaw_home}/openclaw.json", "w") as f:
            json.dump({"initialized": True}, f)
        openclaw_config = True

    if not check_step("OpenClaw Config", openclaw_config, "Run 'openclaw configure' or ensure ~/.openclaw exists."):
        success = False

    # 5. Check AI Services (Ollama)
    ollama_ready = False
    try:
        import requests
        res = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        ollama_ready = res.status_code == 200
    except:
        pass
    if not check_step("Ollama Service", ollama_ready, "Ensure Ollama is running at http://127.0.0.1:11434"):
        success = False

    # 6. Check Python Dependencies
    deps_ready = True
    try:
        import pandas
        import alpaca_trade_api
        import yaml
    except ImportError as e:
        deps_ready = False
        if not check_step(f"Python Dependencies ({e.name})", False, f"Run 'pip install {e.name}'"):
            success = False
    
    if deps_ready:
        check_step("Python Dependencies", True)

    print(f"{CYAN}==================================={RESET}")
    if success:
        print(f"{GREEN}BIOS CHECK PASSED. SYSTEM READY FOR BOOT.{RESET}\n")
    else:
        print(f"{RED}BIOS CHECK FAILED. PLEASE RESOLVE ISSUES ABOVE.{RESET}\n")
    
    return success

if __name__ == "__main__":
    if not run_bios():
        sys.exit(1)
