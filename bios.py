import os
import sys
import logging
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
    print(f"{CYAN}=== FINANCIAL SYSTEM BIOS v3.0 ==={RESET}")
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

    # 4. Check Python Dependencies
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
