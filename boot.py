import subprocess
import time
import sys
import os
from bios import run_bios

# BIOS Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

def boot_system():
    # 1. Run BIOS
    if not run_bios():
        print(f"{CYAN}BIOS check failed. Please resolve issues and try again.{RESET}")
        sys.exit(1)

    print(f"{CYAN}Starting System Boot Sequence...{RESET}")

    processes = []

    # 2. Launch Trading Engine
    print(f"[{GREEN}BOOTING{RESET}] Worker (Trading Engine)...")
    engine = subprocess.Popen(
        [sys.executable, "live_runner.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    processes.append(engine)
    time.sleep(5)

    # 3. Launch Dashboard
    print(f"[{GREEN}BOOTING{RESET}] Dashboard (Streamlit)...")
    dashboard = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    processes.append(dashboard)

    print(f"\n{GREEN}SYSTEM ONLINE.{RESET}")
    print("Dashboard: http://localhost:8501")
    print("All processes are running in separate consoles.")
    print("Press Ctrl+C in this window to end all tasks (or close their respective windows).")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{CYAN}Shutting down system...{RESET}")
        for p in processes:
            p.terminate()
        print("Goodbye.")

if __name__ == "__main__":
    boot_system()
