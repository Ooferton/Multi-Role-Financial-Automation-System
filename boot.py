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
        print(f"{CYAN}Ollama not running. Attempting auto-start...{RESET}")
        try:
            # Start Ollama in a hidden background process
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            print("Waking up AI Brain (this may take 10 seconds)...")
            time.sleep(10)
            # Re-run BIOS
            if not run_bios():
                print("Boot aborted. Please start Ollama manually.")
                sys.exit(1)
        except Exception as e:
            print(f"Auto-start failed: {e}")
            sys.exit(1)

    print(f"{CYAN}Starting System Boot Sequence...{RESET}")

    processes = []

    # 2. Launch OpenClaw Gateway
    print(f"[{GREEN}BOOTING{RESET}] Commander (OpenClaw Gateway)...")
    gw = subprocess.Popen(
        ["cmd", "/c", "openclaw", "gateway", "--force"],
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    processes.append(gw)
    time.sleep(5)  # Let gateway warm up

    # 3. Launch Trading Engine
    print(f"[{GREEN}BOOTING{RESET}] Worker (Trading Engine)...")
    engine = subprocess.Popen(
        [sys.executable, "live_runner.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    processes.append(engine)
    time.sleep(5)

    # 4. Launch CEO Supervisor
    print(f"[{GREEN}BOOTING{RESET}] CEO Supervisor (Manager Service)...")
    manager = subprocess.Popen(
        [sys.executable, "manager_service.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    )
    processes.append(manager)

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
