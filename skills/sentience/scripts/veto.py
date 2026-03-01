import os
import sys

def manage_veto(command):
    lock_path = os.path.join(os.getcwd(), 'data', 'openclaw_veto.lock')
    
    if command == "halt":
        with open(lock_path, 'w') as f:
            f.write("Vetoed by OpenClaw at " + os.popen('date /t').read().strip())
        print("SYSTEM VETOED. All future trades will be blocked until reset.")
    elif command == "resume":
        if os.path.exists(lock_path):
            os.remove(lock_path)
            print("VETO REMOVED. System resumed.")
        else:
            print("System was not vetoed.")
    else:
        print("Usage: python veto.py [halt|resume]")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        manage_veto(sys.argv[1])
    else:
        print("Usage: python veto.py [halt|resume]")
