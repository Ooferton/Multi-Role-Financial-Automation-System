---
name: sentience
description: EMERGENCY CONTROL & MANAGEMENT. You are the CEO. Use this skill to OVERRIDE and CONTROL the trading engine. Use 'scripts/management_summary.py' to see the situation. Use 'scripts/veto.py halt' to STOP the system immediately if performance is unacceptable. Use 'scripts/veto.py resume' to allow trading. You ARE the manager.
---

# Sentience Monitor Skill

Provides a window into the autonomous trading engine's brain and grants **Veto Power** over decisions. Use this as your primary console for managing the system.

## Usage
```powershell
# Get a full management briefing (Status + Performance)
python scripts/management_summary.py

# Get a summary of the current market state
python scripts/monitor.py status

# Dynamically reloads the trading brain with a new strategy preset
python scripts/switch_strategy.py [conservative|aggressive|hft_only]

# Review recent performance and AI reasoning
python scripts/review.py

# Emergency Veto: Stop all trading immediately
python scripts/veto.py halt

# Resume trading: Lift the veto
python scripts/veto.py resume
```
