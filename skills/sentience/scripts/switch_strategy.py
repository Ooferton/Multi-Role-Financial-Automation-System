import json
import os
import sys

def switch_strategy(preset):
    allowed_presets = ["conservative", "aggressive", "hft_only"]
    if preset not in allowed_presets:
        print(f"Error: Invalid preset '{preset}'. Allowed: {allowed_presets}")
        return

    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    config_path = os.path.join(data_dir, 'active_strategies.json')
    
    config = {"preset": preset}
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"STRATEGY SWITCHED: System brain now set to '{preset.upper()}'.")
    print(f"The live_runner will detect and reload this within 1 second.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        switch_strategy(sys.argv[1].lower())
    else:
        print("Usage: python switch_strategy.py [conservative|aggressive|hft_only]")
