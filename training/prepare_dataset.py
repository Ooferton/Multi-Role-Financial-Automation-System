import os
import json
import pandas as pd
import glob
from typing import List, Dict

def prepare_dataset():
    """
    Converts raw system logs, trade history, and rule files into 
    instruction-tuning pairs for Phi-3 fine-tuning.
    """
    dataset: List[Dict] = []
    
    # 1. Process Trade Logs (Behavioral Learning)
    trades_path = "data/trades.csv"
    if os.path.exists(trades_path):
        df = pd.read_csv(trades_path)
        # Group by date to create summary reasoning examples
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        for date, group in df.groupby('date'):
            trades_summary = group[['symbol', 'side', 'qty', 'price', 'reasoning']].to_dict(orient='records')
            
            instruction = f"Review the following trades executed on {date} and explain the system's logic."
            input_data = json.dumps(trades_summary[:10], indent=2) # Keep it manageable
            output = f"On {date}, the system focused on {', '.join(group['symbol'].unique())}. " \
                     f"The primary reasoning across trades was: '{group['reasoning'].iloc[0]}'. " \
                     f"The system maintained a consistent strategy with {len(group)} total executions."
            
            dataset.append({"instruction": instruction, "input": input_data, "output": output})

    # 2. Process SOUL.md (Rule Compliance)
    soul_path = "SOUL.md"
    if os.path.exists(soul_path):
        with open(soul_path, "r") as f:
            soul_content = f.read()
            
        dataset.append({
            "instruction": "Explain the core trading limits and rules of the Sentience system.",
            "input": soul_content,
            "output": "The Sentience system operates under strict quantitative limits: Max Leverage is 5x, Max Position Size is 20%, Max Daily Loss is $2000, and Max Open Positions is 10. It uses HMM for regime detection and CVaR for risk trimming, ensuring unemotional, math-driven execution."
        })

    # 3. Process Nightly Learnings (Self-Reflection)
    learnings_dir = "logs/learnings"
    if os.path.exists(learnings_dir):
        files = glob.glob(os.path.join(learnings_dir, "*.md"))
        for file in files:
            with open(file, "r") as f:
                content = f.read()
                date_str = os.path.basename(file).replace("learnings_", "").replace(".md", "")
                
                dataset.append({
                    "instruction": f"What were the key learnings and strategic reflections for {date_str}?",
                    "input": "",
                    "output": content.strip()
                })

    # 4. Synthetic Corporate/Budgeting Data
    dataset.append({
        "instruction": "How should the system handle a request to reallocate the corporate R&D budget based on Q1 performance?",
        "input": "Q1 Performance: Strategy A (+15%), Strategy B (-5%), R&D Budget: $500k",
        "output": "Based on Q1 performance, I recommend reallocating 20% of the R&D budget from Strategy B to Strategy A to capitalize on the higher alpha. This maintains the total $500k cap while optimizing for realized returns. I will update the budget module accordingly."
    })

    # 5. Multi-User Communication Examples
    dataset.append({
        "instruction": "Relay a status update from the Automated Trader to the Risk Manager regarding market volatility.",
        "input": "User A (Trader): VIX just spiked to 35. User B (Risk Manager): Noted.",
        "output": "Autonomous Relay: User A reports a VIX spike to 35. I have cross-referenced this with the Risk Manager (User B) and am preemptively tightening leverage limits to 1.5x as per SOUL.md protocols. Both users have been synchronized."
    })

    # Save to JSONL
    output_path = "training/dataset.jsonl"
    with open(output_path, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Dataset preparation complete. Saved {len(dataset)} examples to {output_path}")

if __name__ == "__main__":
    prepare_dataset()
