import wandb
import os
import json
import pandas as pd

# Load API key from env file if not set in environment
if not os.getenv("WANDB_API_KEY"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("WANDB_API_KEY="):
                os.environ["WANDB_API_KEY"] = line.split("=")[1].strip()

api = wandb.Api()
entity = "bouajilahamza-diaindustries"
project = "lerobot"

print(f"Fetching runs from {entity}/{project}...")
runs = api.runs(f"{entity}/{project}")

run_summaries = []

for run in runs:
    print(f"Loading run: {run.name} ({run.id}) Status: {run.state}")
    
    # Get config
    config = run.config
    
    # Get summary metrics (last logged values)
    summary = run.summary._json_dict
    
    # Get some history (sampled)
    try:
        history = run.history(samples=100)
        # Convert history to list of dicts for JSON serialization
        history_data = history.to_dict(orient="records")
    except Exception as e:
        print(f"  Error fetching history for {run.id}: {e}")
        history_data = []

    run_summaries.append({
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "config": config,
        "summary": summary,
        "history_len": len(history_data),
        "history_sample": history_data
    })

# Save results to a file for analysis
with open("wandb_fetch_results.json", "w") as f:
    json.dump(run_summaries, f, indent=2)

print(f"\nFetched {len(run_summaries)} runs. Saved to wandb_fetch_results.json")
