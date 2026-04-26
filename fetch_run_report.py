import wandb
import pandas as pd

api = wandb.Api()
entity = "bouajilahamza-diaindustries"
project = "fastvla-rl"
run_name = "devout-frog-45"

runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})

if not runs:
    print(f"Run {run_name} not found.")
else:
    run = runs[0]
    print(f"--- Report for {run_name} ({run.id}) ---")
    print(f"State: {run.state}")
    print(f"Started: {run.created_at}")
    
    # Get config
    config = run.config
    print(f"Config: BC Epochs: {config.get('bc_epochs')}, RL Epochs: {config.get('rl_epochs')}")
    
    # Get metrics
    history = run.history(keys=["loss", "lr", "epoch", "max_coverage", "avg_reward", "_timestamp"])
    if not history.empty:
        avg_loss = history["loss"].mean()
        min_loss = history["loss"].min()
        latest_loss = history["loss"].iloc[-1]
        
        # Calculate speed (it/s)
        time_diff = history["_timestamp"].iloc[-1] - history["_timestamp"].iloc[0]
        steps = len(history)
        speed = steps / time_diff if time_diff > 0 else 0
        
        print(f"Latest Loss: {latest_loss:.4f}")
        print(f"Min Loss: {min_loss:.4f}")
        print(f"Avg Loss: {avg_loss:.4f}")
        
        if "max_coverage" in history and not history["max_coverage"].isnull().all():
            max_cov = history["max_coverage"].max()
            print(f"Max Coverage: {max_cov:.2f}%")
        
        print(f"Estimated Throughput: {speed:.2f} logs/sec")
    else:
        print("No history metrics found yet.")

