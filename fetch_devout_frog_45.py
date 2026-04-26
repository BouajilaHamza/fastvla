import wandb
import pandas as pd
import numpy as np

api = wandb.Api()
entity = "bouajilahamza-diaindustries"
project = "fastvla-rl"
run_id = "qe81p4i3"

run = api.run(f"{entity}/{project}/{run_id}")
print(f"--- Summary for {run.name} ({run_id}) ---")
print(f"State: {run.state}")

# Fetch full history for loss analysis
history = run.history()
if not history.empty:
    print(f"Total Steps Logged: {len(history)}")
    
    if "loss" in history:
        losses = history["loss"].dropna()
        print(f"Starting Loss: {losses.iloc[0]:.4f}")
        print(f"Latest Loss: {losses.iloc[-1]:.4f}")
        print(f"Min Loss: {losses.min():.4f}")
        
    if "lr" in history:
        print(f"Latest LR: {history['lr'].iloc[-1]:.2e}")

    # Speed calculation
    if "_timestamp" in history:
        duration = history["_timestamp"].iloc[-1] - history["_timestamp"].iloc[0]
        it_per_sec = len(history) / duration if duration > 0 else 0
        print(f"Throughput: {it_per_sec:.2f} iterations/sec")

    # Check for videos
    files = run.files()
    videos = [f.name for f in files if f.name.endswith(".mp4")]
    print(f"Videos uploaded: {len(videos)}")
    for v in videos[:3]: # show first 3
        print(f" - {v}")
else:
    print("Run is empty or hasn't synced history yet.")
