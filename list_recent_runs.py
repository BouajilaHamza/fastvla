import wandb
api = wandb.Api()
entity = "bouajilahamza-diaindustries"
project = "fastvla-rl"
runs = api.runs(f"{entity}/{project}", order="-created_at", per_page=5)
for run in runs:
    print(f"Run: {run.name} | ID: {run.id} | State: {run.state} | Created: {run.created_at}")
