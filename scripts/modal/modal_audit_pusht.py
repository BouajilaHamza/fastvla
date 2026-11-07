"""
Audit Arabic Push-T Dataset (hamzabouajila/ar-pusht-image) on Hugging Face.
Inspects instructions, episode counts, columns, and translations.
"""

import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets>=2.20.0", "pyarrow>=15.0.0", "huggingface_hub>=0.26.0")
)

app = modal.App("fastvla-audit-pusht", image=image)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    try:
        _hf_token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        pass

secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})]


@app.function(timeout=300, secrets=secrets)
def audit_pusht():
    from datasets import load_dataset
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from collections import Counter

    repo_id = "hamzabouajila/ar-pusht-image"
    print(f"🔍 Auditing {repo_id}...")

    local_path = hf_hub_download(repo_id=repo_id, filename="data/train-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(local_path)

    print(f"Schema columns: {table.column_names}")
    print(f"Total rows: {len(table)}")

    col_names = table.column_names
    instructions = table["instruction"].to_pylist() if "instruction" in col_names else []
    episodes = table["episode_index"].to_pylist() if "episode_index" in col_names else []

    instr_counts = Counter(instructions)
    print(f"\nUnique instructions found ({len(instr_counts)}):")
    for instr, count in instr_counts.most_common(10):
        print(f"  - '{instr}': {count} occurrences")

    num_episodes = len(set(episodes)) if episodes else "Unknown"
    print(f"\nTotal episodes: {num_episodes}")

    return {
        "columns": col_names,
        "total_rows": len(table),
        "unique_instructions": dict(instr_counts),
        "num_episodes": num_episodes,
    }


@app.local_entrypoint()
def main():
    audit_pusht.remote()
