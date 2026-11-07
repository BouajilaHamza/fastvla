"""
FastVLA: Retag and Update Arabic Push-T Dataset ('hamzabouajila/ar-pusht-image').

Eliminates the mistranslation 'دفع الحجر إلى الهدف' (Push the stone to the goal)
and replaces it with verified Modern Standard Arabic instructions featuring clean
orthography and deterministic episode-level paraphrasing.
"""

import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyarrow>=15.0.0", "huggingface_hub>=0.26.0", "pandas>=2.0.0")
)

app = modal.App("fastvla-update-pusht-dataset", image=image)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    try:
        _hf_token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        pass

secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})]

# Grammatically pristine Modern Standard Arabic variations with correct hamzat wasl
# and natural robotics terminology avoiding mixed-direction Latin characters.
PUSHT_ARABIC_VARIATIONS = [
    "ادفع الكتلة إلى الهدف",
    "ادفع القطعة التائية إلى منطقة الهدف",
    "حرّك الكتلة التائية نحو الهدف وطابقها معه",
    "وجّه القطعة التائية وضعها بدقة داخل الهدف",
    "ادفع الكتلة التائية إلى الموضع المحدد",
]


@app.function(timeout=600, secrets=secrets)
def update_pusht_dataset():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from collections import Counter

    repo_id = "hamzabouajila/ar-pusht-image"
    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    file_path = "data/train-00000-of-00001.parquet"
    print(f"📥 Downloading {file_path} from {repo_id}...")
    local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", token=hf_token)
    table = pq.read_table(local_path)

    col_names = table.column_names
    instruction_idx = col_names.index("instruction")
    episode_indices = table["episode_index"].to_pylist() if "episode_index" in col_names else [0] * len(table)

    print(f"Loaded {len(table)} rows across {len(set(episode_indices))} episodes.")
    print("Applying pristine Modern Standard Arabic instructions...")

    new_instructions = []
    for ep_idx in episode_indices:
        chosen = PUSHT_ARABIC_VARIATIONS[int(ep_idx) % len(PUSHT_ARABIC_VARIATIONS)]
        new_instructions.append(chosen)

    new_instruction_array = pa.array(new_instructions, type=pa.string())
    new_table = table.set_column(instruction_idx, "instruction", new_instruction_array)

    out_local = "/tmp/train-00000-of-00001.parquet"
    print(f"Writing updated parquet to {out_local} (zstd compression)...")
    pq.write_table(new_table, out_local, compression="zstd")

    # Verify updated distribution
    counts = Counter(new_instructions)
    print("\nVerified new instruction distribution:")
    for instr, count in counts.items():
        print(f"  - '{instr}': {count} frames ({count / len(new_instructions) * 100:.1f}%)")

    print(f"\n⬆️ Uploading corrected dataset to {repo_id}...")
    api.upload_file(
        path_or_fileobj=out_local,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="fix(translation): replace hallucinatory 'الحجر' with verified Modern Standard Arabic instructions",
        token=hf_token,
    )

    if os.path.exists(out_local):
        os.remove(out_local)

    print(f"\n🎉 Successfully retagged and pushed {repo_id} to Hugging Face Hub!")
    return dict(counts)


@app.local_entrypoint()
def main():
    print("🚀 Initiating Push-T Arabic dataset update on Modal...")
    res = update_pusht_dataset.remote()
    print("Done! Updated distribution summary:")
    for k, v in res.items():
        print(f"  {k}: {v} frames")
