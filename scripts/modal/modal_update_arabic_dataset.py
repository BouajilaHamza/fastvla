"""
FastVLA: Retag and Fix Arabic Translations for 'hamzabouajila/ar-libero-10-image'.

Fixes the misaligned task indices and hallucinatory machine translations in the
Arabic LIBERO-10 dataset using the exact ground-truth mapping from lerobot/libero_10_image.
Parallelized on Modal across 26 workers for high-speed batch processing.
"""

import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pyarrow>=15.0.0", "huggingface_hub>=0.26.0", "pandas>=2.0.0")
)

app = modal.App("fastvla-retag-arabic-dataset", image=image)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token and os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
    try:
        _hf_token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except Exception:
        pass

secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})]

# Ground-truth mapping from task_index (0..9) with linguistically rich, verified Arabic variations
TASK_INDEX_TO_VARIATIONS = {
    0: [
        "ضع الكوب الأبيض على الطبق الأيسر وضع الكوب الأصفر والأبيض على الطبق الأيمن",
        "انقل الكوب الأبيض إلى الطبق الأيسر والكوب الأصفر والأبيض إلى الطبق الأيمن",
        "ضع الكوب الأبيض فوق الطبق على اليسار والكوب الأصفر والأبيض على الطبق الأيمن",
    ],
    1: [
        "ضع الكوب الأبيض على الطبق وضع حلوى الشوكولاتة على يمين الطبق",
        "ضع الكوب الأبيض فوق الطبق ثم ضع علبة بودينغ الشوكولاتة عن يمين الطبق",
        "انقل الكوب الأبيض إلى الطبق وحلوى الشوكولاتة إلى يمين الطبق",
    ],
    2: [
        "ضع الكوب الأصفر والأبيض في الميكروويف وأغلقه",
        "أدخل الكوب الأصفر والأبيض داخل الميكروويف ثم أغلق الباب",
        "ضع الكوب الأصفر والأبيض داخل الميكروويف وأغلق بابه",
    ],
    3: [
        "شغّل الموقد وضع إبريق الموكا عليه",
        "أشعل الموقد وضع إبريق القهوة فوقه",
        "شغّل عين الموقد وضع ركوة الموكا عليها",
    ],
    4: [
        "ضع حساء الحروف وعلبة الجبن الكريمي في السلة",
        "ضع علبة حساء الحروف وعلبة الجبن في السلة معاً",
        "انقل حساء الحروف وعلبة الجبن الكريمي إلى داخل السلة",
    ],
    5: [
        "ضع حساء الحروف وصلصة الطماطم في السلة",
        "ضع علبة حساء الحروف وصلصة الطماطم داخل السلة",
        "انقل حساء الحروف وعلبة صلصة الطماطم إلى السلة",
    ],
    6: [
        "ضع إبريقي الموكا على الموقد",
        "ضع كلا إبريقي القهوة فوق الموقد",
        "ضع إبريقي القهوة الموكا على سطح الموقد",
    ],
    7: [
        "ضع علبة الجبن الكريمي والزبدة في السلة",
        "ضع قالب الزبدة وعلبة الجبن الكريمي داخل السلة",
        "انقل علبة الجبن والزبدة إلى السلة",
    ],
    8: [
        "ضع الوعاء الأسود في الدرج السفلي من الخزانة وأغلقه",
        "ضع الزبدية السوداء داخل الدرج السفلي للخزانة ثم أغلق الدرج",
        "أدخل الوعاء الأسود في الدرج السفلي للخزانة وأغلقه",
    ],
    9: [
        "التقط الكتاب وضعه في الحجرة الخلفية لحامل الأدوات",
        "احمل الكتاب وضعه في القسم الخلفي من منظم المكتب",
        "ارفع الكتاب وضعه في الحجرة الخلفية لحامل الأغراض",
    ],
}


@app.function(timeout=600, secrets=secrets)
def process_single_chunk(file_path: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    import os

    repo_id = "hamzabouajila/ar-libero-10-image"
    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    file_name = os.path.basename(file_path)
    print(f"📥 [{file_name}] Downloading chunk...")
    local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", token=hf_token)
    table = pq.read_table(local_path)

    col_names = table.column_names
    instruction_idx = col_names.index("instruction")
    task_indices = table["task_index"].to_pylist()
    episode_indices = table["episode_index"].to_pylist() if "episode_index" in col_names else [0] * len(task_indices)

    new_instructions = []
    for t_idx, ep_idx in zip(task_indices, episode_indices):
        variations = TASK_INDEX_TO_VARIATIONS.get(int(t_idx))
        if variations:
            # Deterministic variation selection per episode
            chosen = variations[int(ep_idx) % len(variations)]
        else:
            chosen = "أكمل المهمة المطلوبة بدقة"
        new_instructions.append(chosen)

    new_instruction_array = pa.array(new_instructions, type=pa.string())
    new_table = table.set_column(instruction_idx, "instruction", new_instruction_array)

    out_local = f"/tmp/{file_name}"
    pq.write_table(new_table, out_local, compression="zstd")

    print(f"⬆️ [{file_name}] Uploading corrected chunk to Hugging Face...")
    api.upload_file(
        path_or_fileobj=out_local,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"fix(translations): align verified Arabic instructions for {file_name}",
        token=hf_token,
    )
    if os.path.exists(out_local):
        os.remove(out_local)
    print(f"✅ [{file_name}] Successfully processed and updated!")
    return file_name


@app.function(timeout=1800, secrets=secrets)
def orchestrate_retag():
    from huggingface_hub import HfApi
    import os

    repo_id = "hamzabouajila/ar-libero-10-image"
    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)

    print(f"🔄 Scanning dataset {repo_id}...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
    parquet_files = sorted([f for f in files if f.startswith("data/train-") and f.endswith(".parquet")])
    print(f"Found {len(parquet_files)} parquet chunks. Dispatching parallel workers...")

    results = list(process_single_chunk.map(parquet_files))
    print(f"\n🎉 Successfully retagged all {len(results)} chunks on Hugging Face Hub!")
    return results


@app.local_entrypoint()
def main():
    res = orchestrate_retag.remote()
    print(f"Finished: {len(res)} chunks updated.")
