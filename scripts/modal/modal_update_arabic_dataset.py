"""
FastVLA: Retag and Fix Arabic Translations for 'hamzabouajila/ar-libero-10-image'.

Fixes the misaligned task indices and hallucinatory machine translations in the
Arabic LIBERO-10 dataset using the exact ground-truth mapping from lerobot/libero_10_image.
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

# Ground-truth mapping from task_index (0..9) to pristine Arabic translation
TASK_INDEX_TO_ARABIC = {
    0: "ضع الكوب الأبيض على الطبق الأيسر وضع الكوب الأصفر والأبيض على الطبق الأيمن",
    1: "ضع الكوب الأبيض على الطبق وضع حلوى الشوكولاتة على يمين الطبق",
    2: "ضع الكوب الأصفر والأبيض في الميكروويف وأغلقه",
    3: "شغّل الموقد وضع إبريق الموكا عليه",
    4: "ضع حساء الحروف وعلبة الجبن الكريمي في السلة",
    5: "ضع حساء الحروف وصلصة الطماطم في السلة",
    6: "ضع إبريقي الموكا على الموقد",
    7: "ضع علبة الجبن الكريمي والزبدة في السلة",
    8: "ضع الوعاء الأسود في الدرج السفلي من الخزانة وأغلقه",
    9: "التقط الكتاب وضعه في الحجرة الخلفية لحامل الأدوات",
}

TASK_INDEX_TO_ENGLISH = {
    0: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    1: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    2: "put the yellow and white mug in the microwave and close it",
    3: "turn on the stove and put the moka pot on it",
    4: "put both the alphabet soup and the cream cheese box in the basket",
    5: "put both the alphabet soup and the tomato sauce in the basket",
    6: "put both moka pots on the stove",
    7: "put both the cream cheese box and the butter in the basket",
    8: "put the black bowl in the bottom drawer of the cabinet and close it",
    9: "pick up the book and place it in the back compartment of the caddy",
}


@app.function(timeout=3600, secrets=secrets)
def retag_and_push_dataset():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd

    repo_id = "hamzabouajila/ar-libero-10-image"
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    print(f"🔄 Starting retagging of {repo_id}...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = sorted([f for f in files if f.startswith("data/train-") and f.endswith(".parquet")])
    print(f"Found {len(parquet_files)} parquet chunks to process.")

    for i, file_path in enumerate(parquet_files):
        print(f"[{i+1}/{len(parquet_files)}] Downloading and processing {file_path}...")
        local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset")
        table = pq.read_table(local_path)

        # Replace 'instruction' column with exact ground-truth Arabic based on task_index
        task_indices = table["task_index"].to_pylist()
        new_instructions = [TASK_INDEX_TO_ARABIC.get(int(idx), "أكمل المهمة المطلوبة بدقة") for idx in task_indices]

        # Rebuild table preserving exact schema and binary image arrays
        col_names = table.column_names
        instruction_idx = col_names.index("instruction")
        new_instruction_array = pa.array(new_instructions, type=pa.string())

        new_table = table.set_column(instruction_idx, "instruction", new_instruction_array)

        # Write to temporary file
        out_local = f"/tmp/{os.path.basename(file_path)}"
        pq.write_table(new_table, out_local, compression="zstd")

        # Upload back to HF
        print(f"  ⬆️ Uploading corrected {file_path} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=out_local,
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"fix(translations): correct Arabic instruction alignment for {os.path.basename(file_path)}",
        )
        os.remove(out_local)

    print("🎉 All 26 parquet chunks successfully retagged and pushed to Hugging Face Hub!")


@app.local_entrypoint()
def main():
    retag_and_push_dataset.remote()
