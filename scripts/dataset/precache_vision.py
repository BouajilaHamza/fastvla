#!/usr/bin/env python3
"""
Vision feature precomputation and caching tool for FastVLA.
Encodes raw dataset camera images using the frozen vision backbone once offline,
storing precomputed embeddings to disk (safetensors or pt).
This completely eliminates vision encoder compute and activation memory during training (up to 2x speedup).
"""

import os
import argparse
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from fastvla import FastVLAModel, get_dataset
from fastvla.config import FastVLAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precache_dataset(
    dataset_name: str,
    output_dir: str,
    model_name: str = "openvla-7b",
    batch_size: int = 16,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Initializing vision encoder from {model_name}...")
    cfg = FastVLAConfig(dummy=True if device == "cpu" else False)
    try:
        model = FastVLAModel.from_pretrained(model_name)
    except Exception:
        logger.warning(f"Could not load weights for {model_name}, creating model with dummy vision encoder.")
        model = FastVLAModel(cfg)

    vision_encoder = model.vision_encoder.to(device)
    vision_encoder.eval()

    logger.info(f"Loading dataset {dataset_name}...")
    dataset = get_dataset(dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Precaching {len(dataset)} samples to {output_dir}...")
    cached_manifest = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Caching vision tokens"):
            item = dataset[idx]
            # Process images [num_cams, C, H, W]
            images = item.get("images", {})
            if isinstance(images, dict):
                cam_tensors = [images[k] for k in sorted(images.keys())]
                pixel_values = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(device)  # [1, num_cams, C, H, W]
            else:
                pixel_values = torch.as_tensor(images).unsqueeze(0).to(device)

            B, num_cams, C, H, W = pixel_values.shape
            flat_imgs = pixel_values.view(B * num_cams, C, H, W)
            cam_feats = vision_encoder(flat_imgs)
            if hasattr(cam_feats, "last_hidden_state"):
                cam_feats = cam_feats.last_hidden_state
            
            T_v, D_v = cam_feats.shape[1], cam_feats.shape[2]
            cached_feats = cam_feats.view(B, num_cams * T_v, D_v).squeeze(0).cpu()

            cache_path = os.path.join(output_dir, f"sample_{idx:07d}.pt")
            torch.save(cached_feats, cache_path)
            cached_manifest.append(cache_path)

    logger.info(f"✅ Successfully cached {len(cached_manifest)} visual token files in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Precache vision features for FastVLA training")
    parser.add_argument("--dataset", type=str, default="pusht", help="Dataset name")
    parser.add_argument("--output-dir", type=str, default="data/cached_vision/pusht", help="Output directory")
    parser.add_argument("--model", type=str, default="openvla-7b", help="Model name or path")
    args = parser.parse_args()

    precache_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model,
    )

if __name__ == "__main__":
    main()
