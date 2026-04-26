"""
Evaluation utilities for FastVLA.
Includes tools for recording rollouts and measuring task success.
"""

import os
import torch
import numpy as np
import wandb
from typing import Optional, Any, Tuple
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def record_rollout_video(
    model: torch.nn.Module,
    env: Any,
    preprocess: Any,
    step_or_epoch: int,
    phase: str = "rl",
    instruction: str = "إدفع الكتلة إلى الهدف",
    max_steps: int = 300,
    device: str = "cuda",
    threshold: float = 0.0,
) -> float:
    """
    Perform a rollout in the environment and log a video to WandB.

    Args:
        model: The VLA model to evaluate.
        env: The Gymnasium environment.
        preprocess: Image preprocessing transforms.
        step_or_epoch: Current training step or epoch (for logging).
        phase: Name of the phase (e.g., 'bc', 'rl').
        instruction: Natural language instruction for the robot.
        max_steps: Maximum steps per episode.
        device: Torch device.
        threshold: Only log video if max_coverage exceeds this percentage.

    Returns:
        The maximum coverage achieved during the rollout.
    """
    model.eval()
    frames = []
    obs, _ = env.reset()
    max_coverage = 0.0

    # Prepare input IDs
    if hasattr(model, "tokenizer") and model.tokenizer:
        input_ids = model.tokenizer(instruction, return_tensors="pt")["input_ids"].to(
            device
        )
    else:
        # Fallback for models without direct tokenizer access
        input_ids = torch.zeros((1, 1), dtype=torch.long).to(device)

    for _ in range(max_steps):
        # Extract pixels
        img = obs.get("pixels", obs) if isinstance(obs, dict) else obs
        frames.append(img)

        # Preprocess and predict
        px = preprocess(img).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=px, input_ids=input_ids)
            # FastVLAModel returns (action_preds, loss) or ((action_preds, value_preds), loss)
            if isinstance(outputs, tuple):
                preds = outputs[0]
                action_t = preds[0] if isinstance(preds, tuple) else preds
            else:
                action_t = outputs
            action = action_t[0].cpu().numpy()

        # Get current state for relative delta logic (PushT specific)
        # We perform a zero-action step to extract info safely if needed
        _, _, _, _, info = env.step(np.array([0.0, 0.0]))
        agent_pos = np.array(info.get("agent_pos", [256, 256]))

        # Calculate target position (Relative Delta logic)
        delta = action[:2]
        target_pos = agent_pos + (delta * 15.0)
        target_pos = np.clip(target_pos, [12, 25], [511, 511])

        # Step environment
        obs, _, terminated, truncated, info = env.step(target_pos)

        curr_cov = info.get("coverage", 0.0)
        if curr_cov > max_coverage:
            max_coverage = curr_cov

        if terminated or truncated:
            break

    # Log to WandB if threshold met
    if max_coverage * 100 >= threshold:
        video_path = f"rollout_{phase}_s{step_or_epoch}.mp4"
        try:
            clip = ImageSequenceClip(list(frames), fps=10)
            clip.write_videofile(video_path, codec="libx264", logger=None)

            wandb.log(
                {
                    f"eval/{phase}_video": wandb.Video(
                        video_path,
                        caption=f"Step {step_or_epoch} - Cov: {max_coverage*100:.2f}%",
                    ),
                    f"eval/{phase}_max_coverage": max_coverage * 100,
                }
            )
        except Exception as e:
            print(f"Error during video logging: {e}")
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    return max_coverage
