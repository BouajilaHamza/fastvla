#!/usr/bin/env python3
"""
Standardized Evaluation Harness for LIBERO Benchmark Suites.

Evaluates FastVLA models across LIBERO suites (libero_10, libero_spatial, libero_object, libero_goal).
Supports:
1. Closed-loop physical rollouts in LIBERO MuJoCo gym environments (when libero is installed).
2. Offline multi-trajectory validation and action drift metrics across demonstration episodes.
3. Native Arabic instruction evaluation and dialectal robustness stress testing.

Usage:
    python scripts/evaluation/eval_libero_suite.py --suite libero_10 --num-episodes 10
    python scripts/evaluation/eval_libero_suite.py --suite libero_spatial --eval-mode offline
"""

import argparse
import logging
import time
from typing import Dict, List, Optional
import numpy as np
import torch
from PIL import Image

from fastvla.inference import FastVLAPolicyRunner, EdgeTelemetryHUD
from fastvla.data.datasets import get_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LIBERO_SUITES = {
    "libero_10": "libero_10",
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
}


def run_offline_trajectory_eval(
    runner: FastVLAPolicyRunner,
    suite_name: str,
    num_episodes: int = 5,
    max_steps: int = 50,
    dataset_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluates policy action predictions against held-out demonstration trajectories.
    """
    if not dataset_path or not os.path.exists(dataset_path):
        raise ValueError(
            f"Valid demonstration dataset path must be provided for offline trajectory evaluation. "
            f"Provided: {dataset_path}. Synthetic/random target evaluation is strictly prohibited."
        )

    raise NotImplementedError("Real dataset offline evaluation loader under active implementation.")


def run_gym_closed_loop_eval(
    runner: FastVLAPolicyRunner,
    suite_name: str,
    num_episodes: int = 5,
    max_steps: int = 500,
) -> Dict[str, float]:
    """
    Runs full closed-loop rollouts inside LIBERO / Robosuite MuJoCo gym environment.
    Fails closed if the simulator or environment is missing.
    """
    try:
        import libero
        import libero.libero.envs as libero_envs
        has_libero = True
    except ImportError:
        raise RuntimeError(
            "LIBERO simulator environment not found in current environment.\n"
            "To run closed-loop gym rollouts, install LIBERO: pip install libero\n"
            "Silent fallback to mock/synthetic evaluation is disabled to maintain benchmark integrity."
        )

    raise NotImplementedError(
        "Native LIBERO gym closed-loop evaluation loop is being re-implemented with true "
        "camera observation inputs and environment goal checking (no placeholder success)."
    )


def main():
    parser = argparse.ArgumentParser(description="LIBERO Benchmark Suite Evaluation Harness")
    parser.add_argument("--suite", type=str, choices=list(LIBERO_SUITES.keys()), default="libero_10")
    parser.add_argument("--model-path", type=str, default=None, help="Trained FastVLA checkpoint or HF repo")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--eval-mode", type=str, choices=["gym", "offline"], default="offline")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=16)

    args = parser.parse_args()

    runner = FastVLAPolicyRunner(
        model_id_or_path=args.model_path,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
    )

    if args.eval_mode == "gym":
        results = run_gym_closed_loop_eval(runner, args.suite, args.num_episodes)
    else:
        results = run_offline_trajectory_eval(runner, args.suite, args.num_episodes)

    print("\n" + "=" * 60)
    print("LIBERO BENCHMARK EVALUATION RESULTS")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:10.4f}")
        else:
            print(f"{k:25s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
