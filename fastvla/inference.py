"""
High-Frequency Action Chunking Engine and Edge Robot Runner for FastVLA.

Implements lightweight, zero-training-bloat policy inference, rolling-horizon
action chunk buffering, multi-step flow ODE solvers, and real-time edge telemetry.
"""

import time
import collections
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class ActionChunkBuffer:
    """
    Rolling-horizon action chunk queue for smooth, high-frequency robot control.
    
    Maintains a buffer of predicted continuous actions and handles temporal blending
    between overlapping chunks to prevent jerk and discontinuity at chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 16,
        action_dim: int = 7,
        blend_mode: str = "receding_horizon",
        blend_alpha: float = 0.5,
    ):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.blend_mode = blend_mode
        self.blend_alpha = blend_alpha
        self.buffer = collections.deque()
        self.step_counter = 0

    def add_chunk(self, action_chunk: Union[np.ndarray, torch.Tensor]):
        """
        Add a newly predicted action chunk of shape [chunk_size, action_dim] or [1, chunk_size, action_dim].
        """
        if isinstance(action_chunk, torch.Tensor):
            chunk = action_chunk.detach().cpu().float().numpy()
        else:
            chunk = np.array(action_chunk, dtype=np.float32)

        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]

        assert chunk.shape == (self.chunk_size, self.action_dim), (
            f"Expected chunk shape ({self.chunk_size}, {self.action_dim}), got {chunk.shape}"
        )

        if self.blend_mode == "receding_horizon" or len(self.buffer) == 0:
            # Overwrite remaining buffer or fill freshly
            self.buffer.clear()
            for action in chunk:
                self.buffer.append(action)
        elif self.blend_mode == "linear_blend":
            # Blend remaining actions with new chunk predictions
            current_actions = list(self.buffer)
            self.buffer.clear()
            num_overlap = min(len(current_actions), len(chunk))
            
            for i in range(num_overlap):
                # Exponential / linear blend between prior prediction and fresh chunk
                weight_new = self.blend_alpha + (1.0 - self.blend_alpha) * (i / max(1, num_overlap))
                blended = (1.0 - weight_new) * current_actions[i] + weight_new * chunk[i]
                self.buffer.append(blended)

            for i in range(num_overlap, len(chunk)):
                self.buffer.append(chunk[i])

    def pop_action(self) -> Optional[np.ndarray]:
        """
        Pops the immediate next action for the robot arm. Returns None if buffer is empty.
        """
        if len(self.buffer) > 0:
            self.step_counter += 1
            return self.buffer.popleft()
        return None

    def remaining_steps(self) -> int:
        """Returns number of remaining actions in the buffer."""
        return len(self.buffer)

    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    def clear(self):
        self.buffer.clear()
        self.step_counter = 0


class EdgeTelemetryHUD:
    """
    Real-time HUD and diagnostic profiler tracking sustained control loop Hz,
    policy step latency, and action bounds.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.latencies = collections.deque(maxlen=window_size)
        self.chunk_latencies = collections.deque(maxlen=window_size)
        self.timestamps = collections.deque(maxlen=window_size)
        self.action_saturation_count = 0
        self.total_steps = 0

    def record_step(self, step_latency_ms: float, action: Optional[np.ndarray] = None):
        now = time.perf_counter()
        self.timestamps.append(now)
        self.latencies.append(step_latency_ms)
        self.total_steps += 1

        if action is not None:
            # Check for saturated actuators (|action| >= 0.99)
            if np.any(np.abs(action) >= 0.99):
                self.action_saturation_count += 1

    def record_chunk_inference(self, chunk_latency_ms: float):
        self.chunk_latencies.append(chunk_latency_ms)

    def get_frequency_hz(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed

    def get_metrics(self) -> Dict[str, float]:
        hz = self.get_frequency_hz()
        mean_step_ms = float(np.mean(self.latencies)) if len(self.latencies) > 0 else 0.0
        p95_step_ms = float(np.percentile(self.latencies, 95)) if len(self.latencies) > 0 else 0.0
        mean_chunk_ms = float(np.mean(self.chunk_latencies)) if len(self.chunk_latencies) > 0 else 0.0
        
        return {
            "control_hz": hz,
            "mean_step_latency_ms": mean_step_ms,
            "p95_step_latency_ms": p95_step_ms,
            "mean_chunk_inference_ms": mean_chunk_ms,
            "total_steps": self.total_steps,
            "saturation_rate": (self.action_saturation_count / max(1, self.total_steps)),
        }

    def render_hud(self) -> str:
        m = self.get_metrics()
        return (
            f"[FastVLA HUD] Rate: {m['control_hz']:5.1f} Hz | "
            f"Step Latency: {m['mean_step_latency_ms']:5.2f} ms (p95: {m['p95_step_latency_ms']:5.2f} ms) | "
            f"Chunk Inf: {m['mean_chunk_inference_ms']:5.1f} ms | "
            f"Steps: {m['total_steps']}"
        )


class FastVLAPolicyRunner:
    """
    Minimal, production-ready inference runner for FastVLA policies.
    Executes continuous action chunk prediction with 2-step Midpoint or Euler ODE solvers.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_id_or_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        action_dim: int = 7,
        chunk_size: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        ode_solver: str = "midpoint",
        ode_steps: int = 2,
    ):
        self.device = device
        self.dtype = dtype
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.ode_solver = ode_solver
        self.ode_steps = ode_steps

        self.buffer = ActionChunkBuffer(chunk_size=chunk_size, action_dim=action_dim)
        self.telemetry = EdgeTelemetryHUD()

        if model is not None:
            self.model = model
        elif model_id_or_path is not None:
            from .model import FastVLAModel
            self.model = FastVLAModel.from_pretrained(
                model_id_or_path,
                load_in_4bit=True if device == "cuda" else False,
            )
        else:
            self.model = None

        if self.model is not None:
            self.model.to(device)
            self.model.eval()

        # Image transforms
        if HAS_TORCHVISION:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = None

    def preprocess_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Converts input image to [1, 1, 3, H, W] normalized tensor on target device.
        """
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))

        if isinstance(image, Image.Image):
            if self.transform is not None:
                tensor = self.transform(image).unsqueeze(0).unsqueeze(0)
            else:
                img_resized = image.resize(self.image_size)
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        elif isinstance(image, torch.Tensor):
            tensor = image
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.ndim == 4:
                tensor = tensor.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return tensor.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        instruction: str = "perform task",
    ) -> np.ndarray:
        """
        Performs policy forward pass and ODE integration to generate continuous 16-step action chunk.
        """
        start_time = time.perf_counter()
        pixel_values = self.preprocess_image(image)

        if self.model is None:
            # Standalone mock generation for edge testing without full weights
            chunk = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            for t in range(self.chunk_size):
                chunk[t, :] = np.sin(0.1 * (t + 1)) * 0.5
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.telemetry.record_chunk_inference(latency_ms)
            return chunk

        # Tokenize prompt if tokenizer exists
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            inputs = tokenizer(instruction, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
        else:
            input_ids = None
            attention_mask = None

        # Check if flow matching action head is available
        action_head = getattr(self.model, "action_head", None)
        from .adapters.action_head import FlowMatchingActionHead

        if isinstance(action_head, FlowMatchingActionHead):
            # Midpoint or Euler continuous flow ODE solve
            pooled_hidden = self.model._encode_sequence(pixel_values, input_ids, attention_mask)
            chunk = self._solve_flow_ode(action_head, pooled_hidden)
        else:
            # Fallback to standard model forward call
            action_pred, _ = self.model(pixel_values=pixel_values, input_ids=input_ids)
            chunk = action_pred.detach().cpu().float().numpy()

        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.telemetry.record_chunk_inference(latency_ms)
        return chunk

    def _solve_flow_ode(
        self,
        action_head: nn.Module,
        pooled_hidden: torch.Tensor,
    ) -> np.ndarray:
        """
        High-speed Midpoint / Euler ODE integration from t=0 (prior noise) to t=1 (action chunk).
        """
        batch_size = pooled_hidden.shape[0]
        # Initial noise sample x_0 ~ N(0, I)
        x_t = torch.randn(
            (batch_size, self.chunk_size, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )

        dt = 1.0 / self.ode_steps
        t = 0.0

        for step in range(self.ode_steps):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=self.dtype)
            v_t = action_head.forward_velocity(x_t, t_tensor, pooled_hidden)

            if self.ode_solver == "midpoint" and step < self.ode_steps:
                # Midpoint step
                half_dt = 0.5 * dt
                x_mid = x_t + half_dt * v_t
                t_mid = t + half_dt
                t_mid_tensor = torch.full((batch_size,), t_mid, device=self.device, dtype=self.dtype)
                v_mid = action_head.forward_velocity(x_mid, t_mid_tensor, pooled_hidden)
                x_t = x_t + dt * v_mid
            else:
                # Standard Euler step
                x_t = x_t + dt * v_t

            t += dt

        return x_t.detach().cpu().float().numpy()

    def step(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        instruction: str = "perform task",
        replan_threshold: int = 4,
    ) -> np.ndarray:
        """
        Rolling-horizon step method. Returns a single action [action_dim] for the robot.
        Triggers policy inference asynchronously or when buffer drops below threshold.
        """
        step_start = time.perf_counter()

        # Re-plan if buffer is nearly exhausted
        if self.buffer.remaining_steps() <= replan_threshold:
            chunk = self.predict_action_chunk(image, instruction)
            self.buffer.add_chunk(chunk)

        action = self.buffer.pop_action()
        if action is None:
            # Buffer empty fallback
            action = np.zeros(self.action_dim, dtype=np.float32)

        step_latency = (time.perf_counter() - step_start) * 1000.0
        self.telemetry.record_step(step_latency, action)
        return action
