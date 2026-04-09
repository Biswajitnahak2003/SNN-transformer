"""
Adaptive Timestep Agent — LangGraph-orchestrated pipeline.

Two agent backends:
  1. CNN-based (default for training) — fast, differentiable
  2. Qwen 2.5 1.5B LLM (optional) — per-image threshold recommendation

The LangGraph StateGraph defines the inference flow:
  preprocess → initial_inference → compute_confidence → assign_timesteps
              → adaptive_inference → postprocess
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("[agent] langgraph not installed — using direct pipeline instead.")


# ─── CNN-based Agent (differentiable, for training) ──────────────────────────

class CNNTimestepAgent(nn.Module):
    """
    Lightweight CNN that predicts a per-patch timestep map.

    Input:  4-channel image + initial SNN output (num_classes channels)
    Output: per-patch timestep assignments in [1, T]
    """

    def __init__(self, in_channels=4, num_classes=4, T=4, patch_size=16):
        super().__init__()
        self.T = T
        self.patch_size = patch_size
        total_in = in_channels + num_classes  # image + t=1 predictions

        self.net = nn.Sequential(
            nn.Conv2d(total_in, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(None),  # keep spatial dims
            nn.Conv2d(32, 1, 1),         # single-channel confidence
        )

    def forward(self, image, initial_logits, threshold=0.3):
        """
        Args:
            image:          (B, 4, H, W)
            initial_logits: (B, num_classes, H, W) from t=1 SNN pass
            threshold:      confidence threshold (lower = more foreground)

        Returns:
            timestep_map: (B, num_patches) int tensor in [1, T]
            confidence:   (B, 1, H//p, W//p) raw confidence scores per patch
            soft_timestep: (B, H//p, W//p) — differentiable version
        """
        x = torch.cat([image, initial_logits.detach()], dim=1)
        confidence = torch.sigmoid(self.net(x))  # (B, 1, H, W)

        # Patch-level pooling
        conf_patches = F.avg_pool2d(confidence, self.patch_size, self.patch_size)  # (B, 1, H//p, W//p)

        # High confidence (close to 1) → background → fewer timesteps
        # Low confidence (close to 0) → foreground → more timesteps
        soft_timestep = 1.0 + (self.T - 1.0) * (1.0 - conf_patches.squeeze(1))
        # Quantize for actual execution (straight-through estimator)
        timestep_map = soft_timestep.round().clamp(1, self.T).long().flatten(1)  # (B, num_patches)

        return timestep_map, conf_patches, soft_timestep.squeeze(1)


# ─── Hybrid Uncertainty Computation ─────────────────────────────────────────

def compute_gradient_magnitude(image):
    """Compute Sobel gradient magnitude of the input image."""
    # Average across modality channels first
    gray = image.mean(dim=1, keepdim=True)  # (B, 1, H, W)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=image.dtype, device=image.device
                           ).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=image.dtype, device=image.device
                           ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # Normalise to [0, 1]
    B = grad_mag.size(0)
    for b in range(B):
        gmax = grad_mag[b].max()
        if gmax > 0:
            grad_mag[b] = grad_mag[b] / gmax

    return grad_mag.squeeze(1)  # (B, H, W)


def compute_entropy(logits):
    """Compute per-pixel entropy of prediction probabilities."""
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=1)  # (B, H, W)

    # Normalise
    max_entropy = np.log(logits.size(1))
    return entropy / max_entropy  # in [0, 1]


def compute_hybrid_uncertainty(logits, image, alpha=0.6, beta=0.4):
    """
    Hybrid uncertainty = α * entropy + β * gradient_magnitude.
    High uncertainty → needs more timesteps (foreground / boundaries).
    """
    entropy = compute_entropy(logits)
    grad_mag = compute_gradient_magnitude(image)
    return alpha * entropy + beta * grad_mag  # (B, H, W)


# ─── Optional Qwen LLM Agent ────────────────────────────────────────────────

class QwenThresholdAgent:
    """
    Uses Qwen 2.5 1.5B-Instruct to dynamically recommend a per-image
    threshold for the CNN agent based on image statistics.

    This is NOT used during gradient-based training (non-differentiable).
    Use for inference-time threshold tuning or as a demonstration.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_name = model_name

    def load(self):
        """Lazy-load the LLM (call once before use)."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[QwenAgent] Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
        except Exception:
            # Fallback without 4-bit if bitsandbytes unavailable
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        print("[QwenAgent] Model loaded.")

    def recommend_threshold(self, image, initial_logits):
        """
        Analyse image statistics and recommend a threshold.

        Args:
            image: (4, H, W) single image tensor
            initial_logits: (num_classes, H, W) from t=1

        Returns:
            threshold: float in [0, 1]
        """
        self.load()

        # Compute statistics
        img_np = image.cpu().numpy()
        zero_frac = (np.abs(img_np) < 0.01).mean()
        mean_int = img_np[img_np != 0].mean() if (img_np != 0).any() else 0
        std_int = img_np[img_np != 0].std() if (img_np != 0).any() else 0

        entropy = compute_entropy(initial_logits.unsqueeze(0)).squeeze(0)
        ent_mean = entropy.mean().item()
        ent_max = entropy.max().item()

        grad = compute_gradient_magnitude(image.unsqueeze(0)).squeeze(0)
        grad_mean = grad.mean().item()

        prompt = f"""You are an adaptive timestep agent for a Spiking Neural Network processing brain MRI slices.

Image Statistics:
- Background (near-zero) fraction: {zero_frac:.3f}
- Mean non-zero intensity: {mean_int:.3f}
- Std non-zero intensity: {std_int:.3f}
- Prediction entropy: mean={ent_mean:.3f}, max={ent_max:.3f}
- Gradient magnitude: mean={grad_mean:.3f}

Based on these statistics, recommend an optimal threshold (0.0 to 1.0) for separating background from foreground.
- Lower threshold → more pixels run full timesteps (higher accuracy, slower)
- Higher threshold → fewer pixels run full timesteps (faster, risk missing detail)

For this brain MRI slice, the optimal threshold is:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=20, temperature=0.1,
                do_sample=False
            )
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Parse threshold from response
        try:
            threshold = float(response.split()[0].strip('.,'))
            threshold = max(0.05, min(0.95, threshold))
        except (ValueError, IndexError):
            threshold = 0.3  # fallback default
            print(f"[QwenAgent] Could not parse '{response}', using default {threshold}")

        return threshold


# ─── LangGraph Pipeline ─────────────────────────────────────────────────────

def build_agent_graph(snn_model, cnn_agent, T=4, use_llm=False,
                      qwen_agent=None):
    """
    Build a LangGraph StateGraph for the adaptive timestep inference pipeline.

    Nodes:
      preprocess → initial_inference → compute_confidence
      → assign_timesteps → adaptive_inference

    Returns:
        compiled graph (callable with state dict)
    """
    if not HAS_LANGGRAPH:
        return None

    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Optional, Any

    class PipelineState(TypedDict, total=False):
        image: Any             # (B, 4, H, W) tensor
        gradient_map: Any      # (B, H, W) gradient magnitude
        initial_logits: Any    # (B, C, H, W) from t=1
        uncertainty: Any       # (B, H, W) hybrid uncertainty
        threshold: float       # agent threshold
        timestep_map: Any      # (B, H, W)
        confidence: Any        # (B, 1, H, W)
        final_output: Any      # (B, C, H, W)
        energy_saved: float

    def preprocess(state):
        image = state['image']
        grad = compute_gradient_magnitude(image)
        return {'gradient_map': grad}

    def initial_inference(state):
        image = state['image']
        with torch.no_grad():
            logits, _ = snn_model.forward_single_timestep(image)
        return {'initial_logits': logits}

    def compute_confidence_node(state):
        logits = state['initial_logits']
        image = state['image']
        unc = compute_hybrid_uncertainty(logits, image)

        threshold = 0.3
        if use_llm and qwen_agent is not None:
            threshold = qwen_agent.recommend_threshold(
                image[0], logits[0]
            )
        return {'uncertainty': unc, 'threshold': threshold}

    def assign_timesteps(state):
        image = state['image']
        logits = state['initial_logits']
        threshold = state.get('threshold', 0.3)
        t_map, conf, _ = cnn_agent(image, logits, threshold=threshold)
        return {'timestep_map': t_map, 'confidence': conf}

    def adaptive_inference(state):
        image = state['image']
        t_map = state['timestep_map']
        output = snn_model(image, timestep_map=t_map)

        # Energy calculation
        total_pixels = t_map.numel()
        baseline = total_pixels * T
        actual = t_map.float().sum().item()
        saved = 1.0 - actual / baseline if baseline > 0 else 0.0

        return {'final_output': output, 'energy_saved': saved}

    # Build graph
    builder = StateGraph(PipelineState)
    builder.add_node("preprocess", preprocess)
    builder.add_node("initial_inference", initial_inference)
    builder.add_node("compute_confidence", compute_confidence_node)
    builder.add_node("assign_timesteps", assign_timesteps)
    builder.add_node("adaptive_inference", adaptive_inference)

    builder.set_entry_point("preprocess")
    builder.add_edge("preprocess", "initial_inference")
    builder.add_edge("initial_inference", "compute_confidence")
    builder.add_edge("compute_confidence", "assign_timesteps")
    builder.add_edge("assign_timesteps", "adaptive_inference")
    builder.add_edge("adaptive_inference", END)

    return builder.compile()


# ─── Direct Pipeline (no LangGraph dependency) ──────────────────────────────

class AdaptiveTimestepPipeline:
    """
    Direct (non-LangGraph) pipeline for training integration.
    Equivalent logic, but avoids LangGraph overhead per batch.
    """

    def __init__(self, snn_model, cnn_agent, T=4):
        self.snn_model = snn_model
        self.cnn_agent = cnn_agent
        self.T = T

    @torch.no_grad()
    def compute_timestep_map(self, image, threshold=0.3):
        """
        Run one SNN timestep, then let the CNN agent decide per-patch timesteps.

        Returns:
            timestep_map: (B, H, W)
            confidence:   (B, 1, H, W)
            soft_timestep: (B, H, W) — differentiable version
        """
        logits, _ = self.snn_model.forward_single_timestep(image)
        t_map, conf, soft_t = self.cnn_agent(image, logits, threshold)
        return t_map, conf, soft_t

    def forward(self, image, threshold=0.3):
        """Full adaptive forward pass."""
        t_map, conf, soft_t = self.compute_timestep_map(image, threshold)
        output = self.snn_model(image, timestep_map=t_map)
        return output, t_map, conf
