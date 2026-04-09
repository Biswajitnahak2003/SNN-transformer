"""
Spiking Neural Network UNet with Bipolar Linear Self-Attention.

Architecture:
  - Encoder: 4 stages of SpikingConvBlock with max-pool downsampling
  - Bottleneck: BipolarLinearAttention (Q(K^T V) linear complexity)
  - Decoder: 4 stages with skip connections + transposed-conv upsampling
  - Output: 1x1 conv -> segmentation logits (num_classes)

All intermediate activations use LIF spiking neurons from snntorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


# ─── Building Blocks ─────────────────────────────────────────────────────────

class SpikingConvBlock(nn.Module):
    """Conv2d -> BN -> LIF  (×2)."""

    def __init__(self, in_ch, out_ch, beta=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.lif1  = snn.Leaky(beta=beta,
                               spike_grad=surrogate.fast_sigmoid(),
                               init_hidden=False)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.lif2  = snn.Leaky(beta=beta,
                               spike_grad=surrogate.fast_sigmoid(),
                               init_hidden=False)

    def init_mem(self, batch_size, channels, h, w, device):
        """Return initial membrane potentials for both LIF neurons."""
        mem1 = torch.zeros(batch_size, channels, h, w, device=device)
        mem2 = torch.zeros(batch_size, channels, h, w, device=device)
        return mem1, mem2

    def forward(self, x, mems=None):
        """
        Args:
            x: (B, C_in, H, W)
            mems: tuple (mem1, mem2) or None
        Returns:
            spk2: output spikes (B, C_out, H, W)
            (mem1, mem2): updated membrane potentials
        """
        B, _, H, W = x.shape
        C = self.conv1.out_channels

        if mems is None:
            mem1, mem2 = self.init_mem(B, C, H, W, x.device)
        else:
            mem1, mem2 = mems

        cur1 = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.bn2(self.conv2(spk1))
        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2, (mem1, mem2)


class BipolarLinearAttention(nn.Module):
    """
    Spike-driven self-attention with:
      - Bipolar encoding: spikes ∈ {-1, +1}
      - Linear complexity: Q @ (K^T @ V)  →  O(N d²) instead of O(N² d)
    """

    def __init__(self, dim, num_heads=4, beta=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

        self.q_lif = snn.Leaky(beta=beta,
                               spike_grad=surrogate.fast_sigmoid(),
                               init_hidden=False)
        self.k_lif = snn.Leaky(beta=beta,
                               spike_grad=surrogate.fast_sigmoid(),
                               init_hidden=False)
        self.v_lif = snn.Leaky(beta=beta,
                               spike_grad=surrogate.fast_sigmoid(),
                               init_hidden=False)

        self.bn = nn.BatchNorm2d(dim)
        self.scale = self.head_dim ** -0.5

    def init_mem(self, batch_size, dim, h, w, device):
        z = torch.zeros(batch_size, dim, h, w, device=device)
        return z.clone(), z.clone(), z.clone()   # q_mem, k_mem, v_mem

    def forward(self, x, mems=None):
        """
        Args:
            x: (B, C, H, W) — spike input from encoder
            mems: (q_mem, k_mem, v_mem) or None
        Returns:
            out: (B, C, H, W) — attention output spikes
            (q_mem, k_mem, v_mem): updated membrane potentials
        """
        B, C, H, W = x.shape
        N = H * W

        if mems is None:
            q_mem, k_mem, v_mem = self.init_mem(B, C, H, W, x.device)
        else:
            q_mem, k_mem, v_mem = mems

        # Project & spike
        q_spk, q_mem = self.q_lif(self.q_proj(x), q_mem)
        k_spk, k_mem = self.k_lif(self.k_proj(x), k_mem)
        v_spk, v_mem = self.v_lif(self.v_proj(x), v_mem)

        # Bipolar encoding: {0,1} -> {-1,+1}
        q = 2.0 * q_spk - 1.0
        k = 2.0 * k_spk - 1.0
        v = 2.0 * v_spk - 1.0

        # Reshape to (B, heads, N, d_k)
        q = q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # Feature map for non-negative attention (elu + 1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Linear attention: Q @ (K^T @ V)
        # K^T @ V : (B, heads, d_k, N) @ (B, heads, N, d_k) = (B, heads, d_k, d_k)
        kv = torch.matmul(k.transpose(-2, -1), v)
        # Q @ KV  : (B, heads, N, d_k) @ (B, heads, d_k, d_k) = (B, heads, N, d_k)
        out = torch.matmul(q, kv) * self.scale

        # Normalise by sum of keys (denominator of linear attention)
        k_sum = k.sum(dim=-2, keepdim=True)          # (B, heads, 1, d_k)
        denom = torch.matmul(q, k_sum.transpose(-2, -1)) + 1e-6  # (B, heads, N, 1)
        out = out / denom

        # Reshape back to (B, C, H, W)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        out = self.bn(self.out_proj(out))

        return out, (q_mem, k_mem, v_mem)


# ─── Spiking UNet ────────────────────────────────────────────────────────────

class SpikingUNet(nn.Module):
    """
    UNet with spiking conv blocks and bipolar linear attention bottleneck.
    Designed for single-timestep forward; the wrapper handles the temporal loop.
    """

    def __init__(self, in_channels=4, num_classes=4,
                 base_channels=32, beta=0.5):
        super().__init__()
        ch = base_channels  # 32

        # Encoder
        self.enc1 = SpikingConvBlock(in_channels, ch, beta)       # -> ch
        self.enc2 = SpikingConvBlock(ch,     ch * 2, beta)        # -> ch*2
        self.enc3 = SpikingConvBlock(ch * 2, ch * 4, beta)        # -> ch*4
        self.enc4 = SpikingConvBlock(ch * 4, ch * 8, beta)        # -> ch*8

        self.pool = nn.MaxPool2d(2)

        # Bottleneck attention
        self.attn = BipolarLinearAttention(ch * 8, num_heads=4, beta=beta)
        self.attn_conv = SpikingConvBlock(ch * 8, ch * 8, beta)

        # Decoder  (skip + upsampled)
        # B (attn_conv) is ch*8. e4 is ch*8.
        self.up4 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        # Input to dec4: up4(ch*4) + e4(ch*8) = ch*12
        self.dec4 = SpikingConvBlock(ch * 12, ch * 4, beta)

        # Output of dec4 is ch*4. e3 is ch*4.
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        # Input to dec3: up3(ch*2) + e3(ch*4) = ch*6
        self.dec3 = SpikingConvBlock(ch * 6, ch * 2, beta)

        # Output of dec3 is ch*2. e2 is ch*2.
        self.up2 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        # Input to dec2: up2(ch) + e2(ch*2) = ch*3
        self.dec2 = SpikingConvBlock(ch * 3, ch, beta)

        # Output of dec2 is ch. e1 is ch.
        self.up1 = nn.ConvTranspose2d(ch, ch // 2, 2, stride=2)
        # Input to dec1: up1(ch//2) + e1(ch) = ch*1.5 -> wait, ch must be divisible by 2.
        # Actually up1(ch//2) is 16, e1(ch) is 32 -> 48 channels.
        self.dec1 = SpikingConvBlock(ch // 2 + ch, ch // 2, beta)

        # Final 1x1 conv (non-spiking — produces logits)
        self.head = nn.Conv2d(ch // 2, num_classes, 1)

    def _enc_shapes(self, H, W):
        """Compute spatial shapes at each encoder stage."""
        shapes = [(H, W)]
        for _ in range(4):
            H, W = H // 2, W // 2
            shapes.append((H, W))
        return shapes

    def init_all_mems(self, B, H, W, device):
        """Initialise membrane potentials for every spiking layer."""
        ch = self.enc1.conv1.out_channels
        shapes = self._enc_shapes(H, W)

        mems = {}
        mems['enc1'] = self.enc1.init_mem(B, ch,      *shapes[0], device)
        mems['enc2'] = self.enc2.init_mem(B, ch * 2,  *shapes[1], device)
        mems['enc3'] = self.enc3.init_mem(B, ch * 4,  *shapes[2], device)
        mems['enc4'] = self.enc4.init_mem(B, ch * 8,  *shapes[3], device)

        bH, bW = shapes[4]
        mems['attn'] = self.attn.init_mem(B, ch * 8, bH, bW, device)
        mems['attn_conv'] = self.attn_conv.init_mem(B, ch * 8, bH, bW, device)

        mems['dec4'] = self.dec4.init_mem(B, ch * 4, *shapes[3], device)
        mems['dec3'] = self.dec3.init_mem(B, ch * 2, *shapes[2], device)
        mems['dec2'] = self.dec2.init_mem(B, ch,     *shapes[1], device)
        mems['dec1'] = self.dec1.init_mem(B, ch // 2, *shapes[0], device)

        return mems

    def forward_one_timestep(self, x, mems):
        """
        Single-timestep forward through the entire UNet.

        Returns:
            logits: (B, num_classes, H, W) — raw logits (non-spiking)
            skips:  list of encoder spike maps for visualisation
            mems:   updated membrane potentials
        """
        # Encoder
        e1, mems['enc1'] = self.enc1(x,             mems['enc1'])
        e2, mems['enc2'] = self.enc2(self.pool(e1), mems['enc2'])
        e3, mems['enc3'] = self.enc3(self.pool(e2), mems['enc3'])
        e4, mems['enc4'] = self.enc4(self.pool(e3), mems['enc4'])

        # Bottleneck
        b = self.pool(e4)
        b, mems['attn'] = self.attn(b, mems['attn'])
        b, mems['attn_conv'] = self.attn_conv(b, mems['attn_conv'])

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._pad_cat(d4, e4)
        d4, mems['dec4'] = self.dec4(d4, mems['dec4'])

        d3 = self.up3(d4)
        d3 = self._pad_cat(d3, e3)
        d3, mems['dec3'] = self.dec3(d3, mems['dec3'])

        d2 = self.up2(d3)
        d2 = self._pad_cat(d2, e2)
        d2, mems['dec2'] = self.dec2(d2, mems['dec2'])

        d1 = self.up1(d2)
        d1 = self._pad_cat(d1, e1)
        d1, mems['dec1'] = self.dec1(d1, mems['dec1'])

        logits = self.head(d1)
        return logits, mems

    @staticmethod
    def _pad_cat(up, skip):
        """Centre-crop or pad `up` to match `skip`, then concatenate."""
        dH = skip.size(2) - up.size(2)
        dW = skip.size(3) - up.size(3)
        up = F.pad(up, [dW // 2, dW - dW // 2,
                        dH // 2, dH - dH // 2])
        return torch.cat([up, skip], dim=1)


# ─── Adaptive-Timestep Wrapper ───────────────────────────────────────────────

class AdaptiveTimestepSNN(nn.Module):
    """
    Wraps SpikingViT and runs it for T timesteps.

    In *fixed* mode (no agent):  every patch runs for T timesteps.
    In *adaptive* mode:  a timestep_map (B, num_patches) specifies how many
                         timesteps each patch should run.
    """

    def __init__(self, in_channels=4, num_classes=4, T=4,
                 img_size=128, patch_size=16, embed_dim=256, depth=6, num_heads=8, beta=0.5):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.vit = SpikingViT(in_channels, num_classes, img_size, patch_size,
                              embed_dim, depth, num_heads, beta=beta)

    def forward(self, x, timestep_map=None):
        """
        Args:
            x: (B, C, H, W) input image (same at every timestep).
            timestep_map: (B, num_patches) int tensor, values in [1, T].
                          None → fixed T for all patches.
        Returns:
            output: (B, num_classes, H, W) averaged logits.
        """
        B, C, H, W = x.shape
        device = x.device

        mems = self.vit.init_all_mems(B, device)
        logit_sum = torch.zeros(B, self.num_classes, H, W, device=device)

        if timestep_map is None:
            # Fixed mode: all patches get T timesteps
            count = torch.full((B, 1, H, W), self.T, device=device, dtype=torch.float32)
            for t in range(self.T):
                logits, mems = self.vit.forward_one_timestep(x, mems)
                logit_sum = logit_sum + logits
        else:
            # Adaptive mode: expand timestep_map to pixels
            timestep_map_pixels = self._expand_timestep_map(timestep_map)  # (B, H, W)
            count = timestep_map_pixels.unsqueeze(1).float()  # (B,1,H,W)
            for t in range(self.T):
                logits, mems = self.vit.forward_one_timestep(x, mems)
                # Mask: patches whose assigned T > current step t
                active_pixels = (timestep_map_pixels > t).unsqueeze(1).float()  # (B,1,H,W)
                logit_sum = logit_sum + logits * active_pixels

        output = logit_sum / count.clamp(min=1.0)
        return output

    def _expand_timestep_map(self, timestep_map):
        """Expand (B, num_patches) to (B, H, W) by repeating per patch."""
        B = timestep_map.shape[0]
        H_p = W_p = int(self.num_patches ** 0.5)
        timestep_map = timestep_map.view(B, H_p, W_p)  # (B, H_p, W_p)
        timestep_map_pixels = F.interpolate(timestep_map.unsqueeze(1).float(),
                                            scale_factor=self.patch_size, mode='nearest').squeeze(1).long()
        return timestep_map_pixels

    def forward_single_timestep(self, x, mems=None):
        """Run exactly 1 timestep — used by the agent for confidence."""
        B, C, H, W = x.shape
        if mems is None:
            mems = self.vit.init_all_mems(B, x.device)
        logits, mems = self.vit.forward_one_timestep(x, mems)
        return logits, mems


# ─── Spiking Vision Transformer ─────────────────────────────────────────────

class SpikingMultiHeadAttention(nn.Module):
    """Multi-head self-attention with spiking projections."""

    def __init__(self, dim, num_heads=8, beta=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)
        self.k_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)
        self.v_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)

        self.scale = self.head_dim ** -0.5

    def init_mem(self, batch_size, num_tokens, device):
        z = torch.zeros(batch_size, num_tokens, self.num_heads * self.head_dim, device=device)
        return z.clone(), z.clone(), z.clone()  # q_mem, k_mem, v_mem

    def forward(self, x, mems=None):
        """
        Args:
            x: (B, N, dim)
            mems: (q_mem, k_mem, v_mem) or None
        Returns:
            out: (B, N, dim)
            (q_mem, k_mem, v_mem)
        """
        B, N, _ = x.shape

        if mems is None:
            q_mem, k_mem, v_mem = self.init_mem(B, N, x.device)
        else:
            q_mem, k_mem, v_mem = mems

        q_spk, q_mem = self.q_lif(self.q_proj(x), q_mem)
        k_spk, k_mem = self.k_lif(self.k_proj(x), k_mem)
        v_spk, v_mem = self.v_lif(self.v_proj(x), v_mem)

        # Bipolar encoding
        q = 2.0 * q_spk - 1.0
        k = 2.0 * k_spk - 1.0
        v = 2.0 * v_spk - 1.0

        # Reshape for heads
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, d)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, heads, N, d)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, dim)
        out = self.out_proj(out)

        return out, (q_mem, k_mem, v_mem)


class SpikingMLP(nn.Module):
    """MLP with spiking neurons."""

    def __init__(self, dim, hidden_dim, beta=0.5):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)

    def init_mem(self, batch_size, num_tokens, device):
        z1 = torch.zeros(batch_size, num_tokens, self.fc1.out_features, device=device)
        z2 = torch.zeros(batch_size, num_tokens, self.fc2.out_features, device=device)
        return z1, z2

    def forward(self, x, mems=None):
        B, N, _ = x.shape
        if mems is None:
            mem1, mem2 = self.init_mem(B, N, x.device)
        else:
            mem1, mem2 = mems

        h, mem1 = self.lif1(self.fc1(x), mem1)
        out, mem2 = self.lif2(self.fc2(h), mem2)
        return out, (mem1, mem2)


# ─── Spiking Vision Transformer ─────────────────────────────────────────────

class SpikingMultiHeadAttention(nn.Module):
    """Multi-head self-attention with spiking projections."""

    def __init__(self, dim, num_heads=8, beta=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)
        self.k_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)
        self.v_lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)

        self.scale = self.head_dim ** -0.5

    def init_mem(self, batch_size, num_tokens, device):
        z = torch.zeros(batch_size, num_tokens, self.num_heads * self.head_dim, device=device)
        return z.clone(), z.clone(), z.clone()  # q_mem, k_mem, v_mem

    def forward(self, x, mems=None):
        """
        Args:
            x: (B, N, dim)
            mems: (q_mem, k_mem, v_mem) or None
        Returns:
            out: (B, N, dim)
            (q_mem, k_mem, v_mem)
        """
        B, N, _ = x.shape

        if mems is None:
            q_mem, k_mem, v_mem = self.init_mem(B, N, x.device)
        else:
            q_mem, k_mem, v_mem = mems

        q_spk, q_mem = self.q_lif(self.q_proj(x), q_mem)
        k_spk, k_mem = self.k_lif(self.k_proj(x), k_mem)
        v_spk, v_mem = self.v_lif(self.v_proj(x), v_mem)

        # Bipolar encoding
        q = 2.0 * q_spk - 1.0
        k = 2.0 * k_spk - 1.0
        v = 2.0 * v_spk - 1.0

        # Reshape for heads
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, d)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, heads, N, d)

        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, dim)
        out = self.out_proj(out)

        return out, (q_mem, k_mem, v_mem)


class SpikingMLP(nn.Module):
    """MLP with spiking neurons."""

    def __init__(self, dim, hidden_dim, beta=0.5):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=False)

    def init_mem(self, batch_size, num_tokens, device):
        z1 = torch.zeros(batch_size, num_tokens, self.fc1.out_features, device=device)
        z2 = torch.zeros(batch_size, num_tokens, self.fc2.out_features, device=device)
        return z1, z2

    def forward(self, x, mems=None):
        B, N, _ = x.shape
        if mems is None:
            mem1, mem2 = self.init_mem(B, N, x.device)
        else:
            mem1, mem2 = mems

        h, mem1 = self.lif1(self.fc1(x), mem1)
        out, mem2 = self.lif2(self.fc2(h), mem2)
        return out, (mem1, mem2)


class TransformerBlock(nn.Module):
    """Transformer block: MSA + MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4, beta=0.5):
        super().__init__()
        self.attn = SpikingMultiHeadAttention(dim, num_heads, beta)
        self.mlp = SpikingMLP(dim, dim * mlp_ratio, beta)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def init_mem(self, batch_size, num_tokens, device):
        attn_mem = self.attn.init_mem(batch_size, num_tokens, device)
        mlp_mem = self.mlp.init_mem(batch_size, num_tokens, device)
        return attn_mem, mlp_mem

    def forward(self, x, mems=None):
        if mems is None:
            attn_mem, mlp_mem = self.init_mem(x.size(0), x.size(1), x.device)
        else:
            attn_mem, mlp_mem = mems

        # MSA
        attn_out, attn_mem = self.attn(self.norm1(x), attn_mem)
        x = x + attn_out

        # MLP
        mlp_out, mlp_mem = self.mlp(self.norm2(x), mlp_mem)
        x = x + mlp_out

        return x, (attn_mem, mlp_mem)


class SpikingViT(nn.Module):
    """
    Spiking Vision Transformer for segmentation.
    Patchify -> Transformer blocks -> Upsample to pixels -> Head.
    """

    def __init__(self, in_channels=4, num_classes=4, img_size=128, patch_size=16,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4, beta=0.5):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, beta)
            for _ in range(depth)
        ])

        # Decoder: simple upsample + conv
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='nearest')
        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def init_all_mems(self, batch_size, device):
        mems = {}
        for i, block in enumerate(self.blocks):
            mems[f'block_{i}'] = block.init_mem(batch_size, self.num_patches, device)
        return mems

    def forward_one_timestep(self, x, mems):
        """
        Single timestep forward.
        x: (B, C, H, W)
        Returns: logits (B, num_classes, H, W), updated mems
        """
        B = x.shape[0]

        # Patchify
        patches = self.patch_embed(x)  # (B, embed_dim, H//p, W//p)
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        patches = patches + self.pos_embed

        # Transformer
        for i, block in enumerate(self.blocks):
            patches, mems[f'block_{i}'] = block(patches, mems[f'block_{i}'])

        # Reshape to spatial
        patches = patches.transpose(1, 2).view(B, self.embed_dim, self.img_size // self.patch_size, self.img_size // self.patch_size)

        # Upsample
        up = self.upsample(patches)  # (B, embed_dim, H, W)

        # Head
        logits = self.head(up)

        return logits, mems


class AdaptiveTimestepSNN(nn.Module):
    """
    Wraps SpikingViT and runs it for T timesteps.

    In *fixed* mode (no agent):  every patch runs for T timesteps.
    In *adaptive* mode:  a timestep_map (B, num_patches) specifies how many
                         timesteps each patch should run.
    """

    def __init__(self, in_channels=4, num_classes=4, T=4,
                 img_size=128, patch_size=16, embed_dim=256, depth=6, num_heads=8, beta=0.5):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.vit = SpikingViT(in_channels, num_classes, img_size, patch_size,
                              embed_dim, depth, num_heads, beta=beta)

    def forward(self, x, timestep_map=None):
        """
        Args:
            x: (B, C, H, W) input image (same at every timestep).
            timestep_map: (B, num_patches) int tensor, values in [1, T].
                          None → fixed T for all patches.
        Returns:
            output: (B, num_classes, H, W) averaged logits.
        """
        B, C, H, W = x.shape
        device = x.device

        mems = self.vit.init_all_mems(B, device)
        logit_sum = torch.zeros(B, self.num_classes, H, W, device=device)

        if timestep_map is None:
            # Fixed mode: all patches get T timesteps
            count = torch.full((B, 1, H, W), self.T, device=device, dtype=torch.float32)
            for t in range(self.T):
                logits, mems = self.vit.forward_one_timestep(x, mems)
                logit_sum = logit_sum + logits
        else:
            # Adaptive mode: expand timestep_map to pixels
            timestep_map_pixels = self._expand_timestep_map(timestep_map)  # (B, H, W)
            count = timestep_map_pixels.unsqueeze(1).float()  # (B,1,H,W)
            for t in range(self.T):
                logits, mems = self.vit.forward_one_timestep(x, mems)
                # Mask: patches whose assigned T > current step t
                active_pixels = (timestep_map_pixels > t).unsqueeze(1).float()  # (B,1,H,W)
                logit_sum = logit_sum + logits * active_pixels

        output = logit_sum / count.clamp(min=1.0)
        return output

    def _expand_timestep_map(self, timestep_map):
        """Expand (B, num_patches) to (B, H, W) by repeating per patch."""
        B = timestep_map.shape[0]
        H_p = W_p = int(self.num_patches ** 0.5)
        timestep_map = timestep_map.view(B, H_p, W_p)  # (B, H_p, W_p)
        timestep_map_pixels = F.interpolate(timestep_map.unsqueeze(1).float(),
                                            scale_factor=self.patch_size, mode='nearest').squeeze(1).long()
        return timestep_map_pixels

    def forward_single_timestep(self, x, mems=None):
        """Run exactly 1 timestep — used by the agent for confidence."""
        B, C, H, W = x.shape
        if mems is None:
            mems = self.vit.init_all_mems(B, x.device)
        logits, mems = self.vit.forward_one_timestep(x, mems)
        return logits, mems


class TransformerBlock(nn.Module):
    """Transformer block: MSA + MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4, beta=0.5):
        super().__init__()
        self.attn = SpikingMultiHeadAttention(dim, num_heads, beta)
        self.mlp = SpikingMLP(dim, dim * mlp_ratio, beta)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def init_mem(self, batch_size, num_tokens, device):
        attn_mem = self.attn.init_mem(batch_size, num_tokens, device)
        mlp_mem = self.mlp.init_mem(batch_size, num_tokens, device)
        return attn_mem, mlp_mem

    def forward(self, x, mems=None):
        if mems is None:
            attn_mem, mlp_mem = self.init_mem(x.size(0), x.size(1), x.device)
        else:
            attn_mem, mlp_mem = mems

        # MSA
        attn_out, attn_mem = self.attn(self.norm1(x), attn_mem)
        x = x + attn_out

        # MLP
        mlp_out, mlp_mem = self.mlp(self.norm2(x), mlp_mem)
        x = x + mlp_out

        return x, (attn_mem, mlp_mem)


class SpikingViT(nn.Module):
    """
    Spiking Vision Transformer for segmentation.
    Patchify -> Transformer blocks -> Upsample to pixels -> Head.
    """

    def __init__(self, in_channels=4, num_classes=4, img_size=128, patch_size=16,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4, beta=0.5):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, beta)
            for _ in range(depth)
        ])

        # Decoder: simple upsample + conv
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='nearest')
        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def init_all_mems(self, batch_size, device):
        mems = {}
        for i, block in enumerate(self.blocks):
            mems[f'block_{i}'] = block.init_mem(batch_size, self.num_patches, device)
        return mems

    def forward_one_timestep(self, x, mems):
        """
        Single timestep forward.
        x: (B, C, H, W)
        Returns: logits (B, num_classes, H, W), updated mems
        """
        B = x.shape[0]

        # Patchify
        patches = self.patch_embed(x)  # (B, embed_dim, H//p, W//p)
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        patches = patches + self.pos_embed

        # Transformer
        for i, block in enumerate(self.blocks):
            patches, mems[f'block_{i}'] = block(patches, mems[f'block_{i}'])

        # Reshape to spatial
        patches = patches.transpose(1, 2).view(B, self.embed_dim, self.img_size // self.patch_size, self.img_size // self.patch_size)

        # Upsample
        up = self.upsample(patches)  # (B, embed_dim, H, W)

        # Head
        logits = self.head(up)

        return logits, mems
