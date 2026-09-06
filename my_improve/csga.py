"""Cross-Scale Guided Alignment (CSGA), an experimental FPN upsampler.

DySample-inspired grouped point sampling with high-resolution guidance and an
identity-initialized bounded residual. This is NOT an exact DySample/FreqFusion
reproduction; see CSGA_EXPERIMENT.md for attribution and evaluation boundaries.
Sampling coordinate layout is adapted from tiny-smart/dysample (MIT).
Copyright (c) 2023 Wenze Liu. License: licenses/DySample-MIT.txt.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleGuidedAlignment(nn.Module):
    """Upsample high-level features x2, guided by the lateral low-level features.

At initialization the result and feature gradients equal nearest upsampling.
The gate learns on step one; the offset predictors can learn after it moves.
Offsets are bounded in coarse-feature pixels. The signed residual is not a
convex blend and does not impose a bound on output feature magnitudes.
"""

    def __init__(self, channels=256, groups=4, max_offset=0.25, max_residual=0.5):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("CSGA channels must be a positive integer")
        if not isinstance(groups, int) or groups < 1 or channels % groups:
            raise ValueError("CSGA groups must be positive and divide channels")
        if not math.isfinite(max_offset) or max_offset <= 0:
            raise ValueError("CSGA max_offset must be finite and positive")
        if not math.isfinite(max_residual) or not 0 < max_residual <= 1:
            raise ValueError("CSGA max_residual must be in (0, 1]")
        self.channels = channels
        self.groups = groups
        self.max_offset = float(max_offset)
        self.max_residual = float(max_residual)
        # Do not perturb initialization of subsequent existing encoder/decoder
        # modules, including QLCS/DSQC, when inserting this optional branch.
        with torch.random.fork_rng(devices=[]):
            self.high_offset = nn.Conv2d(channels, 2 * groups * 4, 1)
            self.low_offset = nn.Conv2d(channels, 2 * groups, 1)
        for projection in (self.high_offset, self.low_offset):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.gate = nn.Parameter(torch.zeros(groups))
        self.last_offset_abs = None  # detached training diagnostic, not checkpoint state
        axis = torch.tensor([-0.25, 0.25])
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        # Channel layout: [xy, group, subpixel(y,x)]. Pixel shuffle preserves it.
        init_pos = torch.stack((xx, yy)).reshape(2, 1, 4).repeat(1, groups, 1)
        self.register_buffer("init_pos", init_pos.reshape(1, 2 * groups * 4, 1, 1))

    def sampling_offsets(self, high, low):
        raw = self.high_offset(high) + F.pixel_unshuffle(self.low_offset(low), 2)
        return self.max_offset * raw.float().tanh()

    def sample(self, high, offsets):
        batch, _, height, width = high.shape
        offset = (offsets.float() + self.init_pos.float()).reshape(
            batch, 2, self.groups * 4, height, width
        )
        ys = torch.arange(height, dtype=torch.float32, device=high.device) + 0.5
        xs = torch.arange(width, dtype=torch.float32, device=high.device) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((xx, yy)).reshape(1, 2, 1, height, width)
        normalizer = high.new_tensor([width, height], dtype=torch.float32).reshape(1, 2, 1, 1, 1)
        grid = 2.0 * (base + offset) / normalizer - 1.0
        grid = F.pixel_shuffle(grid.reshape(batch, -1, height, width), 2)
        grid = grid.reshape(batch, 2, self.groups, 2 * height, 2 * width)
        grid = grid.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        # FP32 coordinates/sampling avoid subpixel precision loss and support
        # CPU FP16 inputs. Restore the original feature dtype before fusion.
        sampled = F.grid_sample(
            high.float().reshape(batch * self.groups, -1, height, width), grid,
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        return sampled.reshape(batch, self.channels, 2 * height, 2 * width).to(high.dtype)

    def forward(self, high, low):
        if high.ndim != 4 or low.ndim != 4:
            raise ValueError("CSGA expects BCHW features")
        if high.shape[:2] != low.shape[:2] or high.shape[1] != self.channels:
            raise ValueError("CSGA inputs must have equal batch/channel dimensions")
        if low.shape[-2] != 2 * high.shape[-2] or low.shape[-1] != 2 * high.shape[-1]:
            raise ValueError("CSGA expects an exact x2 lateral feature resolution")
        if high.device != low.device or high.dtype != low.dtype:
            raise ValueError("CSGA inputs must share device and dtype")
        nearest = F.interpolate(high, scale_factor=2.0, mode="nearest")
        offsets = self.sampling_offsets(high, low)
        if self.training:
            self.last_offset_abs = offsets.detach().abs().mean()
        aligned = self.sample(high, offsets)
        mix = (self.max_residual * self.gate.tanh()).repeat_interleave(
            self.channels // self.groups
        ).reshape(1, self.channels, 1, 1).to(high.dtype)
        return nearest + mix * (aligned - nearest)
