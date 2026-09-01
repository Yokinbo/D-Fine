"""查询引导的前景—背景对比门控（QFBCG）。

火力发电厂属于大尺度复合目标，工业园区、煤堆、厂房和道路等邻域背景容易
产生较高置信度的误检。QFBCG 以每个 D-FINE 查询的动态参考框为条件，在框内
采样候选目标证据，同时在扩大的框外环带采样与查询最相关的邻域背景候选。
模块使用内—外邻域对比特征对查询执行小幅残差校正，使原有分类头能够更好
地区分完整火电厂与局部相似背景。

该模块不需要部件级标签，可单独开启，也可放在 QLCS 之后形成累计消融。
关闭时不会进入额外采样或计算路径。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


SpatialShapes = Union[Sequence[Sequence[int]], torch.Tensor]


def _make_inner_offsets(num_points: int, radius: float) -> torch.Tensor:
    """生成由中心向外排列的框内采样点，坐标相对于参考框半宽/半高。"""
    if num_points <= 0:
        raise ValueError("num_points 必须大于 0")
    if not 0.0 <= radius <= 1.0:
        raise ValueError("radius 必须位于 [0, 1] 范围内")

    side = math.ceil(math.sqrt(num_points))
    if side == 1:
        return torch.zeros(1, 2, dtype=torch.float32)

    axis = torch.linspace(-radius, radius, side, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    distance = offsets.square().sum(dim=-1)
    tie_break = offsets[:, 1].abs() * 1e-3 + offsets[:, 0].abs() * 1e-4
    return offsets[torch.argsort(distance + tie_break)[:num_points]]


def _make_background_ring(num_points: int, context_scale: float) -> torch.Tensor:
    """在参考框外侧的方形环带上均匀生成背景采样点。"""
    if num_points <= 0:
        raise ValueError("num_points 必须大于 0")
    if context_scale <= 1.0:
        raise ValueError("context_scale 必须大于 1，才能采样到参考框外背景")

    angles = torch.arange(num_points, dtype=torch.float32) * (2.0 * math.pi / num_points)
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    # 用无穷范数归一化到方形边界，保证轴向和对角方向均位于框外环带。
    directions = directions / directions.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return directions * float(context_scale)


class QueryGuidedForegroundBackgroundContrastGate(nn.Module):
    """利用动态参考框内外证据对 D-FINE 查询进行对比式门控更新。

    Args:
        hidden_dim: 查询和多尺度 memory 的通道数。
        num_levels: 多尺度特征层数。
        num_foreground_points: 参考框内部采样点数量。
        num_background_points: 框外环带采样点数量。
        foreground_radius: 框内采样点相对于半宽/半高的最大比例。
        context_scale: 框外环带相对于参考框半宽/半高的扩张比例。
        gate_hidden_dim: 标量关系门控 MLP 的隐藏维度。
        init_scale: 新残差分支的初始 LayerScale。
        dropout: 注入查询前的残差丢弃率。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_levels: int = 3,
        num_foreground_points: int = 5,
        num_background_points: int = 8,
        foreground_radius: float = 0.60,
        context_scale: float = 1.35,
        gate_hidden_dim: int = 96,
        init_scale: float = 0.01,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_levels <= 0 or gate_hidden_dim <= 0:
            raise ValueError("hidden_dim、num_levels 和 gate_hidden_dim 必须大于 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 范围内")

        self.hidden_dim = int(hidden_dim)
        self.num_levels = int(num_levels)
        self.register_buffer(
            "foreground_offsets",
            _make_inner_offsets(num_foreground_points, foreground_radius),
        )
        self.register_buffer(
            "background_offsets",
            _make_background_ring(num_background_points, context_scale),
        )

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.foreground_norm = nn.LayerNorm(hidden_dim)
        self.background_norm = nn.LayerNorm(hidden_dim)
        self.contrast_norm = nn.LayerNorm(hidden_dim)

        # 标量门控决定当前查询需要注入多少前景—背景差分证据。
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.contrast_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_scale = nn.Parameter(torch.full((hidden_dim,), float(init_scale)))
        self.dropout = nn.Dropout(dropout)
        self.attention_scale = hidden_dim**-0.5

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        # 初始门控为 0.5，配合小 LayerScale 平稳接入 COCO 预训练权重。
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.xavier_uniform_(self.contrast_proj.weight)
        nn.init.zeros_(self.contrast_proj.bias)

    @staticmethod
    def _normalize_spatial_shapes(spatial_shapes: SpatialShapes) -> List[Tuple[int, int]]:
        if isinstance(spatial_shapes, torch.Tensor):
            shapes = spatial_shapes.detach().cpu().tolist()
        else:
            shapes = spatial_shapes
        normalized = [(int(shape[0]), int(shape[1])) for shape in shapes]
        if any(height <= 0 or width <= 0 for height, width in normalized):
            raise ValueError("spatial_shapes 中的高和宽必须大于 0")
        return normalized

    @staticmethod
    def _locations(reference_boxes: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        center = reference_boxes[..., :2].unsqueeze(2)
        size = reference_boxes[..., 2:].clamp_min(1e-6).unsqueeze(2)
        locations = center + 0.5 * size * offsets.view(1, 1, -1, 2)
        return locations.clamp(0.0, 1.0)

    def _sample_multiscale_memory(
        self,
        memory: torch.Tensor,
        locations: torch.Tensor,
        spatial_shapes: SpatialShapes,
    ) -> torch.Tensor:
        """双线性采样并返回 [B,Q,P*L,C] 的多尺度证据 token。"""
        shapes = self._normalize_spatial_shapes(spatial_shapes)
        if len(shapes) != self.num_levels:
            raise ValueError(
                f"QFBCG 配置为 {self.num_levels} 个特征层，但实际收到 {len(shapes)} 个"
            )

        split_sizes = [height * width for height, width in shapes]
        if sum(split_sizes) != memory.shape[1]:
            raise ValueError(
                "memory 长度与 spatial_shapes 不一致："
                f"{memory.shape[1]} != {sum(split_sizes)}"
            )

        sampled_levels = []
        for (height, width), level_memory in zip(
            shapes, memory.split(split_sizes, dim=1)
        ):
            feature = level_memory.transpose(1, 2).reshape(
                memory.shape[0], self.hidden_dim, height, width
            )
            grid = locations.to(dtype=feature.dtype) * 2.0 - 1.0
            sampled = F.grid_sample(
                feature,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            sampled_levels.append(sampled.permute(0, 2, 3, 1))

        sampled = torch.stack(sampled_levels, dim=3)
        return sampled.flatten(2, 3)

    def _query_guided_context(
        self,
        normalized_query: torch.Tensor,
        tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.memory_norm(tokens)
        logits = (tokens * normalized_query.unsqueeze(2)).sum(dim=-1) * self.attention_scale
        weights = F.softmax(logits, dim=2)
        context = (weights.unsqueeze(-1) * tokens).sum(dim=2)
        return context, weights

    def forward(
        self,
        query: torch.Tensor,
        reference_boxes: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: SpatialShapes,
        memory_mask: Optional[torch.Tensor] = None,
        component_tokens: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if query.ndim != 3 or reference_boxes.ndim != 3 or memory.ndim != 3:
            raise ValueError("query、reference_boxes 和 memory 均应为三维张量")
        if query.shape[:2] != reference_boxes.shape[:2] or reference_boxes.shape[-1] != 4:
            raise ValueError("reference_boxes 必须为与 query 对齐的 [B,Q,4] 张量")
        if query.shape[-1] != self.hidden_dim or memory.shape[-1] != self.hidden_dim:
            raise ValueError(f"QFBCG 输入通道必须为 hidden_dim={self.hidden_dim}")

        if memory_mask is not None:
            memory = memory * memory_mask.to(memory.dtype).unsqueeze(-1)

        normalized_query = self.query_norm(query)
        foreground_locations = self._locations(reference_boxes, self.foreground_offsets)
        background_locations = self._locations(reference_boxes, self.background_offsets)

        if component_tokens is None:
            foreground_tokens = self._sample_multiscale_memory(
                memory, foreground_locations, spatial_shapes
            )
        else:
            if component_tokens.ndim != 4:
                raise ValueError("component_tokens 应为 [B,Q,K,C] 四维张量")
            if component_tokens.shape[:2] != query.shape[:2]:
                raise ValueError("component_tokens 的 B、Q 维必须与 query 对齐")
            if component_tokens.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"component_tokens 通道必须为 hidden_dim={self.hidden_dim}"
                )
            # 组合消融时直接复用 QLCS 已学习到的框内潜在部件，避免重复采样。
            foreground_tokens = component_tokens
        background_tokens = self._sample_multiscale_memory(
            memory, background_locations, spatial_shapes
        )
        foreground, foreground_weights = self._query_guided_context(
            normalized_query, foreground_tokens
        )
        # 对环带也采用查询引导聚合，使其更关注“最像目标”的邻域背景候选。
        background, background_weights = self._query_guided_context(
            normalized_query, background_tokens
        )

        foreground = self.foreground_norm(foreground)
        background = self.background_norm(background)
        contrast = self.contrast_norm(foreground - background)
        gate_input = torch.cat(
            (normalized_query, foreground, background, contrast), dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input))
        update = self.contrast_proj(contrast) * gate
        output = query + self.dropout(update) * self.layer_scale

        if not return_details:
            return output

        details = {
            "foreground_locations": foreground_locations,
            "background_locations": background_locations,
            "foreground_weights": foreground_weights,
            "background_weights": background_weights,
            "gate": gate,
        }
        return output, details
