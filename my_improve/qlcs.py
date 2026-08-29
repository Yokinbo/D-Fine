"""查询引导的隐式部件采样（QLCS）。

火力发电厂属于典型复合目标，但当前数据集只有整厂框，没有烟囱、冷却塔、
煤场等部件标注。QLCS 为每个整厂查询设置若干无监督的潜在部件槽，并在当前
参考框内部从多尺度特征图采样。部件槽、采样位置和聚合权重均由查询自适应
学习，因此不要求额外部件标签。

本文件不依赖 D-FINE 的其他实现，便于单独做张量测试和后续消融实验。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


SpatialShapes = Union[Sequence[Sequence[int]], torch.Tensor]


def _make_component_priors(num_components: int, radius: float = 0.60) -> torch.Tensor:
    """生成参考框内部的初始部件位置，并按“由中心向外”排列。"""
    if num_components <= 0:
        raise ValueError("num_components 必须大于 0")
    if not 0.0 <= radius <= 1.0:
        raise ValueError("radius 必须位于 [0, 1] 范围内")

    side = math.ceil(math.sqrt(num_components))
    if side == 1:
        return torch.zeros(1, 2, dtype=torch.float32)

    axis = torch.linspace(-radius, radius, side, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

    # 先放中心与近中心位置，使 5 槽等非平方数配置仍保持空间对称性。
    distance = points.square().sum(dim=-1)
    tie_break = points[:, 1].abs() * 1e-3 + points[:, 0].abs() * 1e-4
    order = torch.argsort(distance + tie_break)
    return points[order[:num_components]]


class QueryGuidedLatentComponentSampler(nn.Module):
    """在查询参考框内发现并聚合无标注潜在部件。

    Args:
        hidden_dim: 查询和多尺度特征的通道数。
        num_levels: 输入特征层数。
        num_components: 每个整厂查询的潜在部件槽数量。
        attention_dim: 尺度与部件注意力的低维嵌入通道数。
        offset_scale: 查询预测的采样偏移相对参考框宽高的最大比例。
        dropout: 注入整厂查询前的残差丢弃率。
        init_scale: 新分支的初始 LayerScale，较小值可稳定加载预训练模型微调。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_levels: int = 3,
        num_components: int = 5,
        attention_dim: int = 64,
        offset_scale: float = 0.35,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_levels <= 0 or attention_dim <= 0:
            raise ValueError("hidden_dim、num_levels 和 attention_dim 必须大于 0")
        if num_components <= 0:
            raise ValueError("num_components 必须大于 0")
        if offset_scale < 0:
            raise ValueError("offset_scale 不能小于 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 范围内")

        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.num_components = num_components
        self.offset_scale = float(offset_scale)

        # 每个槽是一个可学习的“隐式部件原型”，不对应人工指定的部件类别。
        self.component_embed = nn.Parameter(torch.empty(num_components, hidden_dim))
        # 固定且分散的初始位置避免无监督部件槽全部塌缩到同一点；查询仍可预测残差。
        self.register_buffer("component_priors", _make_component_priors(num_components))

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)

        # 每个部件槽在每个特征层上预测独立二维残差。
        self.offset_proj = nn.Linear(hidden_dim, num_levels * 2)

        # 每个框内部件分别进行查询引导的尺度聚合。
        self.scale_query_proj = nn.Linear(hidden_dim, attention_dim)
        self.scale_key_proj = nn.Linear(hidden_dim, attention_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # 当前模块仅做框内部件的基础平均融合；显式关系与背景门控留给下一项改进。
        # LayerScale 让新增残差分支从较小幅度开始，便于稳定微调预训练模型。
        self.layer_scale = nn.Parameter(torch.full((hidden_dim,), float(init_scale)))
        self.dropout = nn.Dropout(dropout)
        self.attention_scale = attention_dim**-0.5

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.component_embed, std=0.02)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)

        for module in (
            self.scale_query_proj,
            self.scale_key_proj,
            self.value_proj,
            self.output_proj,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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

    def _sampling_locations(
        self,
        normalized_query: torch.Tensor,
        reference_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 [B,Q,K,L,2] 采样位置和 [B,Q,K,C] 部件查询。"""
        batch_size, num_queries, _ = normalized_query.shape
        slot_query = normalized_query.unsqueeze(2) + self.component_embed.view(
            1, 1, self.num_components, self.hidden_dim
        )
        learned_offset = self.offset_proj(slot_query).view(
            batch_size,
            num_queries,
            self.num_components,
            self.num_levels,
            2,
        )
        learned_offset = learned_offset.tanh() * self.offset_scale

        priors = self.component_priors.view(1, 1, self.num_components, 1, 2)
        # 第一项消融只观察参考框内部部件；框外背景环带留给后续关系门控模块。
        relative_locations = (priors + learned_offset).clamp(-1.0, 1.0)
        center = reference_boxes[..., :2].unsqueeze(2).unsqueeze(3)
        size = reference_boxes[..., 2:].clamp_min(1e-6).unsqueeze(2).unsqueeze(3)
        locations = center + 0.5 * size * relative_locations
        return locations.clamp(0.0, 1.0), slot_query

    def _sample_multiscale_memory(
        self,
        memory: torch.Tensor,
        locations: torch.Tensor,
        spatial_shapes: SpatialShapes,
    ) -> torch.Tensor:
        """双线性采样并返回 [B,Q,K,L,C] 多尺度部件特征。"""
        shapes = self._normalize_spatial_shapes(spatial_shapes)
        if len(shapes) != self.num_levels:
            raise ValueError(
                f"QLCS 配置为 {self.num_levels} 个特征层，但实际收到 {len(shapes)} 个"
            )

        split_sizes = [height * width for height, width in shapes]
        if sum(split_sizes) != memory.shape[1]:
            raise ValueError(
                "memory 长度与 spatial_shapes 不一致："
                f"{memory.shape[1]} != {sum(split_sizes)}"
            )

        level_memories = memory.split(split_sizes, dim=1)
        sampled_levels = []
        for level, ((height, width), level_memory) in enumerate(zip(shapes, level_memories)):
            feature = level_memory.transpose(1, 2).reshape(
                memory.shape[0], self.hidden_dim, height, width
            )
            # grid_sample 的网格范围为 [-1, 1]，输出形状为 [B,C,Q,K]。
            grid = locations[:, :, :, level, :].to(dtype=feature.dtype) * 2.0 - 1.0
            sampled = F.grid_sample(
                feature,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            sampled_levels.append(sampled.permute(0, 2, 3, 1))

        return torch.stack(sampled_levels, dim=3)

    def forward(
        self,
        query: torch.Tensor,
        reference_boxes: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: SpatialShapes,
        memory_mask: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """用隐式部件上下文更新整厂查询。

        ``reference_boxes`` 使用归一化 ``(cx, cy, w, h)``。D-FINE 在相邻
        解码层之间会对参考框执行 detach，本模块遵循原有迭代回归语义。
        """
        if query.ndim != 3 or memory.ndim != 3 or reference_boxes.ndim != 3:
            raise ValueError("query、memory 和 reference_boxes 均应为三维张量")
        if query.shape[:2] != reference_boxes.shape[:2] or reference_boxes.shape[-1] != 4:
            raise ValueError("reference_boxes 必须为与 query 对齐的 [B,Q,4] 张量")
        if query.shape[-1] != self.hidden_dim or memory.shape[-1] != self.hidden_dim:
            raise ValueError(f"QLCS 输入通道必须为 hidden_dim={self.hidden_dim}")

        if memory_mask is not None:
            memory = memory * memory_mask.to(memory.dtype).unsqueeze(-1)

        normalized_query = self.query_norm(query)
        locations, slot_query = self._sampling_locations(normalized_query, reference_boxes)
        sampled = self._sample_multiscale_memory(memory, locations, spatial_shapes)
        sampled = self.memory_norm(sampled)

        # 尺度注意力：同一潜在部件在 P3/P4/P5 中选择更合适的语义层级。
        scale_query = self.scale_query_proj(slot_query).unsqueeze(3)
        scale_key = self.scale_key_proj(sampled)
        scale_logits = (scale_query * scale_key).sum(dim=-1) * self.attention_scale
        scale_weights = F.softmax(scale_logits, dim=3)
        component_tokens = (
            scale_weights.unsqueeze(-1) * self.value_proj(sampled)
        ).sum(dim=3)

        # QLCS 的第一阶段消融使用无参数平均聚合，避免提前引入关系门控的贡献。
        context = component_tokens.mean(dim=2)
        update = self.output_proj(context)
        output = query + self.dropout(update) * self.layer_scale

        if not return_details:
            return output

        details = {
            "sampling_locations": locations,
            "scale_weights": scale_weights,
            "component_tokens": component_tokens,
        }
        return output, details
