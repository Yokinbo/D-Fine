"""多粒度上下文聚合（MGCA）。

大型火电厂在遥感影像中通常需要结合厂区内部结构和较大范围场景上下文才能
与工业园区、密集厂房等相似背景区分。MGCA 在送入 D-FINE 解码器之前，对
P4/P5 特征并行提取局部、区域和长程上下文，并通过逐位置软选择进行聚合。

模块不改变 D-FINE 的匹配策略、损失函数或预测头。输出投影采用零初始化，
首次接入预训练模型时严格保持原特征，之后由原始检测损失学习残差更新。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiGranularityContextAggregation(nn.Module):
    """在指定金字塔层上共享参数地聚合三种空间粒度的上下文。

    Args:
        hidden_dim: 输入特征通道数。
        bottleneck_dim: 多粒度分支使用的低维通道数。
        num_levels: 输入特征金字塔层数。
        target_levels: 需要增强的层索引；D-FINE 三层特征中 ``[1, 2]`` 为 P4/P5。
        norm_groups: GroupNorm 分组数，避免小批量训练时引入新的 BN 统计波动。
        max_residual_scale: 单个特征元素允许的最大残差幅度。
        dropout: 残差分支的二维丢弃率。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        num_levels: int = 3,
        target_levels: Sequence[int] = (1, 2),
        norm_groups: int = 32,
        max_residual_scale: float = 0.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0 or num_levels <= 0:
            raise ValueError("hidden_dim、bottleneck_dim 和 num_levels 必须大于 0")
        if norm_groups <= 0 or hidden_dim % norm_groups != 0:
            raise ValueError("norm_groups 必须大于 0 且能整除 hidden_dim")
        if max_residual_scale <= 0.0:
            raise ValueError("max_residual_scale 必须大于 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 范围内")

        normalized_levels = tuple(int(level) for level in target_levels)
        if not normalized_levels:
            raise ValueError("target_levels 不能为空")
        if len(set(normalized_levels)) != len(normalized_levels):
            raise ValueError("target_levels 不能包含重复索引")
        if any(level < 0 or level >= num_levels for level in normalized_levels):
            raise ValueError(
                f"target_levels 必须位于 [0, {num_levels})，当前为 {normalized_levels}"
            )

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_levels = int(num_levels)
        self.target_levels = normalized_levels
        self.max_residual_scale = float(max_residual_scale)

        # P4/P5 共用一组上下文提取参数，减少开销并提供跨尺度一致的归纳偏置。
        self.input_norm = nn.GroupNorm(norm_groups, hidden_dim)
        self.input_proj = nn.Conv2d(hidden_dim, bottleneck_dim, kernel_size=1)

        # 局部分支：保留邻域纹理与边界线索。
        self.local_branch = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            padding=1,
            groups=bottleneck_dim,
        )
        # 区域分支：可分解大核在较低成本下覆盖厂区组成关系。
        self.regional_branch = nn.Sequential(
            nn.Conv2d(
                bottleneck_dim,
                bottleneck_dim,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=bottleneck_dim,
            ),
            nn.Conv2d(
                bottleneck_dim,
                bottleneck_dim,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=bottleneck_dim,
            ),
        )
        # 上下文锚分支：平滑局部噪声后捕获更长程的水平和垂直场景结构。
        self.context_branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(
                bottleneck_dim,
                bottleneck_dim,
                kernel_size=(1, 11),
                padding=(0, 5),
                groups=bottleneck_dim,
            ),
            nn.Conv2d(
                bottleneck_dim,
                bottleneck_dim,
                kernel_size=(11, 1),
                padding=(5, 0),
                groups=bottleneck_dim,
            ),
        )

        # 每个分支各提取通道均值和最大值，依据三路响应统计逐位置选择粒度。
        self.granularity_selector = nn.Conv2d(6, 3, kernel_size=3, padding=1)
        self.output_proj = nn.Conv2d(bottleneck_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for module in self.modules():
            if isinstance(module, nn.Conv2d) and module not in {
                self.input_proj,
                self.output_proj,
            }:
                # 三条上下文分支内部均为线性深度卷积。按 fan_in 保持前向方差，
                # 避免 groups=bottleneck_dim 时 fan_out 初始化使串联大核响应过小。
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 等权起步，避免初始时人为偏向某一种上下文粒度。
        nn.init.zeros_(self.granularity_selector.weight)
        nn.init.zeros_(self.granularity_selector.bias)
        # 零初始化使模块首次接入时严格等价于未启用 MGCA 的特征路径。
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _refine_level(
        self,
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.input_norm(feature)
        reduced = F.gelu(self.input_proj(normalized))

        local = self.local_branch(reduced)
        regional = self.regional_branch(reduced)
        contextual = self.context_branch(reduced)
        branch_statistics = torch.cat(
            (
                local.mean(dim=1, keepdim=True),
                local.amax(dim=1, keepdim=True),
                regional.mean(dim=1, keepdim=True),
                regional.amax(dim=1, keepdim=True),
                contextual.mean(dim=1, keepdim=True),
                contextual.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        weights = F.softmax(self.granularity_selector(branch_statistics), dim=1)

        fused = (
            weights[:, 0:1] * local
            + weights[:, 1:2] * regional
            + weights[:, 2:3] * contextual
        )
        residual = self.max_residual_scale * torch.tanh(self.output_proj(F.gelu(fused)))
        output = feature + self.dropout(residual)
        return output, weights, residual

    def forward(
        self,
        features: Sequence[torch.Tensor],
        return_details: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]]:
        if len(features) != self.num_levels:
            raise ValueError(
                f"MGCA 配置为 {self.num_levels} 个特征层，实际收到 {len(features)} 个"
            )

        outputs = list(features)
        scale_weights = []
        residual_magnitudes = []
        for level in self.target_levels:
            feature = outputs[level]
            if feature.ndim != 4 or feature.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"第 {level} 层应为 [B, {self.hidden_dim}, H, W]，"
                    f"实际为 {tuple(feature.shape)}"
                )
            outputs[level], weights, residual = self._refine_level(feature)
            scale_weights.append(weights.mean(dim=(0, 2, 3)))
            residual_magnitudes.append(residual.detach().abs().mean())

        if not return_details:
            return outputs

        details = {
            "target_levels": torch.tensor(self.target_levels, device=outputs[0].device),
            "granularity_weights": torch.stack(scale_weights),
            "mean_absolute_residual": torch.stack(residual_magnitudes),
        }
        return outputs, details


__all__ = ["MultiGranularityContextAggregation"]
