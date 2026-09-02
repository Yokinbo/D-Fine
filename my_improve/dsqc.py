"""解码稳定性感知查询校准（DSQC）。

大型复合火电厂与工业园区、厂房和煤场等背景外观接近，仅依赖最后一层
分类置信度时，少量在解码过程中持续漂移的查询仍可能形成高分误检。
DSQC 不再把参考框外区域直接当作背景，而是比较相邻解码层中同一查询的
语义表示和预测框是否稳定，并据此对最终分类 logits 做有界残差校准。

模块只接收分类损失的监督，框稳定性特征全部停止梯度，因此不会让分类
校准分支反向修改 D-FINE 的 FDR 回归路径。最后一层初始化为零，加载官方
预训练权重时等价于原模型，随后由当前数据集学习校准方向。
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _aligned_box_iou(
    current_boxes: torch.Tensor,
    previous_boxes: torch.Tensor,
) -> torch.Tensor:
    """计算逐查询对齐框的 IoU，输入格式均为归一化 cxcywh。"""
    current_center, current_size = current_boxes[..., :2], current_boxes[..., 2:]
    previous_center, previous_size = previous_boxes[..., :2], previous_boxes[..., 2:]

    current_half = 0.5 * current_size.clamp_min(1e-6)
    previous_half = 0.5 * previous_size.clamp_min(1e-6)
    current_xyxy = torch.cat(
        (current_center - current_half, current_center + current_half), dim=-1
    )
    previous_xyxy = torch.cat(
        (previous_center - previous_half, previous_center + previous_half), dim=-1
    )

    intersection_lt = torch.maximum(current_xyxy[..., :2], previous_xyxy[..., :2])
    intersection_rb = torch.minimum(current_xyxy[..., 2:], previous_xyxy[..., 2:])
    intersection = (intersection_rb - intersection_lt).clamp_min(0.0).prod(dim=-1)
    current_area = current_size.clamp_min(0.0).prod(dim=-1)
    previous_area = previous_size.clamp_min(0.0).prod(dim=-1)
    union = current_area + previous_area - intersection
    return (intersection / union.clamp_min(1e-6)).clamp(0.0, 1.0)


class DecoderStabilityQueryCalibrator(nn.Module):
    """依据相邻解码层的语义和定位稳定性校准分类 logits。

    Args:
        hidden_dim: 解码查询通道数。
        bottleneck_dim: 三路稳定性编码的瓶颈维度。
        max_logit_adjustment: 单个查询允许施加的最大 logit 调整幅度。
        dropout: 校准头中的丢弃率。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        max_logit_adjustment: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_dim 和 bottleneck_dim 必须大于 0")
        if max_logit_adjustment <= 0.0:
            raise ValueError("max_logit_adjustment 必须大于 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 范围内")

        self.hidden_dim = int(hidden_dim)
        self.max_logit_adjustment = float(max_logit_adjustment)
        self.current_norm = nn.LayerNorm(hidden_dim)
        self.previous_norm = nn.LayerNorm(hidden_dim)

        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        # 7维显式信号：语义余弦、框IoU、中心位移2维、尺度变化2维、当前置信度。
        self.stability_proj = nn.Sequential(
            nn.Linear(7, bottleneck_dim),
            nn.GELU(),
        )
        self.calibration_head = nn.Sequential(
            nn.LayerNorm(bottleneck_dim * 3),
            nn.Linear(bottleneck_dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.content_proj[0],
            self.delta_proj[0],
            self.stability_proj[0],
            self.calibration_head[1],
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # 零初始化保证接入官方预训练权重时，首次前向与原始分类结果一致。
        nn.init.zeros_(self.calibration_head[-1].weight)
        nn.init.zeros_(self.calibration_head[-1].bias)

    def forward(
        self,
        logits: torch.Tensor,
        current_query: torch.Tensor,
        previous_query: torch.Tensor,
        current_boxes: torch.Tensor,
        previous_boxes: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if logits.ndim != 3 or current_query.ndim != 3 or previous_query.ndim != 3:
            raise ValueError("logits、current_query 和 previous_query 均应为三维张量")
        if current_boxes.ndim != 3 or previous_boxes.ndim != 3:
            raise ValueError("current_boxes 和 previous_boxes 均应为三维张量")
        if current_query.shape != previous_query.shape:
            raise ValueError("相邻解码层的查询形状必须一致")
        if current_query.shape[:2] != logits.shape[:2]:
            raise ValueError("查询的 B、Q 维必须与 logits 对齐")
        if current_boxes.shape != previous_boxes.shape or current_boxes.shape[-1] != 4:
            raise ValueError("相邻解码层的框必须是形状一致的 [B,Q,4] 张量")
        if current_boxes.shape[:2] != logits.shape[:2]:
            raise ValueError("预测框的 B、Q 维必须与 logits 对齐")
        if current_query.shape[-1] != self.hidden_dim:
            raise ValueError(f"DSQC 查询通道必须为 hidden_dim={self.hidden_dim}")

        current = self.current_norm(current_query)
        previous = self.previous_norm(previous_query.detach())
        semantic_cosine = F.cosine_similarity(current, previous, dim=-1).clamp(-1.0, 1.0)

        # 几何信号仅用作分类校准条件，不让分类损失直接影响回归分支。
        current_boxes_detached = current_boxes.detach()
        previous_boxes_detached = previous_boxes.detach()
        box_iou = _aligned_box_iou(current_boxes_detached, previous_boxes_detached)
        previous_size = previous_boxes_detached[..., 2:].clamp_min(1e-4)
        center_motion = (
            (current_boxes_detached[..., :2] - previous_boxes_detached[..., :2]).abs()
            / previous_size
        ).clamp_max(4.0)
        scale_motion = torch.log(
            current_boxes_detached[..., 2:].clamp_min(1e-4) / previous_size
        ).abs().clamp_max(4.0)
        confidence = logits.detach().sigmoid().amax(dim=-1, keepdim=True)

        stability_signals = torch.cat(
            (
                semantic_cosine.unsqueeze(-1),
                box_iou.unsqueeze(-1),
                center_motion,
                scale_motion,
                confidence,
            ),
            dim=-1,
        )
        fused = torch.cat(
            (
                self.content_proj(current),
                self.delta_proj(current - previous),
                self.stability_proj(stability_signals),
            ),
            dim=-1,
        )

        # 稳定查询保持原分数；校准幅度随语义和几何不稳定性平滑增大。
        semantic_stability = 0.5 * (semantic_cosine + 1.0)
        joint_stability = (semantic_stability * box_iou).clamp(0.0, 1.0)
        instability = (1.0 - joint_stability).unsqueeze(-1)
        raw_adjustment = torch.tanh(self.calibration_head(fused))
        adjustment = self.max_logit_adjustment * instability * raw_adjustment
        calibrated_logits = logits + adjustment

        if not return_details:
            return calibrated_logits

        details = {
            "semantic_cosine": semantic_cosine,
            "box_iou": box_iou,
            "instability": instability,
            "adjustment": adjustment,
        }
        return calibrated_logits, details

