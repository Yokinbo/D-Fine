"""质量感知查询竞争门控（QACG）。

DETR 类检测器会让多个对象查询竞争同一目标。QACG 在最终解码层比较查询间的
预测框重叠、语义相似度和置信度优势，只对存在更强竞争者的查询学习有界残差
校准，从而减少重复或低质量高分预测。模块沿用 D-FINE 原有训练损失，不引入
额外损失项，也不修改框回归结果。
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_box_iou(boxes: torch.Tensor) -> torch.Tensor:
    """计算每张影像内归一化 ``cxcywh`` 框的两两 IoU。"""
    center = boxes[..., :2]
    half_size = 0.5 * boxes[..., 2:].clamp_min(1e-6)
    boxes_xyxy = torch.cat((center - half_size, center + half_size), dim=-1)

    intersection_lt = torch.maximum(
        boxes_xyxy[:, :, None, :2], boxes_xyxy[:, None, :, :2]
    )
    intersection_rb = torch.minimum(
        boxes_xyxy[:, :, None, 2:], boxes_xyxy[:, None, :, 2:]
    )
    intersection = (intersection_rb - intersection_lt).clamp_min(0.0).prod(dim=-1)

    area = (boxes_xyxy[..., 2:] - boxes_xyxy[..., :2]).clamp_min(0.0).prod(dim=-1)
    union = area[:, :, None] + area[:, None, :] - intersection
    return (intersection / union.clamp_min(1e-6)).clamp(0.0, 1.0)


class QualityAwareCompetitiveQueryGate(nn.Module):
    """利用查询间竞争关系校准分类 logits。

    Args:
        hidden_dim: 解码查询通道数。
        bottleneck_dim: 查询与竞争信号的编码维度。
        overlap_threshold: 形成有效竞争关系所需的最低框 IoU。
        max_logit_adjustment: 单个查询的最大 logit 调整幅度。
        dropout: 校准头中的丢弃率。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        overlap_threshold: float = 0.30,
        max_logit_adjustment: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_dim 和 bottleneck_dim 必须大于 0")
        if not 0.0 <= overlap_threshold < 1.0:
            raise ValueError("overlap_threshold 必须位于 [0, 1) 范围内")
        if max_logit_adjustment <= 0.0:
            raise ValueError("max_logit_adjustment 必须大于 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 范围内")

        self.hidden_dim = int(hidden_dim)
        self.overlap_threshold = float(overlap_threshold)
        self.max_logit_adjustment = float(max_logit_adjustment)

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
        )
        # 当前置信度、最大重叠、更强查询重叠、语义竞争、置信度差距。
        self.competition_proj = nn.Sequential(
            nn.Linear(5, bottleneck_dim),
            nn.GELU(),
        )
        self.calibration_head = nn.Sequential(
            nn.LayerNorm(bottleneck_dim * 2),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.query_proj[0],
            self.competition_proj[0],
            self.calibration_head[1],
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # 零初始化保证首次接入时分类结果与 QLCS+DSQC 完全一致。
        nn.init.zeros_(self.calibration_head[-1].weight)
        nn.init.zeros_(self.calibration_head[-1].bias)

    def forward(
        self,
        logits: torch.Tensor,
        queries: torch.Tensor,
        boxes: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if logits.ndim != 3 or queries.ndim != 3 or boxes.ndim != 3:
            raise ValueError("logits、queries 和 boxes 均应为三维张量")
        if logits.shape[:2] != queries.shape[:2] or logits.shape[:2] != boxes.shape[:2]:
            raise ValueError("logits、queries 和 boxes 的 B、Q 维必须一致")
        if queries.shape[-1] != self.hidden_dim:
            raise ValueError(f"QACG 查询通道必须为 hidden_dim={self.hidden_dim}")
        if boxes.shape[-1] != 4:
            raise ValueError("boxes 最后一维必须为 4")

        normalized_queries = self.query_norm(queries)
        with torch.no_grad():
            confidence = logits.sigmoid().amax(dim=-1)
            pairwise_iou = _pairwise_box_iou(boxes.detach())
            semantic_similarity = torch.bmm(
                F.normalize(normalized_queries.detach(), dim=-1),
                F.normalize(normalized_queries.detach(), dim=-1).transpose(1, 2),
            ).clamp_min(0.0)

            query_count = logits.shape[1]
            diagonal = torch.eye(
                query_count, dtype=torch.bool, device=logits.device
            ).unsqueeze(0)
            pairwise_iou = pairwise_iou.masked_fill(diagonal, 0.0)
            semantic_similarity = semantic_similarity.masked_fill(diagonal, 0.0)

            # 行 i 表示当前查询，列 j 表示竞争查询；只保留置信度更高的 j。
            score_advantage = (
                confidence[:, None, :] - confidence[:, :, None]
            ).clamp_min(0.0)
            valid_overlap = (
                (pairwise_iou - self.overlap_threshold)
                / max(1.0 - self.overlap_threshold, 1e-6)
            ).clamp(0.0, 1.0)
            stronger_affinity = valid_overlap * score_advantage

            max_overlap = pairwise_iou.max(dim=-1).values
            stronger_overlap = (valid_overlap * (score_advantage > 0)).max(dim=-1).values
            semantic_competition = (
                stronger_affinity * semantic_similarity
            ).max(dim=-1).values
            confidence_gap = score_advantage.max(dim=-1).values
            competition_strength = torch.maximum(
                stronger_overlap, semantic_competition
            ).clamp(0.0, 1.0)

            competition_signals = torch.stack(
                (
                    confidence,
                    max_overlap,
                    stronger_overlap,
                    semantic_competition,
                    confidence_gap,
                ),
                dim=-1,
            )

        fused = torch.cat(
            (
                self.query_proj(normalized_queries),
                self.competition_proj(competition_signals),
            ),
            dim=-1,
        )
        raw_adjustment = torch.tanh(self.calibration_head(fused))
        adjustment = (
            self.max_logit_adjustment
            * competition_strength.unsqueeze(-1)
            * raw_adjustment
        )
        calibrated_logits = logits + adjustment

        if not return_details:
            return calibrated_logits

        details = {
            "max_overlap": max_overlap,
            "stronger_overlap": stronger_overlap,
            "semantic_competition": semantic_competition,
            "competition_strength": competition_strength,
            "adjustment": adjustment,
        }
        return calibrated_logits, details

