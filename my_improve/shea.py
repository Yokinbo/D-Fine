"""显著—整体证据对齐（SHEA）。

火电厂是由烟囱、冷却塔、厂房等多个部件共同构成的大型复合目标。工业背景
中的单个局部结构可能产生较高分类响应，但通常缺少完整厂区的组合证据。
SHEA 复用 QLCS 在参考框内部得到的潜在部件 token，同时提取查询相关的显著
证据和全部部件的整体证据，并仅对送入分类头的查询特征进行小幅残差细化。

该模块不新增采样位置，不修改回归分支、匹配策略、损失函数或最终 logits。
条件特征均停止梯度，避免新增分类分支反向扰动已经冻结的 QLCS 与回归路径；
输出投影采用零初始化，首次接入时严格等价于未启用 SHEA。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SalientHolisticEvidenceAlignment(nn.Module):
    """用框内显著证据和整体证据细化分类查询。

    Args:
        hidden_dim: 查询及 QLCS 部件 token 的通道数。
        bottleneck_dim: 证据对齐空间的通道数。
        attention_temperature: 显著部件软选择的温度。
        max_residual_scale: 分类查询中单个元素允许的最大残差幅度。
        dropout: 分类残差的丢弃率。
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        attention_temperature: float = 0.7,
        max_residual_scale: float = 0.15,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_dim 和 bottleneck_dim 必须大于 0")
        if attention_temperature <= 0.0:
            raise ValueError("attention_temperature 必须大于 0")
        if max_residual_scale <= 0.0:
            raise ValueError("max_residual_scale 必须大于 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 范围内")

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.attention_temperature = float(attention_temperature)
        self.max_residual_scale = float(max_residual_scale)

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.component_norm = nn.LayerNorm(hidden_dim)

        # 查询与部件共享投影，使二者处于同一证据空间并控制参数开销。
        self.evidence_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.fusion_norm = nn.LayerNorm(bottleneck_dim * 4)
        self.fusion_proj = nn.Sequential(
            nn.Linear(bottleneck_dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        gate_hidden_dim = max(8, bottleneck_dim // 2)
        # 四个显式信号依次描述查询—显著、查询—整体、显著—整体的一致性，
        # 以及显著部件注意力的归一化熵。
        self.alignment_gate = nn.Sequential(
            nn.Linear(4, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.output_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.residual_dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in (
            self.evidence_proj,
            self.fusion_proj[0],
            self.alignment_gate[0],
            self.alignment_gate[2],
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # 只将最后投影置零：保持初始等价，同时让它在第一步即可获得梯度。
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = F.normalize(first.float(), dim=-1, eps=1e-6)
        second = F.normalize(second.float(), dim=-1, eps=1e-6)
        return (first * second).sum(dim=-1).clamp(-1.0, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        component_tokens: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if query.ndim != 3 or component_tokens.ndim != 4:
            raise ValueError("query 应为 [B,Q,C]，component_tokens 应为 [B,Q,K,C]")
        if component_tokens.shape[:2] != query.shape[:2]:
            raise ValueError("component_tokens 的 B、Q 维必须与 query 对齐")
        if query.shape[-1] != self.hidden_dim or component_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"SHEA 输入通道必须为 hidden_dim={self.hidden_dim}")
        if component_tokens.shape[2] <= 0:
            raise ValueError("component_tokens 至少需要一个潜在部件")

        # 保留 query 的恒等梯度路径，但阻断新增分支对 QLCS/回归路径的额外梯度。
        normalized_query = self.query_norm(query.detach())
        normalized_components = self.component_norm(component_tokens.detach())
        query_evidence = self.evidence_proj(normalized_query)
        component_evidence = self.evidence_proj(normalized_components)

        # 在 AMP 下仍用 FP32 计算相似度与 softmax，避免小温度造成数值波动。
        query_key = F.normalize(query_evidence.float(), dim=-1, eps=1e-6)
        component_keys = F.normalize(component_evidence.float(), dim=-1, eps=1e-6)
        attention_logits = (
            (component_keys * query_key.unsqueeze(2)).sum(dim=-1)
            / self.attention_temperature
        )
        attention_weights = F.softmax(attention_logits, dim=2).to(component_evidence.dtype)

        salient_evidence = (
            attention_weights.unsqueeze(-1) * component_evidence
        ).sum(dim=2)
        holistic_evidence = component_evidence.mean(dim=2)
        evidence_gap = (salient_evidence - holistic_evidence).abs()
        evidence_interaction = salient_evidence * holistic_evidence

        component_count = component_tokens.shape[2]
        if component_count > 1:
            attention_entropy = -(
                attention_weights.float()
                * attention_weights.float().clamp_min(1e-8).log()
            ).sum(dim=2) / math.log(component_count)
        else:
            attention_entropy = torch.zeros_like(attention_logits[..., 0])

        alignment_signals = torch.stack(
            (
                self._cosine(query_evidence, salient_evidence),
                self._cosine(query_evidence, holistic_evidence),
                self._cosine(salient_evidence, holistic_evidence),
                attention_entropy.clamp(0.0, 1.0),
            ),
            dim=-1,
        ).to(query_evidence.dtype)

        fused = torch.cat(
            (
                salient_evidence,
                holistic_evidence,
                evidence_gap,
                evidence_interaction,
            ),
            dim=-1,
        )
        fused = self.fusion_proj(self.fusion_norm(fused))
        gate = torch.sigmoid(self.alignment_gate(alignment_signals))
        residual = (
            self.max_residual_scale
            * gate
            * torch.tanh(self.output_proj(fused))
        )
        output = query + self.residual_dropout(residual)

        if not return_details:
            return output

        details = {
            "attention_weights": attention_weights,
            "alignment_signals": alignment_signals,
            "gate": gate,
            "residual": residual,
        }
        return output, details


__all__ = ["SalientHolisticEvidenceAlignment"]
