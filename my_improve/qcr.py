"""Experimental quality-constrained ranking (QCR), training only.

This is a task-specific pairwise regularizer, not a reproduction of RC-DETR.
The original VFL still anchors absolute confidence. Geometry, sample selection,
and quality/focal weights are detached; shared classification features can still
indirectly change localization during training. No trainable parameters.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou


class QualityConstrainedRanking(nn.Module):
    def __init__(self, topk=8, margin=0.5, positive_iou=0.5, negative_iou=0.3):
        super().__init__()
        if not isinstance(topk, int) or isinstance(topk, bool) or topk < 1:
            raise ValueError("qcr_topk must be a positive integer")
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("qcr_margin must be finite and nonnegative")
        if not 0 <= negative_iou < positive_iou <= 1:
            raise ValueError("QCR requires 0 <= negative_iou < positive_iou <= 1")
        self.topk = topk
        self.margin = margin
        self.positive_iou = positive_iou
        self.negative_iou = negative_iou

    @staticmethod
    def warmup_factor(epoch, step=0, epoch_step=1, warmup_epochs=5):
        if epoch_step <= 0 or warmup_epochs < 0:
            raise ValueError("Invalid QCR warmup schedule")
        if warmup_epochs == 0:
            return 1.0
        progress = float(epoch) + float(step) / float(epoch_step)
        return min(1.0, max(0.0, progress / warmup_epochs))

    def forward(self, outputs, targets, indices, protected_indices):
        logits = outputs["pred_logits"].float()
        if logits.ndim != 3 or logits.shape[-1] != 1:
            raise ValueError("QCR currently supports single-class detection only")
        batch_size, num_queries, _ = logits.shape
        if batch_size == 0 or num_queries == 0:
            raise ValueError("QCR requires a nonempty query batch")
        if not len(targets) == len(indices) == len(protected_indices) == batch_size:
            raise ValueError("QCR batch metadata does not match logits")
        # FP32 reductions also avoid overflow when called inside an AMP context.
        total = logits.sum() * 0.0
        stats = {key: logits.new_zeros(()) for key in (
            "positive", "negative", "pairs", "violations", "protected",
            "ambiguous", "empty_gt", "no_positive",
        )}
        for b, target in enumerate(targets):
            scores = logits[b, :, 0]
            with torch.no_grad():
                matched, gt_indices = indices[b]
                matched = matched.to(device=logits.device, dtype=torch.long)
                gt_indices = gt_indices.to(device=logits.device, dtype=torch.long)
                protected = protected_indices[b][0].to(device=logits.device, dtype=torch.long)
                negative_mask = torch.ones(num_queries, dtype=torch.bool, device=logits.device)
                negative_mask[matched] = False
                negative_mask[protected] = False
                stats["protected"] += (~negative_mask).sum()
                gt_boxes = target["boxes"].detach().to(device=logits.device, dtype=torch.float32)
                if len(gt_boxes):
                    boxes = outputs["pred_boxes"][b].detach().float()
                    overlaps = box_iou(
                        box_convert(boxes, "cxcywh", "xyxy"),
                        box_convert(gt_boxes, "cxcywh", "xyxy"),
                    )
                    finite_matches = torch.isfinite(overlaps[matched, gt_indices])
                    # Invalid geometry is never used as evidence for a negative
                    # or a reliable positive.
                    overlaps = torch.nan_to_num(overlaps, nan=1.0, posinf=1.0, neginf=1.0)
                    max_iou = overlaps.max(dim=1).values
                    stats["ambiguous"] += (negative_mask & (max_iou >= self.negative_iou)).sum()
                    negative_mask &= max_iou < self.negative_iou
                    quality = overlaps[matched, gt_indices]
                    reliable = (quality >= self.positive_iou) & finite_matches
                    positive = matched[reliable]
                    quality = quality[reliable]
                    stats["positive"] += positive.numel()
                    if positive.numel() == 0:
                        stats["no_positive"] += 1
                        continue
                else:
                    stats["empty_gt"] += 1
                candidates = negative_mask.nonzero(as_tuple=True)[0]
                k = min(self.topk, candidates.numel())
                if k == 0:
                    continue
                negative = candidates[scores.detach()[candidates].topk(k).indices]
                negative_weight = scores.detach()[negative].sigmoid().square()
                stats["negative"] += k

            neg_logits = scores[negative]
            if len(gt_boxes) == 0:
                # No foreground to rank against: penalize only selected background.
                total = total + (negative_weight * F.softplus(neg_logits)).mean()
            else:
                delta = self.margin + neg_logits[None, :] - scores[positive, None]
                weights = quality[:, None] * negative_weight[None, :]
                # Average by pair count, NOT weight sum: preserve focal attenuation.
                total = total + (weights * F.softplus(delta)).mean()
                stats["pairs"] += delta.numel()
                stats["violations"] += (delta.detach() > 0).sum()

        raw_loss = total / batch_size
        stats = {key: value.detach() / batch_size for key, value in stats.items()}
        stats["raw"] = raw_loss.detach()
        return raw_loss, stats
