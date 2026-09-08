"""Relative Boundary Alignment (RBA), an experimental training-only regularizer.

Symmetric, GT-relative edge error on final normal-query matches that agree with
D-FINE's GO regression assignment. No score supervision, no shrink-only prior,
no new inference parameters. This is a local adaptation, not a published module
or a guarantee of accuracy gains. See RBA_EXPERIMENT.md for sources/limitations.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RelativeBoundaryAlignment(nn.Module):
    def __init__(self, beta=0.1, min_extent=0.05):
        super().__init__()
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("RBA beta must be positive and finite")
        if not math.isfinite(min_extent) or not 0 < min_extent <= 1:
            raise ValueError("RBA min_extent must be in (0, 1]")
        self.beta = float(beta)
        self.min_extent = float(min_extent)

    @staticmethod
    def warmup_factor(epoch, step=0, epoch_step=1, warmup_epochs=5):
        if epoch_step <= 0 or not math.isfinite(warmup_epochs) or warmup_epochs < 0:
            raise ValueError("Invalid RBA warmup schedule")
        progress = float(epoch) + float(step) / epoch_step
        if not math.isfinite(progress):
            raise ValueError("RBA epoch/step must be finite")
        return 1.0 if warmup_epochs == 0 else min(1.0, max(0.0, progress / warmup_epochs))

    @staticmethod
    def edges(boxes):
        return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                          boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)

    def forward(self, outputs, targets, indices, indices_go, num_boxes):
        boxes = outputs["pred_boxes"]
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("RBA expects [batch, queries, 4] normalized cxcywh")
        if not len(targets) == len(indices) == len(indices_go) == boxes.shape[0]:
            raise ValueError("RBA batch/matching lengths differ")
        if not math.isfinite(float(num_boxes)) or num_boxes <= 0:
            raise ValueError("RBA normalizer must be positive and finite")
        # Explicit float32 geometry even under CUDA autocast. Never clamp output
        # boxes to [0,1]: clipping would hide overflow and cut corrective gradients.
        boxes_fp32 = boxes.float()
        if not torch.isfinite(boxes_fp32).all():
            raise FloatingPointError("Non-finite RBA prediction geometry")
        pred_list, target_list = [], []
        eligible = conflicts = invalid_targets = 0
        for batch, ((query, target), (go_query, go_target)) in enumerate(zip(indices, indices_go)):
            query = query.to(boxes.device, dtype=torch.long)
            target = target.to(boxes.device, dtype=torch.long)
            go_query = go_query.to(boxes.device, dtype=torch.long)
            go_target = go_target.to(boxes.device, dtype=torch.long)
            eligible += query.numel()
            if not query.numel():
                continue
            # One target per query in GO. Filter, do NOT overwrite either matcher.
            agree = ((query[:, None] == go_query[None, :]) &
                     (target[:, None] == go_target[None, :])).any(dim=1)
            conflicts += int((~agree).sum().item())
            truth = targets[batch]["boxes"].detach().to(boxes.device, dtype=torch.float32)[target]
            valid = torch.isfinite(truth).all(dim=1) & (truth[:, 2:] > 0).all(dim=1)
            invalid_targets += int((agree & ~valid).sum().item())
            keep = agree & valid
            pred_list.append(boxes_fp32[batch, query[keep]])
            target_list.append(truth[keep])
        selected = sum(len(x) for x in pred_list)
        zero = boxes_fp32.sum() * 0.0  # keep empty batches graph-connected
        loss = zero
        relative_error = oversize = zero.detach()
        if selected:
            pred, truth = torch.cat(pred_list), torch.cat(target_list)
            norm = truth[:, 2:].clamp_min(self.min_extent).repeat(1, 2)
            residual = (self.edges(pred) - self.edges(truth)) / norm
            loss = F.smooth_l1_loss(residual, torch.zeros_like(residual),
                                    beta=self.beta, reduction="none").mean(dim=-1).sum() / num_boxes
            relative_error = residual.detach().abs().mean()
            oversize = (pred.detach()[:, 2:].prod(dim=1) > truth[:, 2:].prod(dim=1)).float().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite RBA loss")
        stats = {"eligible": boxes_fp32.new_tensor(float(eligible)),
                 "selected": boxes_fp32.new_tensor(float(selected)),
                 "conflicts": boxes_fp32.new_tensor(float(conflicts)),
                 "invalid_targets": boxes_fp32.new_tensor(float(invalid_targets)),
                 "raw": loss.detach(), "relative_edge_error": relative_error,
                 "oversize_fraction": oversize}
        return loss, stats
