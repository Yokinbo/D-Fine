"""Proposal-Aligned Denoising: an experimental, training-only DN sampler.

Replace a bounded subset of *positive* reference boxes, not targets, labels,
losses, normal queries, or the negative/padding DN references. No learnable or
persistent state is added. Candidate geometry is always detached and float32.
"""
import math

import torch
from torch import nn


class ProposalAlignedDenoising(nn.Module):
    def __init__(self, max_ratio=0.25, iou_min=0.3, iou_max=0.7,
                 ambiguity_margin=0.05, warmup_epochs=5):
        super().__init__()
        values = (max_ratio, iou_min, iou_max, ambiguity_margin, warmup_epochs)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("PAD parameters must be finite")
        if not 0 <= max_ratio <= 1:
            raise ValueError("PAD max_ratio must be in [0, 1]")
        if not 0 <= iou_min < iou_max <= 1:
            raise ValueError("PAD requires 0 <= iou_min < iou_max <= 1")
        if not 0 <= ambiguity_margin <= 1 or warmup_epochs < 0:
            raise ValueError("Invalid PAD ambiguity margin or warmup")
        self.max_ratio = float(max_ratio)
        self.iou_min = float(iou_min)
        self.iou_max = float(iou_max)
        self.ambiguity_margin = float(ambiguity_margin)
        self.warmup_epochs = float(warmup_epochs)
        self._ratio = None

    def set_progress(self, epoch, step=0, epoch_step=1):
        if not all(math.isfinite(float(value)) for value in (epoch, step, epoch_step)):
            raise ValueError("PAD epoch/step metadata must be finite")
        if epoch < 0 or step < 0 or epoch_step <= 0 or step >= epoch_step:
            raise ValueError("Invalid PAD epoch/step metadata")
        progress = float(epoch) + float(step) / float(epoch_step)
        ramp = 1.0 if self.warmup_epochs == 0 else min(1.0, progress / self.warmup_epochs)
        self._ratio = self.max_ratio * ramp

    @staticmethod
    def _valid_geometry(boxes):
        return (torch.isfinite(boxes).all(-1) & (boxes >= 0).all(-1)
                & (boxes <= 1).all(-1) & (boxes[..., 2:] > 0).all(-1))

    @staticmethod
    def _iou(first, second):
        # Do not clip corners at image boundaries: use the actual reference box.
        first_xyxy = torch.cat((first[:, :2] - first[:, 2:] / 2,
                                first[:, :2] + first[:, 2:] / 2), -1)
        second_xyxy = torch.cat((second[:, :2] - second[:, 2:] / 2,
                                 second[:, :2] + second[:, 2:] / 2), -1)
        intersection = (torch.minimum(first_xyxy[:, None, 2:], second_xyxy[None, :, 2:])
                        - torch.maximum(first_xyxy[:, None, :2], second_xyxy[None, :, :2]))
        intersection = intersection.clamp_min(0).prod(-1)
        union = first[:, 2:].prod(-1)[:, None] + second[:, 2:].prod(-1)[None] - intersection
        return intersection / union.clamp_min(torch.finfo(torch.float32).tiny)

    @staticmethod
    def _inverse_sigmoid(boxes, eps=1e-5):
        # Match src/zoo/dfine/utils.py without importing its package registry.
        boxes = boxes.clamp(min=0, max=1)
        return torch.log(boxes.clamp(min=eps) / (1 - boxes).clamp(min=eps))

    def forward(self, dn_bbox_unact, targets, dn_meta, proposal_boxes, proposal_logits):
        exemplar = dn_bbox_unact if dn_bbox_unact is not None else proposal_boxes
        if exemplar is None:
            exemplar = targets[0]["boxes"] if targets else torch.empty(0)
        keys = ("positive_slots", "requested", "replaced", "candidates", "covered_gt",
                "total_gt", "ambiguous", "invalid_proposals", "ratio", "actual_ratio",
                "original_iou", "replacement_iou", "original_area_ratio", "replacement_area_ratio",
                "original_iou_sum", "replacement_iou_sum", "original_area_ratio_sum",
                "replacement_area_ratio_sum")
        stats = {key: torch.zeros((), device=exemplar.device, dtype=torch.float32) for key in keys}
        if not self.training or self.max_ratio == 0:
            return dn_bbox_unact, stats
        if self._ratio is None:
            raise RuntimeError("PAD training requires set_progress(epoch, step, epoch_step)")
        stats["ratio"].fill_(self._ratio)
        if dn_bbox_unact is None or dn_meta is None or dn_meta.get("dn_positive_idx") is None:
            return dn_bbox_unact, stats
        if dn_bbox_unact.ndim != 3 or dn_bbox_unact.shape[-1] != 4:
            raise ValueError("PAD expects [batch, DN slots, 4] reference logits")
        groups = int(dn_meta["dn_num_group"])
        batch_size, slots, _ = dn_bbox_unact.shape
        positives = dn_meta["dn_positive_idx"]
        if groups < 0 or len(targets) != batch_size or len(positives) != batch_size:
            raise ValueError("PAD DN batch/group metadata mismatch")
        counts = [len(target["labels"]) for target in targets]
        if any(len(target["boxes"]) != count for target, count in zip(targets, counts)):
            raise ValueError("PAD GT boxes/labels counts differ")
        # Validate the standard group-major DN layout, preventing a silent
        # overwrite of negatives or another GT if a caller supplies other meta.
        max_gt = max(counts, default=0)
        for count, indices in zip(counts, positives):
            expected = (torch.arange(groups, device=dn_bbox_unact.device)[:, None] * (2 * max_gt)
                        + torch.arange(count, device=dn_bbox_unact.device)[None]).flatten()
            if indices.ndim != 1 or indices.numel() != groups * count or not torch.equal(indices.to(expected), expected):
                raise ValueError("PAD requires original group-major positive DN indices")
            if expected.numel() and int(expected[-1]) >= slots:
                raise ValueError("PAD positive DN index exceeds reference slots")
        stats["positive_slots"].fill_(sum(counts) * groups)
        stats["total_gt"].fill_(sum(counts))
        budget = math.floor(groups * self._ratio)
        stats["requested"].fill_(sum(counts) * budget)
        if budget == 0 or not sum(counts):
            return dn_bbox_unact, stats
        if proposal_boxes is None or proposal_logits is None:
            raise ValueError("PAD active replacement requires encoder proposal boxes/logits")
        if (proposal_boxes.ndim != 3 or proposal_boxes.shape[0] != batch_size
                or proposal_boxes.shape[-1] != 4 or proposal_logits.ndim != 3
                or proposal_logits.shape[:2] != proposal_boxes.shape[:2]
                or proposal_logits.shape[-1] <= 0):
            raise ValueError("PAD proposal shape mismatch")
        if proposal_boxes.device != dn_bbox_unact.device or proposal_logits.device != dn_bbox_unact.device:
            raise ValueError("PAD proposals and DN references must share a device")
        # Detach locally; do not change the tensors retained for encoder losses.
        proposal_boxes = proposal_boxes.detach().float()
        proposal_logits = proposal_logits.detach().float()
        result = dn_bbox_unact
        for batch, (target, count, positive_indices) in enumerate(zip(targets, counts, positives)):
            if not count:
                continue
            truth = target["boxes"].detach().to(device=dn_bbox_unact.device, dtype=torch.float32)
            labels = target["labels"].detach().to(device=dn_bbox_unact.device, dtype=torch.long)
            if (labels < 0).any() or (labels >= proposal_logits.shape[-1]).any():
                raise ValueError("PAD target class index is outside proposal logit classes")
            valid_gt = self._valid_geometry(truth)
            valid_gt_indices = valid_gt.nonzero(as_tuple=True)[0]
            candidate_valid = (self._valid_geometry(proposal_boxes[batch])
                               & torch.isfinite(proposal_logits[batch]).all(-1))
            stats["invalid_proposals"] += (~candidate_valid).sum()
            candidate_indices = candidate_valid.nonzero(as_tuple=True)[0]
            if not valid_gt_indices.numel() or not candidate_indices.numel():
                continue
            overlap = self._iou(proposal_boxes[batch, candidate_indices], truth[valid_gt])
            best_iou, best_local_gt = overlap.max(dim=1)
            owner = valid_gt_indices[best_local_gt]
            in_range = (best_iou >= self.iou_min) & (best_iou <= self.iou_max)
            if valid_gt_indices.numel() > 1:
                leading = overlap.topk(2, dim=1).values
                ambiguous = (leading[:, 0] - leading[:, 1]) < self.ambiguity_margin
            else:
                ambiguous = torch.zeros_like(in_range)
            stats["ambiguous"] += (in_range & ambiguous).sum()
            eligible = in_range & ~ambiguous
            stats["candidates"] += eligible.sum()
            for gt_index in valid_gt_indices.tolist():
                pool = candidate_indices[eligible & (owner == gt_index)]
                if not pool.numel():
                    continue
                stats["covered_gt"] += 1
                # Class-aware score ordering, without sigmoid saturation or RNG.
                scores = proposal_logits[batch, pool, labels[gt_index]]
                chosen = pool[torch.argsort(scores, descending=True, stable=True)[:budget]]
                num_chosen = chosen.numel()
                group_order = torch.randperm(groups, device=dn_bbox_unact.device)[:num_chosen]
                replaced_slots = positive_indices.to(dn_bbox_unact.device)[gt_index::count][group_order]
                original = dn_bbox_unact[batch, replaced_slots].detach().float().sigmoid()
                replacement = proposal_boxes[batch, chosen]
                current_gt = truth[gt_index:gt_index + 1]
                if result is dn_bbox_unact:
                    result = dn_bbox_unact.clone()
                result[batch, replaced_slots] = self._inverse_sigmoid(replacement).to(dn_bbox_unact.dtype)
                stats["replaced"] += num_chosen
                stats["original_iou"] += self._iou(original, current_gt).sum()
                stats["replacement_iou"] += self._iou(replacement, current_gt).sum()
                gt_area = current_gt[:, 2:].prod(-1).clamp_min(torch.finfo(torch.float32).tiny)
                stats["original_area_ratio"] += (original[:, 2:].prod(-1) / gt_area).sum()
                stats["replacement_area_ratio"] += (replacement[:, 2:].prod(-1) / gt_area).sum()
        normalizer = stats["replaced"].clamp_min(1)
        for key in ("original_iou", "replacement_iou", "original_area_ratio", "replacement_area_ratio"):
            stats[key + "_sum"] = stats[key].clone()
            stats[key] /= normalizer
        stats["actual_ratio"] = stats["replaced"] / stats["positive_slots"].clamp_min(1)
        return result, stats
