"""Criterion and config compatibility; no datasets or pretrained downloads."""

import copy
import unittest

import torch

from src.core import YAMLConfig
from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.matcher import HungarianMatcher


def make_criterion(**kwargs):
    matcher = HungarianMatcher({"cost_class": 2, "cost_bbox": 5, "cost_giou": 2})
    return DFINECriterion(matcher, {"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2},
                          ["vfl", "boxes"], num_classes=1, **kwargs)


def sample_outputs():
    logits = torch.tensor([[[2.0], [0.2], [-0.8]]], requires_grad=True)
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1],
                           [0.9, 0.9, 0.1, 0.1]]], requires_grad=True)
    normal = {"pred_logits": logits, "pred_boxes": boxes}
    outputs = dict(normal)
    outputs.update(aux_outputs=[copy.deepcopy(normal)], pre_outputs=copy.deepcopy(normal),
                   enc_aux_outputs=[copy.deepcopy(normal)], enc_meta={"class_agnostic": False},
                   up=torch.tensor([0.5]), reg_scale=torch.tensor([4.0]))
    dn = {"pred_logits": logits[:, :1].detach().clone().requires_grad_(),
          "pred_boxes": boxes[:, :1].detach().clone().requires_grad_()}
    outputs.update(dn_outputs=[dn], dn_pre_outputs=copy.deepcopy(dn),
                   dn_meta={"dn_positive_idx": [torch.tensor([0])], "dn_num_group": 1})
    targets = [{"labels": torch.tensor([0]), "boxes": boxes[0, :1].detach().clone()}]
    return outputs, targets


class CriterionQCRTests(unittest.TestCase):
    def test_original_losses_unchanged_and_regularizer_only_on_final_queries(self):
        outputs, targets = sample_outputs()
        original = make_criterion()(copy.deepcopy(outputs), targets)
        candidate = make_criterion(use_qcr=True)
        actual = candidate(outputs, targets, epoch=5)
        self.assertEqual(set(actual) - set(original), {"loss_qcr"})
        for key in original:
            torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)
        self.assertGreater(actual["loss_qcr"].item(), 0)
        self.assertFalse(any(v.requires_grad for v in candidate.qcr_stats.values()))
        actual["loss_qcr"].backward()
        self.assertLess(outputs["pred_logits"].grad[0, 0, 0].item(), 0)
        self.assertIsNone(outputs["pred_boxes"].grad)
        self.assertIsNone(outputs["aux_outputs"][0]["pred_logits"].grad)
        self.assertIsNone(outputs["dn_outputs"][0]["pred_logits"].grad)

    def test_zero_weight_is_exact_bypass_without_epoch_requirement(self):
        outputs, targets = sample_outputs()
        original = make_criterion()(copy.deepcopy(outputs), targets)
        candidate = make_criterion(use_qcr=True, qcr_weight=0)
        actual = candidate(outputs, targets)
        self.assertEqual(set(actual), set(original))
        self.assertEqual(candidate.qcr_stats, {})
        for key in original:
            torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)

    def test_epoch_required_and_warmup_scales_only_new_loss(self):
        outputs, targets = sample_outputs()
        candidate = make_criterion(use_qcr=True)
        with self.assertRaises(ValueError):
            candidate(copy.deepcopy(outputs), targets)
        start = candidate(copy.deepcopy(outputs), targets, epoch=0)
        half = candidate(copy.deepcopy(outputs), targets, epoch=2, step=50, epoch_step=100)
        full = candidate(copy.deepcopy(outputs), targets, epoch=5)
        self.assertEqual(start["loss_qcr"].item(), 0)
        torch.testing.assert_close(half["loss_qcr"], full["loss_qcr"] / 2)
        for key in full.keys() - {"loss_qcr"}:
            torch.testing.assert_close(start[key], full[key], atol=0, rtol=0)

    def test_yaml_keeps_model_and_original_criterion_settings(self):
        reference = YAMLConfig("my_improve/dfine_hgnetv2_m_qlcs_dsqc.yml", num_classes=1)
        candidate = YAMLConfig("my_improve/dfine_hgnetv2_m_qlcs_dsqc_qcr.yml")
        for key in ("DFINE", "DFINETransformer", "HGNetv2", "HybridEncoder", "optimizer"):
            self.assertEqual(reference.yaml_cfg[key], candidate.yaml_cfg[key])
        for key, value in reference.yaml_cfg["DFINECriterion"].items():
            self.assertEqual(candidate.yaml_cfg["DFINECriterion"][key], value)
        self.assertIsNone(reference.criterion.qcr)
        self.assertIsNotNone(candidate.criterion.qcr)
        self.assertEqual(reference.criterion.state_dict(), candidate.criterion.state_dict())


if __name__ == "__main__":
    unittest.main()
