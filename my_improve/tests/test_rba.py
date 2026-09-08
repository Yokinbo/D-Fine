"""RBA geometry, gradient isolation and original-criterion compatibility."""
import copy
import unittest

import torch

from my_improve.rba import RelativeBoundaryAlignment
from my_improve.tests.test_qcr_integration import make_criterion, sample_outputs
from src.core import YAMLConfig


def fixture(box=(0.5, 0.5, 0.4, 0.4), truth=(0.5, 0.5, 0.2, 0.2)):
    boxes = torch.tensor([[box, (0.1, 0.1, 0.1, 0.1)]], requires_grad=True)
    logits = torch.ones(1, 2, 1, requires_grad=True)
    targets = [{"boxes": torch.tensor([truth], requires_grad=True), "labels": torch.tensor([0])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return {"pred_boxes": boxes, "pred_logits": logits}, targets, indices


class RBATests(unittest.TestCase):
    def test_exact_optimum_zero(self):
        outputs, targets, indices = fixture(box=(0.5, 0.5, 0.2, 0.2))
        loss, _ = RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(outputs["pred_boxes"].grad.abs().sum().item(), 0)

    def test_oversized_box_shrinks_not_center(self):
        outputs, targets, indices = fixture()
        loss, stats = RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)
        loss.backward()
        grad = outputs["pred_boxes"].grad
        self.assertTrue((grad[0, 0, 2:] > 0).all())
        self.assertTrue((grad[0, 0, :2] == 0).all())
        self.assertTrue((grad[0, 1] == 0).all())
        self.assertIsNone(outputs["pred_logits"].grad)
        self.assertIsNone(targets[0]["boxes"].grad)
        self.assertFalse(any(v.requires_grad for v in stats.values()))

    def test_undersized_box_expands(self):
        outputs, targets, indices = fixture(box=(0.5, 0.5, 0.1, 0.1))
        loss, _ = RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)
        loss.backward()
        self.assertTrue((outputs["pred_boxes"].grad[0, 0, 2:] < 0).all())

    def test_translation_correction(self):
        outputs, targets, indices = fixture(box=(0.7, 0.5, 0.2, 0.2))
        loss, _ = RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)
        loss.backward()
        self.assertGreater(outputs["pred_boxes"].grad[0, 0, 0].item(), 0)

    def test_out_of_image_predictions_keep_gradients(self):
        outputs, targets, indices = fixture(box=(0.8, 0.5, 1.8, 0.4))
        loss, _ = RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)
        loss.backward()
        self.assertGreater(outputs["pred_boxes"].grad[0, 0, 2].item(), 0)

    def test_empty_batch_is_graph_connected(self):
        outputs, targets, _ = fixture()
        targets[0]["boxes"] = torch.empty(0, 4)
        empty = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        loss, stats = RelativeBoundaryAlignment()(outputs, targets, empty, empty, 1)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(stats["selected"].item(), 0)
        self.assertEqual(outputs["pred_boxes"].grad.abs().sum().item(), 0)

    def test_conflicting_go_assignment_is_skipped(self):
        outputs, targets, indices = fixture()
        go = [(torch.tensor([0]), torch.tensor([1]))]
        loss, stats = RelativeBoundaryAlignment()(outputs, targets, indices, go, 1)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(stats["conflicts"].item(), 1)

    def test_invalid_gt_is_skipped(self):
        outputs, targets, indices = fixture(truth=(0.5, 0.5, 0.0, 0.2))
        loss, stats = RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(stats["invalid_targets"].item(), 1)

    def test_small_gt_floor_bounds_gradient(self):
        outputs, targets, indices = fixture(truth=(0.5, 0.5, 1e-7, 1e-7))
        loss, _ = RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)
        loss.backward()
        self.assertTrue(torch.isfinite(outputs["pred_boxes"].grad).all())
        self.assertLessEqual(outputs["pred_boxes"].grad.abs().max().item(), 20)

    def test_normalizer(self):
        outputs, targets, indices = fixture()
        module = RelativeBoundaryAlignment()
        a, _ = module(outputs, targets, indices, indices, 1)
        b, _ = module(outputs, targets, indices, indices, 2)
        torch.testing.assert_close(a / 2, b)

    def test_bad_parameters_and_nonfinite_predictions(self):
        for kwargs in ({"beta": 0}, {"beta": float("nan")}, {"min_extent": 0}, {"min_extent": 2}):
            with self.assertRaises(ValueError):
                RelativeBoundaryAlignment(**kwargs)
        outputs, targets, indices = fixture(box=(float("nan"), 0.5, 0.2, 0.2))
        with self.assertRaises(FloatingPointError):
            RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_half_autocast_geometry_fp32(self):
        outputs, targets, indices = fixture()
        outputs["pred_boxes"] = outputs["pred_boxes"].detach().cuda().half().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = RelativeBoundaryAlignment()(outputs, targets, indices, indices, 1)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(outputs["pred_boxes"].grad).all())


class RBAIntegrationTests(unittest.TestCase):
    def test_original_losses_and_aux_dn_are_unchanged(self):
        outputs, targets = sample_outputs()
        outputs["pred_boxes"] = outputs["pred_boxes"].detach().clone()
        outputs["pred_boxes"][0, 0, 2:] *= 1.5
        outputs["pred_boxes"].requires_grad_()
        original = make_criterion()(copy.deepcopy(outputs), targets)
        criterion = make_criterion(use_rba=True)
        actual = criterion(outputs, targets, epoch=5)
        self.assertEqual(set(actual) - set(original), {"loss_rba"})
        for key in original:
            torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)
        self.assertGreater(actual["loss_rba"].item(), 0)
        actual["loss_rba"].backward()
        self.assertIsNone(outputs["pred_logits"].grad)
        self.assertIsNone(outputs["dn_outputs"][0]["pred_boxes"].grad)
        self.assertIsNone(outputs["aux_outputs"][0]["pred_boxes"].grad)
        self.assertGreater(outputs["pred_boxes"].grad.abs().sum().item(), 0)

    def test_zero_weight_and_eval_bypass(self):
        outputs, targets = sample_outputs()
        original = make_criterion()(copy.deepcopy(outputs), targets)
        for criterion in (make_criterion(use_rba=True, rba_weight=0), make_criterion(use_rba=True).eval()):
            actual = criterion(copy.deepcopy(outputs), targets)
            self.assertEqual(set(actual), set(original))
            for key in original:
                torch.testing.assert_close(actual[key], original[key], atol=0, rtol=0)
            self.assertEqual(criterion.rba_stats, {})

    def test_warmup(self):
        factor = RelativeBoundaryAlignment.warmup_factor
        self.assertEqual(factor(0), 0)
        self.assertEqual(factor(2, 50, 100), 0.5)
        self.assertEqual(factor(5), 1)
        outputs, targets = sample_outputs()
        with self.assertRaises(ValueError):
            make_criterion(use_rba=True)(outputs, targets)

    def test_yaml_original_network_and_criterion_retained(self):
        base = YAMLConfig("my_improve/dfine_hgnetv2_m_dsqc.yml", num_classes=1)
        cfg = YAMLConfig("my_improve/dfine_hgnetv2_m_dsqc_rba.yml")
        for key in ("weight_dict", "losses"):
            self.assertEqual(base.yaml_cfg["DFINECriterion"][key], cfg.yaml_cfg["DFINECriterion"][key])
        self.assertIsNotNone(cfg.criterion.rba)
        self.assertIsNone(cfg.criterion.qcr)
        self.assertEqual(base.criterion.state_dict(), cfg.criterion.state_dict())
        self.assertTrue(cfg.yaml_cfg["DFINETransformer"]["use_dsqc"])
        for key in ("use_qlcs", "use_qfbcg", "use_qacg", "use_mgca", "use_shea"):
            self.assertFalse(cfg.yaml_cfg["DFINETransformer"][key])
        only = YAMLConfig("my_improve/dfine_hgnetv2_m_rba.yml")
        self.assertFalse(only.yaml_cfg["DFINETransformer"]["use_dsqc"])
        self.assertIsNotNone(only.criterion.rba)


if __name__ == "__main__":
    unittest.main()
