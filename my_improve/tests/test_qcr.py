"""Behavioral checks for the optional final-query ranking regularizer.

Run from the repository root with:
    python -m unittest discover -s my_improve/tests -p test_qcr.py -v
"""

import unittest

import torch
import torch.nn.functional as F

from my_improve.qcr import QualityConstrainedRanking


def match_indices(query_ids=(), target_ids=()):
    return (
        torch.tensor(query_ids, dtype=torch.long),
        torch.tensor(target_ids, dtype=torch.long),
    )


def example(logits=(0.4, -0.2), positive_width=0.2, device="cpu", dtype=torch.float32):
    logits = torch.tensor(logits, dtype=dtype, device=device).reshape(1, -1, 1)
    logits.requires_grad_()
    boxes = [[0.5, 0.5, positive_width, 0.2]]
    boxes.extend([[0.1, 0.1, 0.1, 0.1] for _ in range(logits.shape[1] - 1)])
    boxes = torch.tensor([boxes], dtype=dtype, device=device, requires_grad=True)
    targets = [{
        "labels": torch.tensor([0], dtype=torch.long, device=device),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=dtype, device=device),
    }]
    return {"pred_logits": logits, "pred_boxes": boxes}, targets


class QualityConstrainedRankingTests(unittest.TestCase):
    def test_pair_loss_and_gradients_keep_quality_weights_detached(self):
        outputs, targets = example(positive_width=0.12)
        regularizer = QualityConstrainedRanking()
        loss, stats = regularizer(outputs, targets, [match_indices([0], [0])], [match_indices()])
        logits = outputs["pred_logits"]
        positive, negative = logits.detach()[0, :, 0]
        quality = torch.tensor(0.6)
        expected = quality * negative.sigmoid().square() * F.softplus(0.5 + negative - positive)
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-7)

        loss.backward()
        expected_gradient = quality * negative.sigmoid().square() * (0.5 + negative - positive).sigmoid()
        torch.testing.assert_close(logits.grad[0, 0, 0], -expected_gradient, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(logits.grad[0, 1, 0], expected_gradient, rtol=1e-5, atol=1e-7)
        self.assertIsNone(outputs["pred_boxes"].grad)
        for value in stats.values():
            if torch.is_tensor(value):
                self.assertFalse(value.requires_grad)
                self.assertTrue(torch.isfinite(value).all().item())

    def test_matched_union_and_overlapping_queries_are_never_negatives(self):
        outputs, targets = example(logits=(0.4, -0.2, 4.0, 5.0))
        with torch.no_grad():
            outputs["pred_boxes"][0, 3] = targets[0]["boxes"][0]
        protected = [match_indices([2], [0])]
        loss, _ = QualityConstrainedRanking()(outputs, targets, [match_indices([0], [0])], protected)
        loss.backward()
        gradient = outputs["pred_logits"].grad[0, :, 0]
        self.assertLess(gradient[0].item(), 0)
        self.assertGreater(gradient[1].item(), 0)
        self.assertEqual(gradient[2].item(), 0)
        self.assertEqual(gradient[3].item(), 0)

    def test_only_highest_scoring_topk_negatives_receive_gradients(self):
        outputs, targets = example(logits=(0.4, 1.5, 0.5, -1.0))
        loss, _ = QualityConstrainedRanking(topk=2)(
            outputs, targets, [match_indices([0], [0])], [match_indices()]
        )
        loss.backward()
        gradient = outputs["pred_logits"].grad[0, :, 0]
        self.assertGreater(gradient[1].item(), 0)
        self.assertGreater(gradient[2].item(), 0)
        self.assertEqual(gradient[3].item(), 0)

    def test_image_with_gt_but_no_reliable_positive_is_graph_zero(self):
        outputs, targets = example(positive_width=0.08)
        loss, _ = QualityConstrainedRanking()(
            outputs, targets, [match_indices([0], [0])], [match_indices()]
        )
        self.assertEqual(loss.item(), 0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertEqual(outputs["pred_logits"].grad.abs().sum().item(), 0)
        self.assertIsNone(outputs["pred_boxes"].grad)

    def test_no_eligible_negative_is_graph_zero(self):
        outputs, targets = example()
        loss, _ = QualityConstrainedRanking()(
            outputs, targets, [match_indices([0], [0])], [match_indices([1], [0])]
        )
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(outputs["pred_logits"].grad.abs().sum().item(), 0)

    def test_empty_gt_uses_only_topk_negative_bce_with_detached_weights(self):
        outputs, _ = example(logits=(-2.0, 0.0, 1.0))
        targets = [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]
        loss, _ = QualityConstrainedRanking(topk=2)(
            outputs, targets, [match_indices()], [match_indices()]
        )
        selected = outputs["pred_logits"].detach()[0, 1:, 0]
        expected = (selected.sigmoid().square() * F.softplus(selected)).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        gradient = outputs["pred_logits"].grad[0, :, 0]
        self.assertEqual(gradient[0].item(), 0)
        torch.testing.assert_close(gradient[1:], selected.sigmoid().pow(3) / 2)
        self.assertIsNone(outputs["pred_boxes"].grad)

    def test_inactive_image_still_counts_in_batch_mean(self):
        one_outputs, one_targets = example()
        module = QualityConstrainedRanking()
        one_loss, _ = module(one_outputs, one_targets, [match_indices([0], [0])], [match_indices()])
        batched = {key: value.detach().repeat(2, 1, 1).requires_grad_() for key, value in one_outputs.items()}
        with torch.no_grad():
            batched["pred_boxes"][1, 0] = torch.tensor([0.1, 0.1, 0.1, 0.1])
        batch_loss, _ = module(
            batched, one_targets * 2,
            [match_indices([0], [0]), match_indices([0], [0])],
            [match_indices(), match_indices()],
        )
        torch.testing.assert_close(batch_loss, one_loss / 2)

    def test_extreme_logits_remain_finite(self):
        for values in ((-1000.0, 1000.0), (1000.0, -1000.0)):
            with self.subTest(logits=values):
                outputs, targets = example(logits=values)
                loss, _ = QualityConstrainedRanking()(
                    outputs, targets, [match_indices([0], [0])], [match_indices()]
                )
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                self.assertTrue(torch.isfinite(outputs["pred_logits"].grad).all().item())

    def test_multiclass_inputs_are_rejected(self):
        outputs, targets = example()
        outputs["pred_logits"] = outputs["pred_logits"].repeat(1, 1, 2)
        with self.assertRaises(ValueError):
            QualityConstrainedRanking()(outputs, targets, [match_indices([0], [0])], [match_indices()])

    def test_module_adds_no_trainable_parameters(self):
        self.assertEqual(sum(parameter.numel() for parameter in QualityConstrainedRanking().parameters()), 0)

    def test_warmup_is_deterministic_and_resume_uses_epoch_progress(self):
        factor = QualityConstrainedRanking.warmup_factor
        self.assertEqual(factor(0), 0)
        self.assertAlmostEqual(factor(2, step=50, epoch_step=100), 0.5)
        self.assertEqual(factor(5), 1)
        self.assertEqual(factor(86), 1)
        self.assertEqual(factor(0, warmup_epochs=0), 1)
        with self.assertRaises(ValueError):
            factor(1, epoch_step=0)

    def test_invalid_positive_geometry_is_not_a_reliable_match(self):
        outputs, targets = example()
        with torch.no_grad():
            outputs["pred_boxes"][0, 0, 0] = float("nan")
        loss, stats = QualityConstrainedRanking()(
            outputs, targets, [match_indices([0], [0])], [match_indices()]
        )
        self.assertEqual(loss.item(), 0)
        self.assertEqual(stats["positive"].item(), 0)

    def test_invalid_parameters_are_rejected(self):
        for options in ({"topk": 0}, {"topk": 1.5}, {"margin": -1},
                        {"margin": float("nan")}, {"positive_iou": 0.2, "negative_iou": 0.3}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                QualityConstrainedRanking(**options)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is needed for FP16 autocast")
    def test_cuda_amp_forward_and_backward_are_finite(self):
        outputs, targets = example(logits=(-1000.0, 1000.0), device="cuda", dtype=torch.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            loss, _ = QualityConstrainedRanking().cuda()(
                outputs, targets, [match_indices([0], [0])], [match_indices()]
            )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(outputs["pred_logits"].grad).all().item())
        self.assertIsNone(outputs["pred_boxes"].grad)


if __name__ == "__main__":
    unittest.main()
