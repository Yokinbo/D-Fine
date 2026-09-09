"""Synthetic PAD sampling tests; never reads or edits experiment datasets."""
import copy
import unittest

import torch

from my_improve.pad import ProposalAlignedDenoising


def fixture(counts=(1,), groups=8, classes=1, device="cpu", dtype=torch.float32):
    batch, maximum = len(counts), max(counts, default=0)
    truth = torch.tensor([[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]], device=device)
    targets = [{"boxes": truth[:count].clone(),
                "labels": torch.arange(count, device=device).remainder(classes)} for count in counts]
    positive = tuple((torch.arange(groups, device=device)[:, None] * (2 * maximum)
                      + torch.arange(count, device=device)[None]).flatten() for count in counts)
    meta = {"dn_positive_idx": positive, "dn_num_group": groups,
            "dn_num_split": [2 * maximum * groups, 300]}
    refs = torch.full((batch, 2 * maximum * groups, 4), 0.1, device=device, dtype=dtype)
    boxes = torch.tensor([[0.25, 0.5, 0.28, 0.28], [0.26, 0.5, 0.28, 0.28],
                          [0.24, 0.5, 0.28, 0.28], [0.75, 0.5, 0.28, 0.28],
                          [0.76, 0.5, 0.28, 0.28], [0.74, 0.5, 0.28, 0.28]], device=device, dtype=dtype)
    boxes = boxes.unsqueeze(0).expand(batch, -1, -1).clone()
    scores = torch.arange(6, device=device, dtype=dtype).view(1, 6, 1).expand(batch, -1, classes).clone()
    return refs, targets, meta, boxes, scores


def changed_slots(before, after):
    return (before != after).any(-1)


class PadTests(unittest.TestCase):
    def sampler(self, **kwargs):
        sampler = ProposalAlignedDenoising(**kwargs)
        sampler.set_progress(5)
        return sampler

    def test_requires_progress_and_has_no_persistent_state(self):
        sampler = ProposalAlignedDenoising()
        with self.assertRaisesRegex(RuntimeError, "set_progress"):
            sampler(*fixture())
        self.assertEqual(list(sampler.parameters()), [])
        self.assertEqual(sampler.state_dict(), {})
        sampler.set_progress(5)
        self.assertEqual(sampler.state_dict(), {})

    def test_invalid_parameters_and_progress(self):
        for args in ({"max_ratio": -0.1}, {"max_ratio": 1.1}, {"max_ratio": float("nan")},
                     {"iou_min": 0.8}, {"iou_max": 1.1}, {"iou_min": 0.7},
                     {"ambiguity_margin": -0.1}, {"warmup_epochs": -1},
                     {"warmup_epochs": float("inf")}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ProposalAlignedDenoising(**args)
        sampler = ProposalAlignedDenoising()
        for progress in ((-1, 0, 1), (1, -1, 1), (1, 1, 1), (1, 0, 0), (float("nan"), 0, 1)):
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                sampler.set_progress(*progress)

    def test_zero_ratio_and_eval_are_identity_without_rng(self):
        data = fixture()
        for sampler in (ProposalAlignedDenoising(max_ratio=0), ProposalAlignedDenoising().eval()):
            state = torch.get_rng_state().clone()
            result, stats = sampler(*data)
            self.assertIs(result, data[0])
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(float(stats["replaced"]), 0)

    def test_warmup_budget_floor(self):
        sampler = ProposalAlignedDenoising()
        data = fixture()
        for epoch, expected in ((0, 0), (1, 0), (2.5, 1), (5, 2), (100, 2)):
            sampler.set_progress(epoch)
            result, stats = sampler(*data)
            self.assertEqual(int(stats["replaced"]), expected)
            self.assertLessEqual(float(stats["actual_ratio"]), 0.25)
        sampler.set_progress(2, 5, 10)
        self.assertEqual(sampler._ratio, 0.125)

    def test_budget_positive_only_input_and_metadata_unchanged(self):
        data = fixture((2, 1, 0))
        before = [data[0].clone(), data[3].clone(), data[4].clone()]
        meta_copy = copy.deepcopy(data[2])
        result, stats = self.sampler()(*data)
        self.assertEqual(int(stats["positive_slots"]), 24)
        self.assertEqual(int(stats["requested"]), 6)
        self.assertEqual(int(stats["replaced"]), 6)
        for key in ("original_iou", "replacement_iou", "original_area_ratio", "replacement_area_ratio"):
            self.assertTrue(torch.allclose(stats[key + "_sum"], stats[key] * stats["replaced"]))
        for batch, count in enumerate((2, 1, 0)):
            changes = changed_slots(data[0], result)[batch]
            allowed = torch.zeros_like(changes)
            allowed[data[2]["dn_positive_idx"][batch]] = True
            self.assertFalse((changes & ~allowed).any())
            self.assertEqual(int(changes.sum()), count * 2)
            for gt in range(count):
                indices = data[2]["dn_positive_idx"][batch][gt::count]
                selected = result[batch, indices][changes[indices]].sigmoid()
                self.assertTrue(((selected[:, 0] < 0.5) if gt == 0 else (selected[:, 0] > 0.5)).all())
        for old, current in zip(before, (data[0], data[3], data[4])):
            self.assertTrue(torch.equal(old, current))
        for old, current in zip(meta_copy["dn_positive_idx"], data[2]["dn_positive_idx"]):
            self.assertTrue(torch.equal(old, current))
        self.assertEqual(meta_copy["dn_num_split"], data[2]["dn_num_split"])

    def test_unique_highest_score_priority(self):
        data = fixture()
        result, stats = self.sampler()(*data)
        selected = result[0, changed_slots(data[0], result)[0]].sigmoid()
        self.assertEqual(selected.shape[0], 2)
        self.assertEqual(torch.unique(selected, dim=0).shape[0], 2)
        for expected in data[3][0, (1, 2)]:
            self.assertTrue(torch.isclose(selected, expected).all(-1).any())
        self.assertEqual(int(stats["candidates"]), 3)
        self.assertEqual(int(stats["covered_gt"]), 1)

    def test_insufficient_candidates_preserves_original_remaining(self):
        data = list(fixture())
        data[3], data[4] = data[3][:, :1], data[4][:, :1]
        result, stats = self.sampler()(*data)
        self.assertEqual(int(stats["replaced"]), 1)
        self.assertEqual(int(changed_slots(data[0], result).sum()), 1)

    def test_class_aware_selection(self):
        data = fixture((2,), classes=2)
        data[4][0, :, 0] = torch.tensor([3, 2, 1, 6, 5, 4])
        data[4][0, :, 1] = torch.tensor([1, 2, 3, 4, 5, 6])
        result, _ = self.sampler()(*data)
        selected = result[0, changed_slots(data[0], result)[0]].sigmoid()
        for expected in data[3][0, (0, 1, 4, 5)]:
            self.assertTrue(torch.isclose(selected, expected).all(-1).any())

    def test_near_tied_gt_ownership_rejected(self):
        data = fixture((2,))
        data[1][0]["boxes"][1] = data[1][0]["boxes"][0] + torch.tensor([0.001, 0, 0, 0])
        state = torch.get_rng_state().clone()
        result, stats = self.sampler()(*data)
        self.assertIs(result, data[0])
        self.assertEqual(int(stats["ambiguous"]), 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_no_candidates_empty_gt_no_dn(self):
        data = list(fixture())
        data[3][:, :, 2:] = 0.01
        state = torch.get_rng_state().clone()
        result, _ = self.sampler()(*data)
        self.assertIs(result, data[0])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        sampler = self.sampler()
        data = fixture((0, 0))
        result, stats = sampler(*data)
        self.assertIs(result, data[0])
        self.assertEqual(float(stats["total_gt"]), 0)
        result, stats = sampler(None, data[1], {"dn_positive_idx": None}, data[3], data[4])
        self.assertIsNone(result)
        self.assertEqual(float(stats["ratio"]), 0.25)
        self.assertTrue(all(float(stats[key]) == 0 for key in stats if key.endswith("_sum")))

    def test_invalid_geometry_skip_and_class_index_fail(self):
        data = fixture()
        data[3][0, 0, 0] = float("nan")
        data[3][0, 1, 2] = -0.2
        data[4][0, 2, 0] = float("inf")
        result, stats = self.sampler()(*data)
        self.assertIs(result, data[0])
        self.assertEqual(int(stats["invalid_proposals"]), 3)
        data = fixture()
        data[1][0]["boxes"][0, 2] = 0
        result, stats = self.sampler()(*data)
        self.assertIs(result, data[0])
        data[1][0]["labels"][0] = 1
        with self.assertRaisesRegex(ValueError, "class index"):
            self.sampler()(*data)

    def test_detached_replacements_stats_no_encoder_gradient(self):
        data = fixture()
        data[0].requires_grad_()
        data[3].requires_grad_()
        data[4].requires_grad_()
        data[1][0]["boxes"].requires_grad_()
        result, stats = self.sampler()(*data)
        result.sum().backward()
        self.assertIsNone(data[3].grad)
        self.assertIsNone(data[4].grad)
        self.assertIsNone(data[1][0]["boxes"].grad)
        changes = changed_slots(data[0], result)
        self.assertTrue((data[0].grad[changes] == 0).all())
        self.assertTrue((data[0].grad[~changes] == 1).all())
        self.assertTrue(all(not value.requires_grad and value.ndim == 0 and value.dtype == torch.float32
                            and torch.isfinite(value) for value in stats.values()))

    def test_preserve_reference_dtype_and_reconstruct_progress(self):
        data = fixture(dtype=torch.float16)
        first, resumed = self.sampler(), ProposalAlignedDenoising()
        resumed.set_progress(5)
        torch.manual_seed(42)
        original_result, original_stats = first(*data)
        torch.manual_seed(42)
        resumed_result, resumed_stats = resumed(*data)
        self.assertTrue(torch.equal(original_result, resumed_result))
        self.assertEqual(original_result.dtype, torch.float16)
        self.assertTrue(all(torch.equal(original_stats[key], resumed_stats[key]) for key in original_stats))

    def test_invalid_dn_layout_rejected(self):
        data = fixture()
        data[2]["dn_positive_idx"][0][0] = 1
        with self.assertRaisesRegex(ValueError, "group-major"):
            self.sampler()(*data)

    def test_corners_not_clipped(self):
        # cx=0 is allowed. Box corners can legitimately extend left of the image.
        data = fixture()
        data[1][0]["boxes"][0, 0] = 0
        data[3][0, :, 0] = 0
        data[3][0, :, 2:] = 0.28
        result, stats = self.sampler()(*data)
        self.assertEqual(int(stats["replaced"]), 2)
        self.assertAlmostEqual(float(stats["replacement_iou"]), 0.2 ** 2 / 0.28 ** 2, places=5)
        selected = result[0, changed_slots(data[0], result)[0]].sigmoid()
        self.assertTrue((selected[:, 0] < 0.001).all())
        self.assertTrue(torch.allclose(selected[:, 2:], torch.full((2, 2), 0.28)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_float32_geometry(self):
        data = fixture((2, 0), device="cuda", dtype=torch.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            result, stats = self.sampler()(*data)
        self.assertEqual(int(stats["replaced"]), 4)
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(all(value.dtype == torch.float32 and torch.isfinite(value) for value in stats.values()))


if __name__ == "__main__":
    unittest.main()
