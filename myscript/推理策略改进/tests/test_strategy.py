"""Behavior tests for crop-response inference; run without model or GIS packages.

The synthetic detector clips known rectangles to each crop, deliberately giving
partial detections higher confidence than complete detections. Ground truth is
owned by this test double only and never passed to the inference strategy.
"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy import (  # noqa: E402
    Observation,
    StrategyConfig,
    Window,
    boundary_response,
    grid_windows,
    run_strategy,
)


class ClippingDetector:
    """Deterministic, exact visibility detector with recorded real forward calls."""

    def __init__(self, objects, width=1400, height=900):
        self.objects = tuple(objects)
        self.width = width
        self.height = height
        self.calls = []

    def observations(self, window):
        result = []
        for box in self.objects:
            clipped = (
                max(box[0], window.x0, 0),
                max(box[1], window.y0, 0),
                min(box[2], window.x0 + window.size, self.width),
                min(box[3], window.y0 + window.size, self.height),
            )
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                score = 0.78 if clipped == box else 0.97
                result.append(Observation(clipped, score, window))
        return result

    def __call__(self, window):
        self.calls.append(window.key)
        return self.observations(window)


def run_detector(detector, windows, config=None):
    observations = [obs for window in windows for obs in detector.observations(window)]
    return run_strategy(
        observations,
        windows,
        detector,
        detector.width,
        detector.height,
        config or StrategyConfig(),
    )


class BoundaryResponseTests(unittest.TestCase):
    def test_following_crop_edge_and_natural_edge_use_original_coordinates(self):
        first = Observation((300, 200, 612, 400), 0.9, Window(100, 100, 512))
        second = Observation((300, 200, 676, 400), 0.9, Window(164, 100, 512))
        response = boundary_response(first, second, 1400, 900)
        self.assertTrue(response["comparable"])
        self.assertAlmostEqual(response["ratios"][2], 1.0)
        self.assertTrue(response["following"][2])
        self.assertAlmostEqual(response["ratios"][0], 0.0)
        self.assertTrue(response["stable_sides"][0])
        self.assertFalse(response["following"][0])

    def test_unmoving_safe_box_has_stable_sides(self):
        box = (300, 200, 550, 400)
        first = Observation(box, 0.9, Window(100, 100, 512))
        second = Observation(box, 0.9, Window(164, 100, 512))
        response = boundary_response(first, second, 1400, 900)
        self.assertEqual(response["stable_sides"], [True] * 4)
        self.assertFalse(any(response["following"]))

    def test_scale_changes_repeated_crops_and_transformed_views_are_incomparable(self):
        first = Observation((300, 200, 612, 400), 0.9, Window(100, 100, 512))
        variants = (
            replace(first, window=Window(164, 100, 768)),
            replace(first),
            replace(first, window=Window(164, 100, 512), view="hflip"),
        )
        for second in variants:
            with self.subTest(second=second):
                response = boundary_response(first, second, 1400, 900)
                self.assertFalse(response["comparable"])
                self.assertEqual(response["ratios"], [None] * 4)
                self.assertFalse(any(response["following"]))

    def test_real_image_boundary_is_not_a_moving_artificial_edge(self):
        first = Observation((800, 180, 1000, 420), 0.9, Window(488, 0, 512))
        second = Observation((800, 180, 936, 420), 0.9, Window(424, 0, 512))
        response = boundary_response(first, second, 1000, 800)
        self.assertFalse(response["following"][2])
        self.assertFalse(response["stable_sides"][2])


class GridTests(unittest.TestCase):
    def test_offset_grids_cover_borders_remainders_and_small_images(self):
        for width, height in ((1301, 907), (512, 512), (171, 93), (1500, 200)):
            for offset in ((0, 0), (127, 91), (255, 255)):
                with self.subTest(size=(width, height), offset=offset):
                    windows = grid_windows(width, height, 512, 256, offset)
                    self.assertEqual(len(windows), len({w.key for w in windows}))
                    self.assertTrue(windows)
                    # Every integer pixel must belong to at least one crop on
                    # both axes; grid_windows produces a Cartesian product.
                    for axis, length in (("x0", width), ("y0", height)):
                        covered = set()
                        for window in windows:
                            start = getattr(window, axis)
                            self.assertGreaterEqual(start, 0)
                            self.assertLessEqual(start, max(length - 512, 0))
                            covered.update(range(start, min(start + 512, length)))
                        self.assertEqual(len(covered), length)
                    self.assertTrue(any(w.x0 == 0 and w.y0 == 0 for w in windows))

    def test_offsets_change_interior_windows(self):
        normal = {window.key for window in grid_windows(1800, 1300, 512, 256)}
        shifted = {window.key for window in grid_windows(1800, 1300, 512, 256, (80, 120))}
        self.assertNotEqual(normal, shifted)


class InferenceBehaviorTests(unittest.TestCase):
    def assertBoxAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), 4)
        for coordinate, truth in zip(actual, expected):
            self.assertAlmostEqual(coordinate, truth, places=7)

    def test_response_recovers_full_extent_despite_higher_partial_confidence(self):
        full_box = (400, 150, 850, 440)
        detector = ClippingDetector([full_box])
        result = run_detector(detector, [Window(128, 64, 512)])
        self.assertEqual(len(result.predictions), 1)
        prediction = result.predictions[0]
        self.assertBoxAlmostEqual(prediction.box, full_box)
        self.assertIn(prediction.status, {"recovered", "stable"})
        self.assertLessEqual(prediction.refine_count, 3)
        self.assertTrue(any(key[2] > 512 for key in detector.calls))
        self.assertLessEqual(len(detector.calls), 3)

    def test_empty_scene_and_weak_candidate_never_invent_detections(self):
        window = Window(0, 0, 512)
        calls = []

        def empty_detector(crop):
            calls.append(crop.key)
            return []

        empty = run_strategy([], [window], empty_detector, 1200, 800)
        self.assertEqual(empty.predictions, [])
        self.assertEqual(calls, [])
        weak = Observation((300, 120, 512, 380), 0.1, window)
        result = run_strategy([weak], [window], empty_detector, 1200, 800)
        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(tuple(result.predictions[0].box), weak.box)
        self.assertAlmostEqual(result.predictions[0].score, weak.score)
        self.assertEqual(calls, [])
        self.assertNotIn(result.predictions[0].status, {"recovered", "stable"})

    def test_missing_recheck_keeps_original_observation(self):
        window = Window(0, 0, 512)
        seed = Observation((300, 120, 512, 380), 0.9, window)
        result = run_strategy([seed], [window], lambda _: [], 1200, 800)
        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(tuple(result.predictions[0].box), seed.box)
        self.assertNotIn(result.predictions[0].status, {"recovered", "stable"})

    def test_real_image_edge_cannot_be_recovered_by_padding(self):
        detector = ClippingDetector([(900, 180, 1050, 420)], width=1000, height=800)
        result = run_detector(detector, [Window(488, 0, 512)])
        self.assertEqual(len(result.predictions), 1)
        prediction = result.predictions[0]
        self.assertEqual(prediction.status, "image_boundary")
        self.assertLessEqual(prediction.box[2], detector.width)
        self.assertFalse(prediction.trusted_sides[2])
        self.assertTrue(all(x >= 0 and y >= 0 for x, y, _ in detector.calls))

    def test_neighboring_factories_remain_separate(self):
        boxes = [(210, 180, 390, 400), (470, 180, 740, 400)]
        detector = ClippingDetector(boxes)
        result = run_detector(detector, [Window(0, 0, 512), Window(256, 0, 512)])
        self.assertEqual(len(result.predictions), 2)
        for actual, expected in zip(sorted(tuple(p.box) for p in result.predictions), sorted(boxes)):
            self.assertBoxAlmostEqual(actual, expected)

    def test_ambiguous_recheck_does_not_union_candidate_boxes(self):
        source = Window(0, 0, 512)
        seed = Observation((300, 180, 512, 380), 0.9, source)

        def ambiguous_detector(window):
            right = min(window.x0 + window.size, 900)
            return [
                # 两个不同位置假设均与seed相容，彼此IoU低于同窗重复query阈值。
                Observation((300, 165, right, 365), 0.9, window),
                Observation((300, 195, right, 395), 0.9, window),
            ]

        result = run_strategy([seed], [source], ambiguous_detector, 1400, 900)
        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(result.predictions[0].status, "ambiguous")
        self.assertEqual(tuple(result.predictions[0].box), seed.box)

    def test_global_and_target_budgets_and_cache_avoid_repeated_forwards(self):
        detector = ClippingDetector([(400, 80, 1000, 180), (400, 300, 1000, 420)])
        windows = [Window(128, 0, 512), Window(192, 0, 512)]
        config = replace(StrategyConfig(), max_refine_windows=2, max_refines_per_target=1)
        result = run_detector(detector, windows, config)
        self.assertLessEqual(len(detector.calls), 2)
        self.assertEqual(len(detector.calls), len(set(detector.calls)))
        self.assertTrue(set(detector.calls).isdisjoint({window.key for window in windows}))
        self.assertEqual(result.stats["refine_windows"], len(detector.calls))
        self.assertTrue(all(pred.refine_count <= 1 for pred in result.predictions))

    def test_zero_budget_still_exports_candidates(self):
        detector = ClippingDetector([(400, 150, 850, 440)])
        config = replace(StrategyConfig(), max_refine_windows=0)
        result = run_detector(detector, [Window(128, 64, 512)], config)
        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(detector.calls, [])
        self.assertEqual(result.predictions[0].status, "budget_exhausted")

    def test_full_global_budget_still_allows_another_target_to_reuse_cached_crop(self):
        detector = ClippingDetector([(400, 80, 1000, 180), (400, 300, 1000, 420)])
        config = replace(StrategyConfig(), max_refine_windows=1, max_refines_per_target=1)
        result = run_detector(detector, [Window(128, 0, 512)], config)
        self.assertEqual(len(result.predictions), 2)
        self.assertEqual(len(detector.calls), 1)
        self.assertGreaterEqual(result.stats["cache_hits"], 1)
        self.assertEqual(result.stats["refine_windows"], 1)
        self.assertTrue(all(pred.refine_count == 1 for pred in result.predictions))

    def test_ablation_policies_are_reproducible_and_overlap_does_not_recheck(self):
        for policy in ("response", "static", "fixed", "random", "overlap"):
            with self.subTest(policy=policy):
                config = replace(StrategyConfig(), policy=policy, random_seed=19)
                first = ClippingDetector([(400, 150, 850, 440)])
                second = ClippingDetector([(400, 150, 850, 440)])
                windows = [Window(128, 64, 512)]
                result_a = run_detector(first, windows, config)
                result_b = run_detector(second, windows, config)
                self.assertEqual(first.calls, second.calls)
                self.assertEqual(result_a.predictions, result_b.predictions)
                if policy == "overlap":
                    self.assertEqual(first.calls, [])


class ValidationTests(unittest.TestCase):
    def test_integer_budgets_and_empty_base_windows_are_validated(self):
        for name, value in (
            ("max_refines_per_target", math.nan),
            ("max_refine_windows", math.inf),
            ("max_refines_per_target", 1.5),
            ("shift_pixels", math.nan),
        ):
            with self.subTest(field=name, value=value):
                with self.assertRaises(ValueError):
                    replace(StrategyConfig(), **{name: value}).validate()
        for window in (Window(0, 0, 0), Window(-1, 0, 512)):
            with self.subTest(window=window):
                with self.assertRaises(ValueError):
                    run_strategy([], [window], lambda _: [], 1400, 900)

    def test_nonfinite_thresholds_are_rejected(self):
        for name in (
            "max_area_ratio", "min_object_pixels", "stable_tolerance",
            "response_max", "candidate_confidence", "edge_margin_ratio",
        ):
            for value in (math.nan, math.inf):
                with self.subTest(field=name, value=value):
                    with self.assertRaises(ValueError):
                        replace(StrategyConfig(), **{name: value}).validate()

    def test_invalid_image_or_observation_fails_explicitly(self):
        window = Window(0, 0, 512)
        for width, height in ((0, 1000), (1000, -1)):
            with self.subTest(size=(width, height)):
                with self.assertRaises(ValueError):
                    run_strategy([], [window], lambda _: [], width, height)
        malformed = (
            Observation((300, 100, 200, 400), 0.9, window),
            Observation((math.nan, 100, 400, 400), 0.9, window),
            Observation((100, 100, 400, 400), math.inf, window),
            Observation((100, 100, 400, 400), 0.9, Window(0, 0, 0)),
        )
        for observation in malformed:
            with self.subTest(observation=observation):
                with self.assertRaises(ValueError):
                    run_strategy([observation], [observation.window], lambda _: [], 1400, 900)


if __name__ == "__main__":
    unittest.main()
