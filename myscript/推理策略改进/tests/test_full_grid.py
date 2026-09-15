"""整幅重叠网格回归；避免仅用一个初始窗口掩盖片段自竞争问题。"""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy import Observation, StrategyConfig, Window, grid_windows, run_strategy


class GridDetector:
    """真框仅由测试替身持有；被截断的观测刻意比分辨完整的观测分数高。"""

    def __init__(self, boxes, *, duplicate_query=False, width=1800, height=1400):
        self.boxes = boxes
        self.duplicate_query = duplicate_query
        self.width, self.height = width, height
        self.calls = []

    def observe(self, window):
        observations = []
        for box in self.boxes:
            visible = (max(box[0], window.x0, 0), max(box[1], window.y0, 0),
                       min(box[2], window.x0 + window.size, self.width),
                       min(box[3], window.y0 + window.size, self.height))
            if visible[2] <= visible[0] or visible[3] <= visible[1]:
                continue
            score = 0.78 if visible == box else 0.97
            observations.append(Observation(visible, score, window))
            if self.duplicate_query and visible[2] - visible[0] > 4:
                repeated = (visible[0] + 0.2, visible[1], visible[2], visible[3])
                observations.append(Observation(repeated, 0.07, window))
        return observations

    def __call__(self, window):
        self.calls.append(window.key)
        return self.observe(window)


def infer_grid(detector, config=None):
    config = config or StrategyConfig()
    windows = grid_windows(detector.width, detector.height, config.base_size, config.stride, config.grid_offset)
    observations = [observation for window in windows for observation in detector.observe(window)]
    return run_strategy(observations, windows, detector, detector.width, detector.height, config)


class FullGridRegressionTests(unittest.TestCase):
    def assert_boxes(self, result, expected):
        actual = sorted(tuple(round(value, 5) for value in item.box) for item in result.predictions)
        self.assertEqual(actual, sorted(expected))

    def test_one_factory_across_full_grid_has_one_complete_output(self):
        box = (400, 150, 850, 440)
        result = infer_grid(GridDetector([box]))
        self.assert_boxes(result, [box])
        self.assertIn(result.predictions[0].status, {"stable", "recovered"})
        self.assertTrue(all(result.predictions[0].trusted_sides))

    def test_oversized_factory_does_not_compete_with_its_own_fragments(self):
        box = (450, 300, 1100, 700)
        detector = GridDetector([box])
        result = infer_grid(detector, replace(StrategyConfig(), max_refines_per_target=5))
        self.assert_boxes(result, [box])
        self.assertTrue(all(result.predictions[0].trusted_sides))
        self.assertEqual(len(detector.calls), len(set(detector.calls)))

    def test_tiny_corner_fragments_do_not_block_two_axis_recovery(self):
        box = (250, 250, 850, 850)
        result = infer_grid(GridDetector([box]), replace(StrategyConfig(), max_refines_per_target=5))
        self.assert_boxes(result, [box])
        self.assertTrue(all(result.predictions[0].trusted_sides))

    def test_low_score_duplicate_queries_do_not_disable_recovery(self):
        box = (400, 150, 850, 440)
        result = infer_grid(GridDetector([box], duplicate_query=True))
        self.assert_boxes(result, [box])
        self.assertIn(result.predictions[0].status, {"stable", "recovered"})

    def test_nearby_actual_factories_are_not_joined(self):
        boxes = [(450, 300, 1050, 650), (1090, 330, 1310, 610)]
        result = infer_grid(GridDetector(boxes), replace(StrategyConfig(), max_refines_per_target=5))
        self.assert_boxes(result, boxes)

    def test_complete_output_survives_higher_confidence_partial_queries(self):
        box = (400, 150, 850, 440)
        result = infer_grid(GridDetector([box]))
        self.assertEqual(len(result.predictions), 1)
        self.assertAlmostEqual(result.predictions[0].score, 0.78)
        self.assert_boxes(result, [box])

    def test_changing_grid_phase_preserves_same_complete_factory(self):
        box = (400, 300, 900, 650)
        for offset in ((0, 0), (80, 120), (255, 255)):
            with self.subTest(offset=offset):
                config = replace(StrategyConfig(), grid_offset=offset, max_refines_per_target=5)
                result = infer_grid(GridDetector([box]), config)
                self.assert_boxes(result, [box])

    def test_real_raster_edge_is_never_claimed_fully_recovered(self):
        detector = GridDetector([(1700, 500, 1900, 800)])
        result = infer_grid(detector, replace(StrategyConfig(), max_refines_per_target=5))
        self.assertTrue(result.predictions)
        for prediction in result.predictions:
            self.assertLessEqual(prediction.box[2], detector.width)
            if prediction.box[2] == detector.width:
                self.assertFalse(prediction.trusted_sides[2])
                self.assertNotIn(prediction.status, {"stable", "recovered"})

    def test_global_forward_budget_holds_for_full_grid_with_many_fragments(self):
        detector = GridDetector([(450, 300, 1100, 700), (250, 800, 1000, 1200)])
        config = replace(StrategyConfig(), max_refine_windows=2, max_refines_per_target=5)
        result = infer_grid(detector, config)
        self.assertLessEqual(len(detector.calls), 2)
        self.assertEqual(result.stats["refine_windows"], len(detector.calls))
        self.assertEqual(len(detector.calls), len(set(detector.calls)))


if __name__ == "__main__":
    unittest.main()
