"""覆盖匹配、置信度与 AP 分离、固定格网分层及无真值边界情况。"""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation.py"
SPEC = importlib.util.spec_from_file_location("boundary_response_evaluation_test", MODULE_PATH)
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def prediction(box, score=0.9, class_id=1):
    return SimpleNamespace(box=box, score=score, class_id=class_id)


def truth(box, plant_id="plant"):
    return SimpleNamespace(box=box, plant_id=plant_id)


class EvaluationTests(unittest.TestCase):
    def evaluate(self, predictions, truths, **kwargs):
        return evaluation.evaluate_predictions(predictions, truths, width=2048, height=2048, **kwargs)

    def test_matching_considers_best_unmatched_ground_truth(self):
        # 第二个预测对已匹配 A 的 IoU 更高，但也可正确匹配 B。
        truths = [truth((100, 100, 200, 200), "A"), truth((130, 100, 230, 200), "B")]
        predictions = [prediction((100, 100, 200, 200), 0.95), prediction((105, 100, 205, 200), 0.8)]
        result = self.evaluate(predictions, truths)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 0))
        self.assertEqual(result["scene_single_class_AP50_101pt"], 1.0)
        self.assertEqual(result["duplicate_boxes"], 0)
        self.assertEqual([row["prediction_index"] for row in result["per_ground_truth"]], [0, 1])

    def test_ap_uses_low_confidence_predictions(self):
        result = self.evaluate([prediction((100, 100, 200, 200), 0.2)], [truth((100, 100, 200, 200))])
        self.assertEqual(result["scene_single_class_AP50_101pt"], 1.0)
        self.assertEqual(result["scene_single_class_AP75_101pt"], 1.0)
        self.assertEqual(result["scene_single_class_mAP50_95_101pt"], 1.0)
        self.assertEqual((result["tp"], result["fn"]), (0, 1))
        self.assertEqual(result["recall"], 0.0)

    def test_false_positive_preceding_true_positive_reduces_ap(self):
        result = self.evaluate(
            [prediction((400, 400, 500, 500), 0.99), prediction((100, 100, 200, 200), 0.9)],
            [truth((100, 100, 200, 200))], valid_area_km2=2,
        )
        self.assertEqual(result["scene_single_class_AP50_101pt"], 0.5)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["fp_per_km2"], 0.5)
        self.assertEqual(result["duplicate_boxes"], 0)

    def test_duplicate_not_double_counted_against_neighboring_truths(self):
        box = (100, 100, 200, 200)
        result = self.evaluate([prediction(box, 0.99), prediction(box, 0.9), prediction(box, 0.8)],
                               [truth(box, "A"), truth(box, "B")])
        self.assertEqual((result["tp"], result["fp"], result["duplicate_boxes"]), (2, 1, 1))

    def test_oversize_position_and_outer_edge_are_separate(self):
        truths = [truth((100, 100, 700, 200), "large"), truth((480, 300, 560, 380), "split"),
                  truth((0, 500, 50, 550), "outer"), truth((700, 700, 750, 750), "whole")]
        result = self.evaluate([], truths)
        strata = result["reference_grid_strata"]
        self.assertEqual(strata["oversized"]["gt_count"], 1)
        self.assertEqual(strata["position_only_truncated"]["gt_count"], 2)
        self.assertEqual(strata["contained"]["gt_count"], 1)
        self.assertEqual(strata["internal_visible_lt_0_9"]["gt_count"], 2)
        self.assertEqual(strata["internal_visible_lt_0_7"]["gt_count"], 2)
        self.assertEqual(strata["image_outer_boundary_touch"]["gt_count"], 1)
        self.assertFalse(hasattr(truths[0], "max_visible_ratio"))

    def test_last_reference_cell_does_not_shift_back(self):
        # 宽600的末格应是 [512,600]，不能退至88形成与首格的重叠。
        result = evaluation.evaluate_predictions([], [truth((450, 100, 590, 200))], width=600, height=600)
        self.assertAlmostEqual(result["per_ground_truth"][0]["max_visible_ratio"], 78 / 140)
        self.assertEqual(result["reference_grid_strata"]["internal_visible_lt_0_7"]["gt_count"], 1)

    def test_reference_offset_is_normalized_and_grid_is_fixed(self):
        a = self.evaluate([], [truth((480, 300, 560, 380))], reference_offset=(100, 100))
        b = self.evaluate([], [truth((480, 300, 560, 380))], reference_offset=(612, -412))
        self.assertEqual(a["reference_grid_strata"], b["reference_grid_strata"])
        self.assertEqual(a["per_ground_truth"][0]["max_visible_ratio"], 1.0)

    def test_edge_errors_are_normalized_and_only_use_true_positives(self):
        result = self.evaluate([prediction((105, 96, 195, 308))], [truth((100, 100, 200, 300))])
        errors = result["edge_errors_tp_only"]
        self.assertEqual(errors["matched_gt_count"], 1)
        self.assertEqual(errors["mean_signed"], {"left": 0.05, "top": -0.02, "right": -0.05, "bottom": 0.04})
        self.assertAlmostEqual(errors["mean_absolute_all_edges"], 0.04)
        empty = self.evaluate([], [truth((100, 100, 200, 300))])["edge_errors_tp_only"]
        self.assertIsNone(empty["mean_absolute_all_edges"])

    def test_empty_cases_remain_strict_json(self):
        result = self.evaluate([prediction((100, 100, 200, 200))], [])
        self.assertIsNone(result["scene_single_class_AP50_101pt"])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["fp_per_km2"])
        self.assertEqual(result["fp"], 1)
        json.dumps(result, allow_nan=False)
        missing = self.evaluate([], [truth((100, 100, 200, 200))])
        self.assertEqual(missing["scene_single_class_AP50_101pt"], 0.0)
        json.dumps(missing, allow_nan=False)

    def test_class_mismatch_is_not_a_true_positive(self):
        result = self.evaluate([prediction((100, 100, 200, 200), class_id=2)], [truth((100, 100, 200, 200))])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_invalid_area_or_boxes_raise_actionable_errors(self):
        for area in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.evaluate([], [], valid_area_km2=area)
        with self.assertRaises(ValueError):
            self.evaluate([prediction((100, 100, 100, 200))], [])

    def test_no_coco_max_dets_limit(self):
        truths = [truth((20 + index * 8, 20, 24 + index * 8, 24), str(index)) for index in range(120)]
        result = self.evaluate([prediction(item.box) for item in truths], truths)
        self.assertEqual(result["tp"], 120)
        self.assertEqual(result["scene_single_class_AP50_101pt"], 1.0)


if __name__ == "__main__":
    unittest.main()
