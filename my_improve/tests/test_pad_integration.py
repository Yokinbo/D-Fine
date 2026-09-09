"""CPU-only decoder integration, original DN invariants, inference and RNG parity."""
import copy
import unittest
from unittest.mock import patch

import torch

from src.core import YAMLConfig
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.denoising import get_contrastive_denoising_training_group
from my_improve.tests.test_qcr_integration import make_criterion


def tiny_decoder(use_pad=False, **kwargs):
    return DFINETransformer(num_classes=1, hidden_dim=32, feat_channels=[32, 32, 32],
        feat_strides=[8, 16, 32], num_levels=3, nhead=4, num_points=2,
        num_layers=2, dim_feedforward=64, num_queries=6, num_denoising=8,
        reg_max=8, layer_scale=1, use_dsqc=True, dsqc_bottleneck_dim=8,
        dsqc_layers=[1], use_pad=use_pad, **kwargs)


def assert_tree_equal(test, a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        test.assertEqual(set(a), set(b))
        for k in a:
            assert_tree_equal(test, a[k], b[k])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree_equal(test, x, y)
    else:
        test.assertEqual(a, b)


class PADIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(3407)
        self.base = tiny_decoder()
        torch.manual_seed(3407)
        self.candidate = tiny_decoder(True)
        self.features = [torch.rand(1, 32, size, size) for size in (8, 4, 2)]
        with torch.no_grad():
            mem, shapes = self.base._get_encoder_input(self.features)
            _, _, boxes, _ = self.base._get_decoder_input(mem, shapes)
            truth = boxes[0][0, :1].detach().clone()
            truth[:, 2:] *= 1.3  # synthetic eligible ref IoU ~0.59, no real labels
        self.targets = [{"labels": torch.tensor([0]), "boxes": truth}]

    def test_inference_and_state_parity_without_progress(self):
        assert_tree_equal(self, self.base.state_dict(), self.candidate.state_dict())
        self.assertEqual(sum(p.numel() for p in self.base.parameters()),
                         sum(p.numel() for p in self.candidate.parameters()))
        with torch.no_grad():
            a = self.base.eval()(self.features)
            b = self.candidate.eval()(self.features)
        assert_tree_equal(self, a, b)
        self.assertEqual(self.candidate.pad_stats, {})

    def test_epoch_zero_training_and_rng_exact_parity(self):
        self.candidate.pad.set_progress(0)
        torch.manual_seed(90)
        a = self.base(self.features, self.targets)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(90)
        b = self.candidate(self.features, self.targets)
        assert_tree_equal(self, a, b)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(self.candidate.pad_stats["replaced"].item(), 0)

    def test_zero_max_ratio_bypasses_without_progress(self):
        model = tiny_decoder(True, pad_max_ratio=0)
        model.load_state_dict(self.base.state_dict(), strict=True)
        torch.manual_seed(19)
        a = self.base(self.features, self.targets)
        torch.manual_seed(19)
        b = model(self.features, self.targets)
        assert_tree_equal(self, a, b)

    def test_active_changes_only_positive_refs_and_keeps_original_loss_keys(self):
        captured = {}

        def spy_dn(*args, **kwargs):
            result = get_contrastive_denoising_training_group(*args, **kwargs)
            captured["original_refs"] = result[1].detach().clone()
            captured["original_logits"] = result[0].detach().clone()
            captured["original_mask"] = result[2].detach().clone()
            captured["meta"] = copy.deepcopy(result[3])
            return result

        def capture_inputs(module, args, kwargs):
            captured["input_content"] = args[0].detach().clone()
            captured["input_refs"] = args[1].detach().clone()
            captured["input_mask"] = kwargs["attn_mask"].detach().clone()

        self.candidate.pad.set_progress(5)
        hook = self.candidate.decoder.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        try:
            with patch("src.zoo.dfine.dfine_decoder.get_contrastive_denoising_training_group", spy_dn), \
                 patch.object(self.candidate, "_get_decoder_input", wraps=self.candidate._get_decoder_input) as spy_input:
                outputs = self.candidate(self.features, self.targets)
                self.assertEqual(spy_input.call_count, 1)
        finally:
            hook.remove()
        stats = self.candidate.pad_stats
        self.assertGreater(stats["replaced"].item(), 0)
        self.assertLessEqual(stats["replaced"].item(), stats["positive_slots"].item() * .25)
        dn_count = captured["meta"]["dn_num_split"][0]
        changed = (captured["input_refs"][:, :dn_count] != captured["original_refs"]).any(-1)
        allowed = torch.zeros_like(changed)
        allowed[0, captured["meta"]["dn_positive_idx"][0]] = True
        self.assertFalse((changed & ~allowed).any())
        assert_tree_equal(self, captured["original_logits"], captured["input_content"][:, :dn_count])
        assert_tree_equal(self, captured["original_mask"], captured["input_mask"])
        assert_tree_equal(self, captured["meta"], outputs["dn_meta"])
        self.assertFalse(any(v.requires_grad for v in stats.values()))
        losses = make_criterion()(outputs, self.targets)
        self.assertFalse(any("pad" in k or "rba" in k or "qcr" in k for k in losses))
        sum(losses.values()).backward()
        grad = self.candidate.enc_bbox_head.layers[-1].weight.grad
        self.assertIsNotNone(grad)  # normal encoder supervision was NOT detached
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_empty_targets_and_no_eligible_fallback_exact(self):
        empty = [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)}]
        unreachable = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[.5, .5, .00001, .00001]])}]
        self.candidate.pad.set_progress(5)
        for targets in (empty, unreachable):
            torch.manual_seed(92)
            a = self.base(self.features, targets)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(92)
            b = self.candidate(self.features, targets)
            assert_tree_equal(self, a, b)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(self.candidate.pad_stats["replaced"].item(), 0)

    def test_training_requires_progress(self):
        with self.assertRaises(RuntimeError):
            self.candidate(self.features, self.targets)

    def test_configuration_preserves_dsqc_and_disables_old_candidates(self):
        base = YAMLConfig("my_improve/dfine_hgnetv2_m_dsqc.yml", num_classes=1)
        cfg = YAMLConfig("my_improve/dfine_hgnetv2_m_dsqc_pad.yml")
        for key in ("weight_dict", "losses", "matcher"):
            self.assertEqual(base.yaml_cfg["DFINECriterion"][key], cfg.yaml_cfg["DFINECriterion"][key])
        for key, value in base.yaml_cfg["DFINETransformer"].items():
            self.assertEqual(value, cfg.yaml_cfg["DFINETransformer"][key])
        self.assertIsNone(cfg.criterion.qcr)
        self.assertIsNone(cfg.criterion.rba)
        self.assertTrue(cfg.yaml_cfg["DFINETransformer"]["use_pad"])
        for key in ("use_qlcs", "use_qfbcg", "use_qacg", "use_mgca", "use_shea"):
            self.assertFalse(cfg.yaml_cfg["DFINETransformer"][key])
        only = YAMLConfig("my_improve/dfine_hgnetv2_m_pad.yml")
        self.assertFalse(only.yaml_cfg["DFINETransformer"]["use_dsqc"])
        self.assertTrue(only.yaml_cfg["DFINETransformer"]["use_pad"])

    def test_invalid_denoising_configuration(self):
        with self.assertRaises(ValueError):
            tiny_decoder(True, aux_loss=False)


if __name__ == "__main__":
    unittest.main()
