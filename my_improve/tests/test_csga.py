"""Independent coordinate, gradient, and numerical checks for CSGA.

Run: python -m unittest discover -s my_improve/tests -p test_csga.py -v
"""

import unittest

import torch
import torch.nn.functional as F

from my_improve.csga import CrossScaleGuidedAlignment


def features(channels=4, device="cpu", dtype=torch.float32):
    high = torch.randn(2, channels, 3, 5, device=device, dtype=dtype)
    low = torch.randn(2, channels, 6, 10, device=device, dtype=dtype)
    return high, low


class CrossScaleGuidedAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1807)

    def test_constructor_does_not_advance_random_state(self):
        before = torch.random.get_rng_state().clone()
        CrossScaleGuidedAlignment(channels=16, groups=4)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)

    def test_initial_forward_and_high_gradient_are_exact_nearest(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2)
        high, low = features()
        high.requires_grad_()
        low.requires_grad_()
        reference_high = high.detach().clone().requires_grad_()
        result = module(high, low)
        expected = F.interpolate(reference_high, scale_factor=2, mode="nearest")
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        weights = torch.randn_like(result)
        (result * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(high.grad, reference_high.grad, rtol=0, atol=0)
        torch.testing.assert_close(low.grad, torch.zeros_like(low), rtol=0, atol=0)
        self.assertGreater(module.gate.grad.abs().sum().item(), 0)
        for projection in (module.high_offset, module.low_offset):
            torch.testing.assert_close(projection.weight.grad,
                                       torch.zeros_like(projection.weight), rtol=0, atol=0)

    def test_zero_offset_sampling_matches_bilinear_on_rectangular_features(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2)
        high, low = features()
        offset = module.sampling_offsets(high, low)
        torch.testing.assert_close(offset, torch.zeros_like(offset), rtol=0, atol=0)
        actual = module.sample(high, offset)
        expected = F.interpolate(high, scale_factor=2, mode="bilinear", align_corners=False)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)

    def test_group_coordinate_and_guidance_subpixel_order_against_analytic_planes(self):
        """Reference uses analytic planes, not the production shuffle/grid code."""
        module = CrossScaleGuidedAlignment(channels=4, groups=2)
        h, w = 3, 5
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        high = torch.stack((x + 10 * y, 2 * x + 5 * y,
                            3 * x + 7 * y, 4 * x + 2 * y)).float().unsqueeze(0)
        oy, ox = torch.meshgrid(torch.arange(2 * h), torch.arange(2 * w), indexing="ij")
        # Each group and coordinate sees distinct phases and spatial values.
        phase = (2 * (oy % 2) + ox % 2).float()
        low = torch.stack((0.2 + phase * 0.3, -0.1 - phase * 0.2,
                           -0.4 + oy * 0.07, 0.3 - ox * 0.08)).unsqueeze(0)
        with torch.no_grad():
            for channel in range(4):
                module.low_offset.weight[channel, channel, 0, 0] = 1
        actual = module.sample(high, module.sampling_offsets(high, low))
        x0, y0 = ox.float() / 2 - 0.25, oy.float() / 2 - 0.25
        expected_channels = []
        for channel, (sx, sy) in enumerate(((1, 10), (2, 5), (3, 7), (4, 2))):
            group = channel // 2
            sample_x = (x0 + 0.25 * low[0, group].tanh()).clamp(0, w - 1)
            sample_y = (y0 + 0.25 * low[0, 2 + group].tanh()).clamp(0, h - 1)
            expected_channels.append(sx * sample_x + sy * sample_y)
        expected = torch.stack(expected_channels).unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=5e-6)

    def test_nonzero_gate_propagates_to_both_features_and_offset_predictors(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2)
        with torch.no_grad():
            module.gate.fill_(0.4)
            module.high_offset.weight.normal_(0, 0.03)
            module.low_offset.weight.normal_(0, 0.03)
        high, low = features()
        high.requires_grad_()
        low.requires_grad_()
        output = module(high, low)
        (output * torch.randn_like(output)).sum().backward()
        for tensor in (high, low, module.gate, module.high_offset.weight,
                       module.high_offset.bias, module.low_offset.weight, module.low_offset.bias):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_two_training_steps_activate_initially_zero_offset_predictors(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.5)
        high, low = features()
        target = F.interpolate(high, scale_factor=2, mode="bilinear", align_corners=False)
        F.mse_loss(module(high, low), target).backward()
        self.assertGreater(module.gate.grad.abs().sum().item(), 0)
        self.assertEqual(module.high_offset.weight.grad.abs().sum().item(), 0)
        self.assertEqual(module.low_offset.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(module(high, low), target).backward()
        self.assertGreater(module.high_offset.weight.grad.abs().sum().item(), 1e-10)
        self.assertGreater(module.low_offset.weight.grad.abs().sum().item(), 1e-10)
        optimizer.step()
        self.assertGreater(module.high_offset.weight.abs().sum().item(), 0)
        self.assertGreater(module.low_offset.weight.abs().sum().item(), 0)

    def test_offsets_and_signed_residual_are_bounded(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2)
        high, low = features()
        with torch.no_grad():
            module.high_offset.bias.fill_(1000)
            module.low_offset.bias.fill_(-3000)
            module.gate.copy_(torch.tensor([-1000.0, 1000.0]))
        offsets = module.sampling_offsets(high, low)
        self.assertTrue(torch.isfinite(offsets).all().item())
        self.assertLessEqual(offsets.abs().max().item(), module.max_offset)
        nearest = F.interpolate(high, scale_factor=2, mode="nearest")
        aligned = module.sample(high, offsets)
        output = module(high, low)
        expected = nearest.clone()
        expected[:, :2] -= 0.5 * (aligned[:, :2] - nearest[:, :2])
        expected[:, 2:] += 0.5 * (aligned[:, 2:] - nearest[:, 2:])
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_constant_features_remain_constant_at_borders(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2)
        high = torch.arange(1, 5).float().reshape(1, 4, 1, 1).expand(1, 4, 3, 5)
        low = torch.randn(1, 4, 6, 10)
        with torch.no_grad():
            module.gate.fill_(0.9)
            module.high_offset.weight.normal_(0, 1)
            module.low_offset.weight.normal_(0, 1)
        torch.testing.assert_close(module(high, low),
                                   F.interpolate(high, scale_factor=2, mode="nearest"),
                                   rtol=0, atol=5e-7)

    def _amp_check(self, device, dtype):
        module = CrossScaleGuidedAlignment(channels=4, groups=2).to(device)
        with torch.no_grad():
            module.gate.fill_(0.3)
            module.low_offset.weight.normal_(0, 0.02)
        high, low = features(device=device, dtype=dtype)
        high.requires_grad_()
        low.requires_grad_()
        with torch.autocast(device_type=device, dtype=dtype):
            output = module(high, low)
            loss = output.float().square().mean()
        self.assertEqual(output.dtype, dtype)
        self.assertTrue(torch.isfinite(output).all().item())
        loss.backward()
        for tensor in (high, low, module.high_offset.weight, module.low_offset.weight, module.gate):
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_cpu_bfloat16_autocast(self):
        self._amp_check("cpu", torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_float16_autocast(self):
        self._amp_check("cuda", torch.float16)

    def test_explicit_half_cpu_sampling(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2).half()
        high, _ = features(dtype=torch.float16)
        offsets = torch.zeros(2, 16, 3, 5, dtype=torch.float16)
        result = module.sample(high, offsets)
        expected = F.interpolate(high.float(), scale_factor=2,
                                 mode="bilinear", align_corners=False).half()
        self.assertEqual(result.dtype, torch.float16)
        torch.testing.assert_close(result, expected, rtol=0.001, atol=0.001)

    def test_invalid_constructor_settings(self):
        for kwargs in ({"channels": 0}, {"channels": 4.5}, {"channels": 3, "groups": 2},
                       {"groups": 0}, {"groups": 1.5}, {"max_offset": 0},
                       {"max_offset": float("inf")}, {"max_offset": float("nan")},
                       {"max_residual": 0}, {"max_residual": 1.1},
                       {"max_residual": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CrossScaleGuidedAlignment(**kwargs)

    def test_invalid_input_shapes_and_dtypes(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2)
        high, low = features()
        for h, l in ((high[0], low), (high, low[0]), (high[:, :2], low[:, :2]),
                     (high, low[:1]), (high, low[..., :-1]), (high, low[:, :, :-1]),
                     (high, low.double())):
            with self.subTest(high=tuple(h.shape), low=tuple(l.shape), dtype=l.dtype):
                with self.assertRaises(ValueError):
                    module(h, l)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_mixed_devices_rejected(self):
        module = CrossScaleGuidedAlignment(channels=4, groups=2).cuda()
        high, low = features()
        with self.assertRaises(ValueError):
            module(high.cuda(), low)


if __name__ == "__main__":
    unittest.main()
