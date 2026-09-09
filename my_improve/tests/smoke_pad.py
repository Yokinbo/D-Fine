"""Full 512px PAD integration smoke test, CPU by default.

Run: python -m my_improve.tests.smoke_pad
GPU execution requires the explicit --device cuda argument. No data/checkpoint
files are read or written, and no pretrained weights are downloaded.

The synthetic target deliberately surrounds an existing encoder proposal, so
active replacement is an integration test, NOT evidence of real-data coverage
or improved accuracy.
"""
import argparse

import torch

from src.core import YAMLConfig
from src.solver.det_engine import train_one_epoch


@torch.no_grad()
def engineered_target(model, images):
    """Make one valid synthetic GT with proposal IoU approximately 1/1.3**2."""
    features = model.encoder(model.backbone(images))
    memory, shapes = model.decoder._get_encoder_input(features)
    _, _, proposal_boxes, _ = model.decoder._get_decoder_input(memory, shapes)
    proposals = proposal_boxes[0][0].detach().float()
    scaled_sizes = proposals[:, 2:] * 1.3
    # Keep the same center. Only use proposals whose expanded box stays inside
    # the image, avoiding boundary clipping changing the engineered IoU.
    maximum_sizes = 2 * torch.minimum(proposals[:, :2], 1 - proposals[:, :2])
    valid = (torch.isfinite(proposals).all(-1)
             & (proposals[:, 2:] > 0).all(-1)
             & (scaled_sizes < maximum_sizes - 1e-5).all(-1))
    eligible = valid.nonzero(as_tuple=True)[0]
    if not eligible.numel():
        raise AssertionError("Synthetic fixture has no valid expandable encoder proposal")
    chosen = int(eligible[0])
    truth = proposals[chosen:chosen + 1].clone()
    truth[:, 2:] = scaled_sizes[chosen:chosen + 1].clamp(max=1)
    return [{"labels": torch.zeros(1, dtype=torch.long, device=images.device),
             "boxes": truth}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested explicitly but unavailable")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    opts = dict(num_classes=1, eval_spatial_size=[512, 512],
                HGNetv2={"pretrained": False})
    base = YAMLConfig("my_improve/dfine_hgnetv2_m_dsqc.yml", **opts)
    candidate = YAMLConfig("my_improve/dfine_hgnetv2_m_dsqc_pad.yml", **opts)
    torch.manual_seed(3407)
    reference = base.model.to(device).eval()
    torch.manual_seed(3407)
    model = candidate.model.to(device).eval()
    assert reference.decoder.pad is None
    assert model.decoder.pad is not None
    assert model.decoder.decoder.dsqc is not None
    state_a, state_b = reference.state_dict(), model.state_dict()
    assert state_a.keys() == state_b.keys()
    assert all(torch.equal(state_a[key], state_b[key]) for key in state_a)
    print("Identical state tensors:", len(state_a),
          "Parameters:", sum(parameter.numel() for parameter in model.parameters()), flush=True)
    images = torch.rand(1, 3, 512, 512, device=device)
    with torch.no_grad():
        expected, actual = reference(images), model(images)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert model.decoder.pad_stats == {}
    print("512px DSQC / DSQC+PAD inference parity: EXACT", flush=True)
    # The original criterion configuration must be unchanged, including FDR/DDF.
    for key in ("weight_dict", "losses", "matcher"):
        assert base.yaml_cfg["DFINECriterion"][key] == candidate.yaml_cfg["DFINECriterion"][key]
    del reference, expected, actual, state_a, state_b, base

    model.train()
    targets = engineered_target(model, images)
    criterion = candidate.criterion.to(device)
    assert criterion.qcr is None and criterion.rba is None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    final_weight = model.decoder.dec_bbox_head[-1].layers[-1].weight
    before = final_weight.detach().clone()
    captured_losses = {}

    def capture_loss(module, inputs, losses):
        captured_losses.update({key: value.detach() for key, value in losses.items()})

    hook = criterion.register_forward_hook(capture_loss)
    # Engine must reconstruct the schedule from the resumed epoch, not retain
    # this deliberately stale zero-progress value.
    model.decoder.pad.set_progress(0)
    try:
        metrics = train_one_epoch(
            model, criterion, [(images, targets)], optimizer, device,
            epoch=5, use_wandb=False, max_norm=0.1,
            num_visualization_sample_batch=0, print_freq=1)
    finally:
        hook.remove()
    assert model.decoder.pad._ratio == model.decoder.pad.max_ratio
    assert metrics["pad_ratio"] == model.decoder.pad.max_ratio
    assert metrics["pad_replaced"] > 0, "Engine step failed to exercise actual PAD replacement"
    assert 0 < metrics["pad_actual_ratio"] <= model.decoder.pad.max_ratio
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert captured_losses and all(torch.isfinite(value).all() for value in captured_losses.values())
    original_prefixes = ("loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")
    assert all(key.startswith(original_prefixes) for key in captured_losses)
    assert any("loss_fgl_dn" in key for key in captured_losses)
    assert any("loss_ddf_dn" in key for key in captured_losses)
    assert final_weight.grad is not None and torch.isfinite(final_weight.grad).all()
    assert final_weight.grad.abs().sum() > 0
    assert not torch.equal(before, final_weight)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    print("Full 512px FP32 original DN/FGL/DDF loss + backward + optimizer: PASS", flush=True)
    print("PAD schedule restored from engine epoch 5:", model.decoder.pad._ratio,
          "Replaced positive slots:", metrics["pad_replaced"], flush=True)
    print("Synthetic engineered-geometry coverage only; no accuracy claim or training-data coverage claim.", flush=True)


if __name__ == "__main__":
    main()
