"""512px DSQC -> DSQC+RBA parity and real-engine synthetic optimizer steps.

python -m my_improve.tests.smoke_rba
No checkpoint/data writes; does NOT establish accuracy improvement.
"""
import torch
from src.core import YAMLConfig
from src.solver.det_engine import train_one_epoch


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = dict(num_classes=1, eval_spatial_size=[512, 512], HGNetv2={"pretrained": False})
    base = YAMLConfig("my_improve/dfine_hgnetv2_m_dsqc.yml", **opts)
    candidate = YAMLConfig("my_improve/dfine_hgnetv2_m_dsqc_rba.yml", **opts)
    torch.manual_seed(3407)
    reference = base.model.to(device).eval()
    torch.manual_seed(3407)
    model = candidate.model.to(device).eval()
    a, b = reference.state_dict(), model.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    print("Identical state tensors:", len(a), "Parameters:", sum(p.numel() for p in model.parameters()))
    assert model.decoder.decoder.dsqc is not None
    images = torch.rand(1, 3, 512, 512, device=device)
    with torch.no_grad():
        expected, actual = reference(images), model(images)
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    print("512px DSQC inference parity: EXACT")
    del reference, expected, actual, a, b, base
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Fixed synthetic target, no validation-derived ground truth or optimization.
    targets = [{"labels": torch.tensor([0], device=device),
                "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.4]], device=device)}]
    criterion = candidate.criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for amp in ((False, True) if device.type == "cuda" else (False,)):
        model.train()
        before = model.decoder.dec_bbox_head[-1].layers[-1].weight.detach().clone()
        scaler = torch.cuda.amp.GradScaler(init_scale=128.0) if amp else None
        metrics = train_one_epoch(model, criterion, [(images, targets)], optimizer, device,
            epoch=5, use_wandb=False, max_norm=0.1, scaler=scaler,
            num_visualization_sample_batch=0, print_freq=1)
        assert metrics["rba_selected"] > 0
        assert metrics["rba_weighted"] > 0
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert not torch.equal(before, model.decoder.dec_bbox_head[-1].layers[-1].weight)
        if amp:
            assert scaler.get_scale() >= 128.0, "AMP overflow: optimizer step may have been skipped"
        print(f"Full {'AMP' if amp else 'FP32'} loss/backward/optimizer + active RBA: PASS")


if __name__ == "__main__":
    main()
