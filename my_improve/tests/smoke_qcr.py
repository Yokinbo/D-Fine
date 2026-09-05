"""Full-model smoke check, random inputs, no checkpoints/dataset writes.

Run: python -m my_improve.tests.smoke_qcr
"""

import torch

from src.core import YAMLConfig
from src.solver.det_engine import train_one_epoch


def main():
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options = dict(num_classes=1, eval_spatial_size=[512, 512], HGNetv2={"pretrained": False})
    reference = YAMLConfig("my_improve/dfine_hgnetv2_m_qlcs_dsqc.yml", **options)
    candidate = YAMLConfig("my_improve/dfine_hgnetv2_m_qlcs_dsqc_qcr.yml", **options)
    torch.manual_seed(3407)
    original_model = reference.model.eval()
    torch.manual_seed(3407)
    model = candidate.model.eval()
    original_state, state = original_model.state_dict(), model.state_dict()
    assert original_state.keys() == state.keys()
    assert all(torch.equal(original_state[k], state[k]) for k in state)
    params = sum(p.numel() for p in model.parameters())
    print(f"Identical initial state tensors: {len(state)}; parameters: {params}")
    torch.manual_seed(42)
    images = torch.rand(1, 3, 512, 512, device=device)
    original_model.to(device)
    model.to(device)
    with torch.no_grad():
        original_output = original_model(images)
        candidate_output = model(images)
    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(original_output[key], candidate_output[key], atol=0, rtol=0)
    print("512px inference logits/boxes: exactly equal")
    del original_model, original_output, candidate_output, original_state, state, reference
    if device.type == "cuda":
        torch.cuda.empty_cache()

    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]], device=device),
                "labels": torch.tensor([0], device=device)}]
    model.train()
    # Use a real final prediction as a synthetic GT to exercise reliable pairs.
    with torch.no_grad():
        probe = model(images, targets=targets)
        targets[0]["boxes"] = probe["pred_boxes"][0, :1].detach().clone()
    criterion = candidate.criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # One real training-engine step exercises all original losses, DN and logging.
    metrics = train_one_epoch(model, criterion, [(images, targets)], optimizer, device,
                              epoch=5, use_wandb=False, max_norm=0.1,
                              num_visualization_sample_batch=0, print_freq=1)
    assert metrics["loss"] > 0
    assert "qcr_weighted" in metrics
    assert metrics["qcr_pairs"] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print("Full loss + backward + optimizer + QCR diagnostics: PASS")
    if device.type == "cuda":
        amp_metrics = train_one_epoch(
            model, criterion, [(images, targets)], optimizer, device,
            epoch=5, use_wandb=False, max_norm=0.1,
            scaler=torch.cuda.amp.GradScaler(),
            num_visualization_sample_batch=0, print_freq=1,
        )
        assert amp_metrics["loss"] > 0
        assert all(torch.isfinite(p).all() for p in model.parameters())
        print("Full CUDA AMP training-engine step: PASS")


if __name__ == "__main__":
    main()
