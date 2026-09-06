"""512px CSGA integration check; synthetic data, no checkpoint/dataset writes.

python -m my_improve.tests.smoke_csga
"""

import copy
import argparse

import torch

from src.core import YAMLConfig
from src.solver.det_engine import train_one_epoch


def compare_outputs(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_outputs(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            compare_outputs(x, y)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", choices=("dsqc", "qlcs_dsqc"), default="dsqc")
    args = parser.parse_args()
    torch.set_num_threads(2)
    # Tight parity tests use FP32 convolution/matmul, not TF32. Production
    # training precision settings are deliberately not modified by this script.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    options = dict(num_classes=1, eval_spatial_size=[512, 512], HGNetv2={"pretrained": False})
    reference = YAMLConfig(f"my_improve/dfine_hgnetv2_m_{args.reference}.yml", **options)
    candidate = YAMLConfig(f"my_improve/dfine_hgnetv2_m_{args.reference}_csga.yml", **options)
    print(f"Comparison: {args.reference} -> {args.reference}_csga")
    torch.manual_seed(3407)
    original = reference.model.to(device)
    torch.manual_seed(3407)
    model = candidate.model.to(device)
    original_state, state = original.state_dict(), model.state_dict()
    assert original_state.keys() <= state.keys()
    assert all(torch.equal(original_state[key], state[key]) for key in original_state)
    assert all(key.startswith("encoder.csga_upsamplers.") for key in state.keys() - original_state.keys())
    print(f"Shared state tensors identical: {len(original_state)}")
    decoder = model.decoder.decoder
    assert (decoder.qlcs is not None) == (args.reference == "qlcs_dsqc")
    assert decoder.dsqc is not None
    assert model.decoder.mgca is None
    assert decoder.qacg is None and decoder.qfbcg is None and decoder.shea is None
    assert candidate.criterion.qcr is None
    assert reference.yaml_cfg["DFINECriterion"] == {
        k: v for k, v in candidate.yaml_cfg["DFINECriterion"].items() if k != "use_qcr"
    }
    new_params = sum(p.numel() for p in model.parameters())
    old_params = sum(p.numel() for p in original.parameters())
    assert new_params - old_params == 20568
    print(f"Parameters: {old_params} -> {new_params}, +{new_params-old_params}")
    images = torch.rand(1, 3, 512, 512, device=device)
    targets = [{"labels": torch.tensor([0], device=device),
                "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]], device=device)}]
    with torch.no_grad():
        compare_outputs(original.eval()(images), model.eval()(images))
        torch.manual_seed(123)
        ref_train = original.train()(images, targets=targets)
        torch.manual_seed(123)
        new_train = model.train()(images, targets=targets)
        compare_outputs(ref_train, new_train)
        ref_loss = reference.criterion.to(device)(ref_train, targets, epoch=0)
        new_loss = candidate.criterion.to(device)(new_train, targets, epoch=0)
        compare_outputs(ref_loss, new_loss)
    print("Zero-gate 512px eval/train (incl. DN/aux) outputs and original losses: EXACT")
    del original, reference, original_state, state, ref_train, new_train, ref_loss, new_loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    criterion = candidate.criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    metrics = train_one_epoch(
        model, criterion, [(images, targets)] * 2, optimizer, device, epoch=0,
        use_wandb=False, max_norm=0.1,
        num_visualization_sample_batch=0, print_freq=1,
    )
    assert metrics["loss"] > 0 and "qcr_weighted" not in metrics
    assert all(torch.isfinite(p).all() for p in model.parameters())
    for upsampler in model.encoder.csga_upsamplers:
        assert upsampler.gate.detach().abs().sum() > 0
        assert upsampler.high_offset.weight.detach().abs().sum() > 0
        assert upsampler.low_offset.weight.detach().abs().sum() > 0
    print("Two full optimizer steps: gates and BOTH offset branches learned; losses finite")

    if device.type == "cuda":
        # A random untrained detector can overflow the default initial AMP scale.
        # Use a conservative scale ONLY for this synthetic smoke check; this does
        # not change train.py's actual GradScaler settings or training recipe.
        scaler = torch.cuda.amp.GradScaler(init_scale=128.0)
        scale_before = scaler.get_scale()
        amp_metrics = train_one_epoch(
            model, criterion, [(images, targets)], optimizer, device, epoch=0,
            use_wandb=False, max_norm=0.1, scaler=scaler,
            num_visualization_sample_batch=0, print_freq=1,
        )
        assert scaler.get_scale() >= scale_before, "AMP step skipped due to overflow"
        assert amp_metrics["loss"] > 0
        assert all(torch.isfinite(p).all() for p in model.parameters())
        print("Full CUDA AMP optimizer step (synthetic init_scale=128): PASS")

    model.eval()
    with torch.no_grad():
        normal_indices = []
        hook = model.decoder.enc_score_head.register_forward_hook(
            lambda module, args, output: normal_indices.append(output.max(-1).values.topk(300).indices)
        )
        normal_output = model(images)
        hook.remove()
        deployed = copy.deepcopy(model).deploy()
        deploy_indices = []
        hook = deployed.decoder.enc_score_head.register_forward_hook(
            lambda module, args, output: deploy_indices.append(output.max(-1).values.topk(300).indices)
        )
        deploy_output = deployed(images)
        hook.remove()
        same_set = torch.equal(normal_indices[0].sort().values, deploy_indices[0].sort().values)
        print(f"Deploy top-k same set: {same_set}; same order: {torch.equal(normal_indices[0], deploy_indices[0])}")
        if same_set:
            # Preserve the box/score pairing: align by encoder source index,
            # never sort confidence scores independently from their boxes.
            normal_order = normal_indices[0].argsort(dim=1)
            deploy_order = deploy_indices[0].argsort(dim=1)
            for key in ("pred_logits", "pred_boxes"):
                channels = normal_output[key].shape[-1]
                left = normal_output[key].gather(1, normal_order[..., None].expand(-1, -1, channels))
                right = deploy_output[key].gather(1, deploy_order[..., None].expand(-1, -1, channels))
                torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)
                print(f"Source-aligned deploy {key} max difference: {(left-right).abs().max().item():.8f}")
        else:
            print("Top-k cutoff changed: end-to-end parity not established for this random fixture")
        # Fused convolutions can reorder nearly tied encoder top-k candidates
        # in a randomly initialized detector. Test the new encoder path on the
        # SAME backbone features, then the decoder on IDENTICAL memory; do not
        # hide an end-to-end discrepancy behind a loose tolerance.
        features = model.backbone(images)
        memory = model.encoder(features)
        deployed_memory = deployed.encoder(features)
        for left, right in zip(memory, deployed_memory):
            torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)
        normal_decoder = model.decoder(memory)
        deployed_decoder = deployed.decoder(memory)
        for key in ("pred_logits", "pred_boxes"):
            torch.testing.assert_close(normal_decoder[key], deployed_decoder[key], atol=1e-4, rtol=1e-4)
            difference = (normal_output[key] - deploy_output[key]).abs().max().item()
            print(f"End-to-end random-model deploy {key} max difference: {difference:.8f}")
            assert torch.isfinite(deploy_output[key]).all()
    print("Encoder same-input / decoder same-memory deploy parity: PASS")
    try:
        from calflops import calculate_flops
    except ImportError:
        print("calflops unavailable: profiling skipped")
    else:
        flops, _, _ = calculate_flops(deployed.cpu(), input_shape=(1, 3, 512, 512),
                                      print_results=False, print_detailed=False, output_as_string=False)
        print(f"Existing calflops deployment convention: {flops / 1e9:.6f} G (not measured latency)")


if __name__ == "__main__":
    main()
