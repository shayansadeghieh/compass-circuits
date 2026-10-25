"""Inspect residual representations and causally patch attention/MLP outputs.

(1) Ridge linear probes ask if x and y can be recovered from intermediate
    residual vectors; a high score alone does NOT establish causal use.
(2) Activation patches ask if replacing one intermediate component changes
    the result toward a counterfactual with +2 x and unchanged y.
"""

import argparse
import json
from pathlib import Path
import torch
from .data import batch, coordinates, x_only_pairs
from .model import hook_site, load_model
from .train import choose_device


@torch.no_grad()
def collect_residuals(model, tokens):
    """Collect last-token pre/post residual stream at each layer on CPU."""
    snapshots = {}
    handles = []
    for i in range(len(model.blocks)):
        for name in ("resid_pre", "resid_post"):
            key = f"layer_{i}.{name}"
            def capture(_module, _inputs, output, key=key):
                snapshots[key] = output[:, -1, :].detach().float().cpu()
            handles.append(hook_site(model, i, name).register_forward_hook(capture))
    try:
        model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    return snapshots


def linear_probe(train_x, train_xy, test_x, test_xy, ridge=1.0):
    """Closed-form ridge regression, trained on separate samples; report R^2."""
    train_x, test_x = train_x.double(), test_x.double()
    train_xy, test_xy = train_xy.double(), test_xy.double()
    m = train_x.mean(0, keepdim=True)
    mean_xy = train_xy.mean(0, keepdim=True)
    centered = train_x - m
    weights = torch.linalg.solve(centered.T @ centered + ridge * torch.eye(centered.size(1), dtype=torch.double),
                                 centered.T @ (train_xy - mean_xy))
    pred = (test_x - m) @ weights + mean_xy
    ss_err = ((pred - test_xy)**2).sum(dim=0)
    ss_tot = ((test_xy - test_xy.mean(dim=0, keepdim=True))**2).sum(dim=0).clamp_min(1e-9)
    return (1 - ss_err / ss_tot).tolist()


@torch.no_grad()
def probes(model, device, n):
    tr_tokens, tr_xy = batch(n, model.cfg.seq_len, device)
    te_tokens, te_xy = batch(n, model.cfg.seq_len, device)
    tr = collect_residuals(model, tr_tokens)
    te = collect_residuals(model, te_tokens)
    result = {}
    for key in tr:
        r2x, r2y = linear_probe(tr[key], tr_xy.cpu(), te[key], te_xy.cpu())
        result[key] = {"x_r2": round(r2x, 4), "y_r2": round(r2y, 4)}
    return result


@torch.no_grad()
def patch_components(model, device, n):
    base, donor = x_only_pairs(n, model.cfg.seq_len, device)
    base_xy = coordinates(base)[:, -1, :]
    donor_xy = coordinates(donor)[:, -1, :]
    bx, by = model(base)
    dx, dy = model(donor)
    offset = model.cfg.seq_len
    base_pred = torch.stack([bx.argmax(-1), by.argmax(-1)], -1) - offset
    donor_pred = torch.stack([dx.argmax(-1), dy.argmax(-1)], -1) - offset
    valid = (base_pred == base_xy).all(-1) & (donor_pred == donor_xy).all(-1)
    num_valid = int(valid.sum().item())
    if num_valid == 0:
        return {"valid_pairs": 0, "message": "Model failed both baseline conditions; train longer first."}
    result = {"valid_pairs": num_valid, "candidate_pairs": n, "patches": {}}
    for layer in range(len(model.blocks)):
        for site_name in ("attn_out", "mlp_out"):
            site = hook_site(model, layer, site_name)
            stored = []
            def capture(_module, _inputs, output):
                stored.append(output[:, -1:, :].detach().clone())
            h = site.register_forward_hook(capture)
            try:
                model(donor)
            finally:
                h.remove()
            donor_act = stored[0]
            def replace(_module, _inputs, output):
                modified = output.clone()
                modified[:, -1:, :] = donor_act
                return modified
            h = site.register_forward_hook(replace)
            try:
                px, py = model(base)
            finally:
                h.remove()
            patched = torch.stack([px.argmax(-1), py.argmax(-1)], -1) - offset
            key = f"layer_{layer}.{site_name}"
            result["patches"][key] = {
                "target_donor_x_accuracy": round(float((patched[valid, 0] == donor_xy[valid, 0]).float().mean()), 4),
                "base_y_preservation": round(float((patched[valid, 1] == base_xy[valid, 1]).float().mean()), 4),
                "both": round(float(((patched[valid, 0] == donor_xy[valid, 0]) &
                                       (patched[valid, 1] == base_xy[valid, 1])).float().mean()), 4),
                "mean_predicted_x_shift": round(float((patched[valid, 0] - base_pred[valid, 0]).float().mean()), 4),
                "mean_predicted_y_shift": round(float((patched[valid, 1] - base_pred[valid, 1]).float().mean()), 4),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/compass.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--probe-examples", type=int, default=2048)
    parser.add_argument("--patch-pairs", type=int, default=512)
    parser.add_argument("--out", default="runs/analysis.json")
    args = parser.parse_args()
    if args.probe_examples < 2 or args.patch_pairs < 1:
        parser.error("probe-examples must be >= 2 and patch-pairs >= 1")
    device = choose_device(args.device)
    model = load_model(args.model, device)
    print("Probing residual representations...")
    probing = probes(model, device, args.probe_examples)
    for name, report in probing.items():
        print(f"  {name:24} x R²={report['x_r2']:+.3f}   y R²={report['y_r2']:+.3f}")
    print("Performing +2 x counterfactual component patches...")
    patches = patch_components(model, device, args.patch_pairs)
    print(f"  Valid model-correct pairs: {patches['valid_pairs']}/{args.patch_pairs}")
    for name, report in patches.get("patches", {}).items():
        print(f"  {name:24} x donor={report['target_donor_x_accuracy']:.3f}  "
              f"y preserved={report['base_y_preservation']:.3f}  both={report['both']:.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"probes": probing, "counterfactual_patching": patches}, indent=2))
    print(f"Saved analysis to {out}")


if __name__ == "__main__":
    main()
