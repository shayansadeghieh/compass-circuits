"""Train a small causal transformer to predict displacement from N/E/S/W tokens."""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from .data import batch
from .model import Config, CompassTransformer, loss_and_accuracy, save_model


def choose_device(preference: str) -> torch.device:
    if preference != "auto":
        device = torch.device(preference)
        if preference == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable; try --device cpu")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: CompassTransformer, device: torch.device, batch_size: int = 2048):
    model.eval()
    tokens, xy = batch(batch_size, model.cfg.seq_len, device)
    loss, joint, per_axis = loss_and_accuracy(model(tokens), xy, model.cfg.seq_len)
    return {"loss": float(loss.item()), "joint_accuracy": float(joint.item()),
            "x_accuracy": float(per_axis[0].item()), "y_accuracy": float(per_axis[1].item())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--out", default="runs/compass.pt")
    args = p.parse_args()
    if args.steps < 1 or args.seq_len < 1 or args.batch_size < 1 or args.eval_every < 1:
        p.error("steps, seq-len, batch-size, and eval-every must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    cfg = Config(seq_len=args.seq_len)
    model = CompassTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"device={device} | parameters={total_params:,} | sequence_length={cfg.seq_len}")
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        tokens, xy = batch(args.batch_size, cfg.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = loss_and_accuracy(model(tokens), xy, cfg.seq_len)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            report = evaluate(model, device)
            report["step"] = step
            history.append(report)
            print(f"step {step:5} | val loss {report['loss']:.3f} | x {report['x_accuracy']:.3f} | "
                  f"y {report['y_accuracy']:.3f} | joint {report['joint_accuracy']:.3f}", flush=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(path))
    path.with_suffix(".json").write_text(json.dumps({"config": vars(args), "history": history}, indent=2))
    print(f"Saved model to {path} and metrics to {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
