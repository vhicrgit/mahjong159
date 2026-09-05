"""Read-only reproduction of the DAgger target-support masking bug.

Run from repository root with the existing PyTorch Python environment:
  OMP_NUM_THREADS=1 python -m tools.perf.audit_teacher_mask
Does not update checkpoints or train the model. Uses lists at the NumPy/Torch
boundary so this diagnostic also works with the local NumPy ABI mismatch.
"""

import json
from pathlib import Path

import numpy as np
import torch

from backend.rl.model import build_model


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260905)
    q = torch.tensor([[1., 2., 3.]], requires_grad=True)
    p = torch.tensor([[1., 0., 0.]])
    masked = q.masked_fill(p <= 0, -1e9)
    loss = -(p * torch.log_softmax(masked, -1)).sum()
    loss.backward()
    result = {"toy": {"old_loss": loss.item(), "old_grad": q.grad.tolist(),
                      "old_accuracy": float(masked.argmax(-1).item() == 0)}}
    q2 = q.detach().clone().requires_grad_(True)
    loss2 = -(p * torch.log_softmax(q2, -1)).sum()
    loss2.backward()
    result["toy"].update(correct_loss=loss2.item(), correct_grad=q2.grad.tolist(),
                         correct_accuracy=float(q2.argmax(-1).item() == 0))
    assert loss.item() == 0 and q.grad.abs().sum().item() == 0
    assert loss2.item() > 0 and q2.grad.abs().sum().item() > 0

    results = []
    for data, checkpoints in [
        ("teach64_conf.npz", ["bc_r2_s3.pt", "ei64_a.pt"]),
        ("teach64b_conf.npz", ["ei64_a.pt", "ei64_b.pt"]),
    ]:
        path = Path("models") / data
        if not path.exists():
            continue
        with np.load(path) as d:
            n = len(d["feats"])
            # Reproduce the first-dataset validation split in dagger_train.py.
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(1234))
            idx = idx[:max(500, n // 20)].tolist()
            x = torch.tensor(d["feats"][idx].tolist(), dtype=torch.float32)
            target = torch.tensor(d["target"][idx].tolist(), dtype=torch.float32)
            best = torch.tensor(d["bests"][idx].tolist(), dtype=torch.long)
        # features.py encodes count=0 in channel zero of each tile's 4 channels.
        legal = x[:, :112].reshape(-1, 28, 4)[:, :, 0] < 0.5
        assert legal.gather(1, best[:, None]).all()
        assert ((target > 0).sum(1) == 1).all()
        for ckpt in checkpoints:
            ck = torch.load(Path("models") / ckpt, map_location="cpu", weights_only=True)
            model = build_model(ck["size"], feat_dim=ck.get("feat_dim", 628))
            model.load_state_dict(ck["model"])
            model.eval()
            with torch.no_grad():
                logits, _ = model(x)
            old = logits.masked_fill(target <= 0, -1e9)
            real = logits.masked_fill(~legal, -1e9)
            results.append({"data": data, "checkpoint": ckpt, "validation_n": len(idx),
                            "old_accuracy": (old.argmax(1) == best).float().mean().item(),
                            "legal_accuracy": (real.argmax(1) == best).float().mean().item(),
                            "old_policy_loss": -(target * old.log_softmax(-1)).sum(1).mean().item(),
                            "legal_policy_loss": -(target * real.log_softmax(-1)).sum(1).mean().item()})
    result["checkpoints"] = results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
