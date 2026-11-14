"""Post-training weight quantisation: how many bits per weight does a trained calculator need?
Usage: quant.py <checkpoint> [v1|v2]"""
import copy, json, sys
import numpy as np, torch
import mathgen as mg, mathgen2 as m2
from model import Net, Cfg, quantize, QLinear

path, fmt = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "v1")
dev = "cuda"
ck = torch.load(path, map_location=dev)
base = Net(Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})).to(dev).eval()
base.load_state_dict(ck["state"])

if fmt == "v1":
    Z = np.load("data.npz")
    buckets = [(torch.from_numpy(Z["test_P"]).to(dev).long(), torch.arange(len(Z["test_P"])), mg.C_MAX)]
    dec = mg.decode
else:
    Z = np.load(sys.argv[3] if len(sys.argv) > 3 else "data2.npz")
    W = [int(w) for w in Z["widths"]]
    buckets = [(torch.from_numpy(Z[f"tP{b}"]).to(dev).long(), torch.from_numpy(Z[f"tI{b}"]).long(), min(2 * W[b] + 96, W[-1] + 24)) for b in range(len(W))]
    dec = m2.decode
lv = Z["test_level"]
truth = [mg.decode(r[: list(r).index(mg.EOS)]) for r in Z["test_A"]]


@torch.no_grad()
def score(model):
    ok = np.zeros(len(truth), bool)
    for P, I, mx in buckets:
        for i in range(0, len(P), 2000):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model.generate(P[i : i + 2000], mx).cpu().tolist()
            for j, row in zip(I[i : i + 2000].tolist(), toks):
                row = row[: row.index(mg.EOS)] if mg.EOS in row else row
                ok[j] = dec(row).split("]")[-1] == truth[j]
    return [float(ok[lv == l].mean()) for l in range(mg.NLEVELS)]


def quantize_best(w, bits):
    """Per-row symmetric uniform quantiser with the clipping range picked by a small search for the least squared error
    (what a careful post-training quantiser does; the fixed-range version in model.py is for training)."""
    q = 2 ** (bits - 1) - 1
    mx = w.abs().amax(-1, keepdim=True).clamp(min=1e-8)
    best, best_err = None, None
    for frac in torch.linspace(0.2, 1.0, 33, device=w.device):
        step = mx * frac / q
        wq = (w / step).round().clamp(-q, q) * step
        err = (wq - w).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = wq, err
        else:
            take = err < best_err
            best = torch.where(take, wq, best)
            best_err = torch.where(take, err, best_err)
    return best


out = {"fp": score(base)}
print("full precision      ", " ".join(f"{x:.3f}" for x in out["fp"]), " mean %.4f" % np.mean(out["fp"]), flush=True)
for bits in (8, 6, 5, 4, 3, 2):
    m = copy.deepcopy(base)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, QLinear):
                mod.weight.copy_(quantize_best(mod.weight.float(), bits).to(mod.weight.dtype))
    out[bits] = score(m)
    print(f"{bits}-bit block weights  ", " ".join(f"{x:.3f}" for x in out[bits]), " mean %.4f" % np.mean(out[bits]), flush=True)
json.dump(out, open(path.replace(".pt", ".quant.json"), "w"))
