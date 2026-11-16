"""Held-out loss of an existing checkpoint: teacher-forced cross-entropy per token on the standard test set of a pool file.
Usage: eval_loss.py <checkpoint> <data.npz>   -> <checkpoint minus .pt>.loss.json"""
import json, sys
import numpy as np, torch
import torch.nn.functional as Fn
import mathgen3 as m3
import model as M
from model import Net, Cfg

path, data = sys.argv[1], sys.argv[2]
M.FUSED = True
dev = "cuda"
ck = torch.load(path, map_location=dev)
Z = np.load(data)
W, PW = [int(w) for w in Z["widths"]], [int(w) for w in Z["pw"]]
cfg = {k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__} | {"max_pos": max(PW) + max(W) + 64}
model = Net(Cfg(**cfg)).to(dev).eval()
model.load_state_dict(ck["state"])
lv = Z["test_level"]
NL = m3.NLEVELS
per, nll, tok, okc, n = np.zeros((NL, 2)), 0.0, 0.0, 0, 0
for b in range(len(W)):
    P, T, I = torch.from_numpy(Z[f"tP{b}"]).to(dev).long(), torch.from_numpy(Z[f"tT{b}"]).to(dev).long(), Z[f"tI{b}"]
    rows = max(16, min(1024, int(3e5 / (PW[b] + W[b]))))
    for i in range(0, len(P), rows):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(P[i : i + rows], T[i : i + rows])
        Tb = T[i : i + rows]
        ce = Fn.cross_entropy(logits.flatten(0, 1).float(), Tb.flatten(), ignore_index=0, reduction="none").view(Tb.shape).sum(1)
        cnt = (Tb != 0).sum(1)
        nll += ce.sum().item(); tok += cnt.sum().item(); n += len(Tb)
        okc += ((logits.argmax(-1) == Tb) | (Tb == 0)).all(1).sum().item()
        lb = lv[I[i : i + rows]]
        for l in range(NL):
            m = torch.from_numpy(lb == l).to(dev)
            per[l] += [ce[m].sum().item(), cnt[m].sum().item()]
res = {"test": nll / tok, "test_tf_exact": okc / n, "test_level": (per[:, 0] / np.maximum(per[:, 1], 1)).tolist()}
print(path, "test loss %.4f  tf-exact %.4f" % (res["test"], res["test_tf_exact"]))
json.dump(res, open(path.replace(".pt", ".loss.json"), "w"))
