"""Zero-shot: a checkpoint trained on the seven old levels, on the 14-level test set. It can only read the tiers written
in its own alphabet (L1-L9: the old grammar; the equation tiers use characters it has never seen).
Usage: eval_old.py <checkpoint> <data file> [out.json]"""
import json, sys
import numpy as np, torch
import mathgen as mg, mathgen2 as m2, mathgen3 as m3
import model as M
from model import Net, Cfg

path, data = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else path.replace(".pt", ".hard.json")
dev = "cuda"
ck = torch.load(path, map_location=dev)
Z = np.load(data)
W, PW = [int(w) for w in Z["widths"]], [int(w) for w in Z["pw"]]
cfg = {k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__} | {"max_pos": max(PW) + max(W) + 64}
model = Net(Cfg(**cfg)).to(dev).eval()
model.load_state_dict(ck["state"])
lv = Z["test_level"]
truth = [m3.decode(r[: list(r).index(mg.EOS)]) for r in Z["test_A"]]
ok = np.zeros(len(truth), bool)
seen = np.zeros(len(truth), bool)
ngen = np.zeros(len(truth))
vocab = model.cfg.vocab
for b in range(len(W)):
    P, I = torch.from_numpy(Z[f"tP{b}"]).to(dev).long(), Z[f"tI{b}"]
    keep = (P < vocab).all(1).cpu().numpy() & (lv[I] < 9)  # readable by this model
    P, I = P[torch.from_numpy(keep).to(dev)], I[keep]
    n = max(16, min(1024, int(1.2e6 / (PW[b] + W[b]))))
    for i in range(0, len(P), n):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            toks = model.generate_long(P[i : i + n], min(2 * W[b] + 96, W[-1] + 24)).cpu().tolist()
        for j, row in zip(I[i : i + n].tolist(), toks):
            text = m3.decode(row[: row.index(mg.EOS)] if mg.EOS in row else row)
            ok[j], seen[j], ngen[j] = text.split("]")[-1] == truth[j], True, len(text) + 1
res = {"em": [float(ok[(lv == l) & seen].mean()) if (seen & (lv == l)).any() else None for l in range(m3.NLEVELS)],
       "n": [int((seen & (lv == l)).sum()) for l in range(m3.NLEVELS)], "params": ck["cfg"]["params"]}
print(" ".join(f"{m3.LEVELS[l].split()[0]}={x:.3f}" for l, x in enumerate(res["em"]) if x is not None))
json.dump(res, open(out, "w"))
np.save(out.replace(".json", ".ok.npy"), ok)
