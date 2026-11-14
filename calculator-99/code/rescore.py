"""Re-evaluate a saved v2 checkpoint by greedy generation and rewrite its result files.
Usage: rescore.py <checkpoint> <data file>"""
import json, sys
import numpy as np, torch
import mathgen as mg, mathgen2 as m2
from model import Net, Cfg

path, data = sys.argv[1], sys.argv[2]
dev = "cuda"
ck = torch.load(path, map_location=dev)
model = Net(Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})).to(dev).eval()
model.load_state_dict(ck["state"])
Z = np.load(data)
W = [int(w) for w in Z["widths"]]
lv = Z["test_level"]
truth = [mg.decode(r[: list(r).index(mg.EOS)]) for r in Z["test_A"]]
preds = [""] * len(truth)
n_prompt = np.zeros(len(truth))
with torch.no_grad():
    for b in range(len(W)):
        P = torch.from_numpy(Z[f"tP{b}"]).to(dev).long(); I = Z[f"tI{b}"]
        n_prompt[I] = (P != 0).sum(1).cpu().numpy()
        for i in range(0, len(P), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model.generate(P[i : i + 1024], min(2 * W[b] + 96, W[-1] + 24)).cpu().tolist()
            for j, row in zip(I[i : i + 1024].tolist(), toks):
                row = row[: row.index(mg.EOS)] if mg.EOS in row else row
                preds[j] = m2.decode(row)
ans = [t.split("]")[-1] if "]" in t else "" for t in preds]
em = np.array([a == b for a, b in zip(ans, truth)])
close = []
for a, b in zip(ans, truth):
    try:
        close.append(abs(float(a) - float(b)) <= 0.01 * abs(float(b)) + 1e-9)
    except ValueError:
        close.append(False)
close = np.array(close)
lvl = lambda x: [float(x[lv == l].mean()) for l in range(mg.NLEVELS)]
res = json.load(open(path.replace(".pt", ".json")))
old = res["final"]["mean_em"]
tle = float(np.mean((n_prompt + np.array([len(t) for t in preds])) * len(model.sched)))
res["final"] = {"em_gen": lvl(em), "close": lvl(close), "mean_em": float(np.mean(lvl(em))), "tle": tle, "em": lvl(em)}
res["mean_solution_chars"] = float(np.mean([len(t) for t in preds]))
res["preds"] = [preds[i] for i in range(0, len(preds), len(preds) // 280)]
json.dump(res, open(path.replace(".pt", ".json"), "w"))
np.save(path.replace(".pt", ".ok.npy"), em)
print(f"{path}: exact {old:.4f} -> {res['final']['mean_em']:.4f}   " + " ".join(f"{x:.3f}" for x in lvl(em)) + f"   within1% {np.mean(lvl(close)):.4f}  tle {tle:.0f}")
