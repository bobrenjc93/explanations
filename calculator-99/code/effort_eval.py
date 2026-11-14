"""Thinking effort: evaluate one model with its thinking block forced to each effort, and a confidence cascade
(answer at low effort; think harder only when the model is unsure of what it wrote), with and without the brief stage.
Usage: effort_eval.py <checkpoint> <data file>"""
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
N = len(truth)
apps = len(model.sched)
d = model.cfg.d
n_prompt = np.zeros(N)
EFFORTS = [("{", "none"), ("<", "brief"), ("[", "full")]


@torch.no_grad()
def run(opener):
    ok, conf, ntok, first = np.zeros(N, bool), np.zeros(N), np.zeros(N), [""] * N
    for b in range(len(W)):
        P = torch.from_numpy(Z[f"tP{b}"]).to(dev).long(); I = Z[f"tI{b}"]
        n_prompt[I] = (P != 0).sum(1).cpu().numpy()
        for i in range(0, len(P), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks, logp = model.generate(P[i : i + 1024], min(2 * W[b] + 96, W[-1] + 24), conf=True, prefix=[m2.STOI[opener]] if opener else None)
            for j, row, lp in zip(I[i : i + 1024].tolist(), toks.cpu().tolist(), logp.cpu().tolist()):
                row = row[: row.index(mg.EOS)] if mg.EOS in row else row
                text = m2.decode(row)
                ok[j] = text.split("]")[-1] == truth[j]
                conf[j] = float(np.exp(lp)); ntok[j] = len(row) + 1; first[j] = text[:1]
    return ok, conf, ntok, first


lvl = lambda x: [float(x[lv == l].mean()) for l in range(mg.NLEVELS)]
gflop = lambda tokens: float(np.mean(tokens)) * apps * 24 * d * d / 1e9
res = {"efforts": {}}
runs = {}
for opener, name in EFFORTS:
    ok, conf, ntok, _ = run(opener)
    runs[name] = (ok, conf, ntok)
    res["efforts"][name] = {"exact": lvl(ok), "mean": float(np.mean(lvl(ok))), "tokens": float(ntok.mean()), "gflop": gflop(n_prompt + ntok),
                            "tokens_by_level": lvl(ntok)}
    print(f"effort {name:5s}: exact {np.mean(lvl(ok)):.4f}  " + " ".join(f"{x:.3f}" for x in lvl(ok)) + f"   tokens {ntok.mean():.1f}  GFLOP {res['efforts'][name]['gflop']:.3f}", flush=True)
# unforced: which effort does the model pick on its own?
ok, conf, ntok, first = run(None)
pick = {name: float(np.mean([f == o for f in first])) for o, name in EFFORTS}
res["unforced"] = {"mean": float(np.mean(lvl(ok))), "pick": pick, "gflop": gflop(n_prompt + ntok)}
print("unforced:", res["unforced"], flush=True)
# cascade: accept an attempt when the probability of everything the model wrote is at least tau, else go one effort up
res["cascade"] = []
(o0, c0, t0), (o1, c1, t1), (o2, c2, t2) = runs["none"], runs["brief"], runs["full"]
for tau in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]:
    use0 = c0 >= tau
    use1 = ~use0 & (c1 >= tau)
    use2 = ~use0 & ~use1
    ok = np.where(use0, o0, np.where(use1, o1, o2))
    tokens = t0 + np.where(~use0, t1, 0) + np.where(use2, t2, 0)  # every attempt is paid for; the prompt is cached once
    res["cascade"].append({"tau": tau, "mean": float(np.mean(lvl(ok))), "exact": lvl(ok), "tokens": float(tokens.mean()), "gflop": gflop(n_prompt + tokens),
                           "share": [float(use0.mean()), float(use1.mean()), float(use2.mean())],
                           "share_full_by_level": lvl(use2.astype(float))})
    print(f"cascade tau {tau:5.3f}: exact {res['cascade'][-1]['mean']:.4f}  GFLOP {res['cascade'][-1]['gflop']:.3f}  answered at none/brief/full: "
          + "/".join(f"{x:.2f}" for x in res["cascade"][-1]["share"]), flush=True)
# two-stage cascade: no thinking, then straight to a full block (skips the brief attempt)
res["cascade2"] = []
for tau in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.01]:
    use0 = c0 >= tau
    ok = np.where(use0, o0, o2)
    tokens = t0 + np.where(~use0, t2, 0)
    res["cascade2"].append({"tau": tau, "mean": float(np.mean(lvl(ok))), "exact": lvl(ok), "tokens": float(tokens.mean()), "gflop": gflop(n_prompt + tokens),
                            "share": [float(use0.mean()), 0.0, float((~use0).mean())], "share_full_by_level": lvl((~use0).astype(float))})
    print(f"none->full tau {tau:5.3f}: exact {res['cascade2'][-1]['mean']:.4f}  GFLOP {res['cascade2'][-1]['gflop']:.3f}  answered without thinking: "
          f"{use0.mean():.2f}", flush=True)
np.savez(path.replace(".pt", ".effort.npz"), level=lv, n_prompt=n_prompt, **{f"{k}_{n}": v for n, r in runs.items() for k, v in zip(("ok", "conf", "tok"), r)})
json.dump(res, open(path.replace(".pt", ".effort.json"), "w"))
