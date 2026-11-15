"""A simple probe of range generalisation: sin(x) on the 0.01 grid from -40 to 40.
Training only ever asked for sin directly at |x| <= 9.99 (the L6 generator); inside that range the model can use a
2,000-entry table. Outside it, a model that had learned what sine is could reduce the argument; a table cannot.
Usage: sin_probe.py <out.json> [range=lo:hi:step, in hundredths] <name=checkpoint> ..."""
import json, math, sys
import numpy as np, torch
import mathgen as mg, mathgen3 as m3
import model as M
from model import Net, Cfg

out_path = sys.argv[1]
span = [a for a in sys.argv[2:] if a.startswith("range=")]  # range=lo:hi:step in hundredths, default -4000:4000:1
lo, hi, step = [int(v) for v in span[0][6:].split(":")] if span else (-4000, 4000, 1)
specs = [a.split("=") for a in sys.argv[2:] if not a.startswith("range=")]
dev = "cuda"
KS = list(range(lo, hi + 1, step))
texts = ["sin(" + f"{k / 100:.2f}".rstrip("0").rstrip(".") + ")" for k in KS]
truth = []
for k in KS:
    try:
        truth.append(mg.fmt(math.sin(k / 100)))
    except mg.Reject:
        truth.append(None)  # |sin x| below the display range
prompts = [[mg.BOS] + m3.encode(t + "=") for t in texts]
w = max(len(p) for p in prompts)
P = torch.tensor([[mg.PAD] * (w - len(p)) + p for p in prompts], device=dev)
res = {"x": [k / 100 for k in KS], "text": texts, "truth": truth, "held_out": [bool(mg.is_test(t)) for t in texts], "models": {}}
for name, path in specs:
    ck = torch.load(path, map_location=dev)
    net = Net(Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})).to(dev).eval()
    net.load_state_dict(ck["state"])
    outs = []
    for i in range(0, len(P), 2048):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            toks = net.generate_long(P[i : i + 2048], 120).cpu().tolist()
        outs += [m3.decode(r[: r.index(mg.EOS)] if mg.EOS in r else r) for r in toks]
    ans = [o.split("]")[-1] for o in outs]
    ok = [a == t for a, t in zip(ans, truth)]
    res["models"][name] = {"params": ck["cfg"]["params"], "ans": ans, "ok": ok}
    x = np.array(res["x"]); okv = np.array(ok); valid = np.array([t is not None for t in truth])
    inr = np.abs(x) <= 9.99
    print(f"{name}: inside |x|<=9.99: {okv[inr & valid].mean():.4f}   10-20: {okv[(np.abs(x) > 9.99) & (np.abs(x) <= 20) & valid].mean():.4f}   20-40: {okv[(np.abs(x) > 20) & valid].mean():.4f}", flush=True)
json.dump(res, open(out_path, "w"))
