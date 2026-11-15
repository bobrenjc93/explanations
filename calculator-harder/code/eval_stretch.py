"""A trained checkpoint on the stretch set (build_stretch.py): problems larger than anything in its training pool.
Usage: eval_stretch.py <checkpoint> [stretch.json]   -> <checkpoint minus .pt>.stretch.json"""
import collections, json, sys
import numpy as np, torch
import mathgen as mg, mathgen3 as m3
import model as M
from model import Net, Cfg
from diagnose import diagnose, fault_offset

path = sys.argv[1]
S = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "hard/stretch.json"))
M.FUSED = True
dev = "cuda"
CAP = 7000
ck = torch.load(path, map_location=dev)
cfg = {k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__} | {"max_pos": m3.S_P_MAX + CAP + 64}
model = Net(Cfg(**cfg)).to(dev).eval()
model.load_state_dict(ck["state"])
N = len(S["text"])
order = np.argsort(S["full_len"])
ok, glen, preds = np.zeros(N, bool), np.zeros(N, int), [""] * N
i = 0
while i < N:
    budget = min(2 * S["full_len"][order[min(i + 255, N - 1)]] + 96, CAP)
    rows = max(8, min(256, int(4e9 / (model.cfg.d * budget * 4 * len(model.sched)))))
    idx = order[i : i + rows]
    budget = min(2 * max(S["full_len"][j] for j in idx) + 96, CAP)
    prompts = [[mg.BOS] + m3.encode(S["text"][j] + "=") for j in idx]
    w = max(len(p) for p in prompts)
    P = torch.tensor([[mg.PAD] * (w - len(p)) + p for p in prompts], device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        toks = model.generate_long(P, budget).cpu().tolist()
    for j, row in zip(idx, toks):
        text = m3.decode(row[: row.index(mg.EOS)] if mg.EOS in row else row)
        ok[j], glen[j], preds[j] = text.split("]")[-1] == S["truth"][j], len(text) + 1, text
    torch.cuda.empty_cache()
    i += rows
    print(f"{i}/{N}", flush=True)
lv = np.array(S["level"])
kinds = {}  # first fault of every wrong answer, per level (only meaningful for models that write full solutions)
for l in range(m3.OLD, m3.NLEVELS):
    c = collections.Counter(diagnose(preds[j], S["ref"][j])[0] for j in np.where((lv == l) & ~ok)[0] if preds[j].startswith("["))
    kinds[m3.LEVELS[l]] = dict(c.most_common())
# where in its own output each wrong solution first goes wrong (token offset); training outputs never exceeded m3.T_MAX tokens
fault = [fault_offset(preds[j], S["ref"][j]) if (not ok[j] and preds[j].startswith("[")) else None for j in range(N)]
res = {"kinds": kinds, "fault_pos": fault, "examples": {m3.LEVELS[l]: [{"text": S["text"][j], "truth": S["truth"][j], "pred": preds[j][-160:], "ref": S["ref"][j][-160:]}
                                                   for j in np.where((lv == l) & ~ok)[0][:3]] for l in range(m3.OLD, m3.NLEVELS)},
       "em": {m3.LEVELS[l]: float(ok[lv == l].mean()) for l in range(m3.OLD, m3.NLEVELS)}, "ok": [int(x) for x in ok], "gen_len": [int(x) for x in glen],
       "params": ck["cfg"]["params"]}
print(" ".join(f"{k.split()[0]}={v:.3f}" for k, v in res["em"].items()))
json.dump(res, open(path.replace(".pt", ".stretch.json"), "w"))
