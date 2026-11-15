"""Where a run's wrong answers come from. For every failed test problem the model's own worked solution is re-checked step
by step: the first arithmetic step that is wrong names the failure; if every arithmetic step is right, the first place
the solution departs from the reference does.
Usage: errs3.py <run prefix, e.g. runs/hard/H_full_3m> <data file>   -> <prefix>.errs.json"""
import collections, json, re, sys
import numpy as np
import mathgen as mg, mathgen3 as m3

prefix, data = sys.argv[1], sys.argv[2]
Z = np.load(data)
side = json.load(open(data.replace(".npz", ".test.json")))
preds = json.load(open(prefix + ".preds.json"))
ok = np.load(prefix + ".ok.npy")
ref = [""] * len(preds)
for b in range(len(Z["widths"])):
    for i, row in zip(Z[f"tI{b}"].tolist(), Z[f"tT{b}"].tolist()):
        ref[i] = m3.decode(row[: row.index(mg.EOS)] if mg.EOS in row else row)
lv = np.array(side["level"])
from diagnose import diagnose

out = {"levels": {}, "where": {}}
for l in range(m3.NLEVELS):
    idx = np.where((lv == l) & ~ok)[0]
    c, pos = collections.Counter(), []
    for i in idx:
        kind, where = diagnose(preds[i], ref[i])
        c[kind] += 1
        if where is not None:
            pos.append(where)
    out["levels"][m3.LEVELS[l]] = {"failed": int(len(idx)), "n": int((lv == l).sum()), "kinds": dict(c.most_common())}
    out["where"][m3.LEVELS[l]] = [float(np.mean([a <= p < a + 0.2 or (a == 0.8 and p >= 1) for p in pos])) if pos else 0.0 for a in (0, .2, .4, .6, .8)]
    if len(idx):
        print(f"{m3.LEVELS[l]:22s} failed {len(idx):5d}/{(lv == l).sum()}  " + ", ".join(f"{k} {v}" for k, v in c.most_common(4)))
# accuracy by sub-type for the new tiers
kinds = collections.defaultdict(list)
for i, (l, k) in enumerate(zip(side["level"], side["kind"])):
    if l >= m3.OLD:
        kinds[(l, k)].append(bool(ok[i]))
out["kinds"] = [{"level": m3.LEVELS[l], "kind": k, "n": len(v), "em": float(np.mean(v))} for (l, k), v in sorted(kinds.items(), key=lambda kv: (kv[0][0], len(kv[0][1]), kv[0][1]))]
json.dump(out, open(prefix + ".errs.json", "w"))
