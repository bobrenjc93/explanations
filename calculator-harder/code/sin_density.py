"""How often the worked solutions of composite (L7) training problems recall sin(x) at each size of x, and which of the
probe's arguments (sin_probe.py) ever occur there. Direct sin problems (L6) only use |x| <= 9.99; larger arguments reach
sin through computed values such as sin(5.5^2).
Usage: sin_density.py <out.json> [problems] [sin_probe.json]"""
import collections, json, os, random, re, sys
import numpy as np
import mathgen as mg, mathgen2 as m2

N = int(sys.argv[2]) if len(sys.argv) > 2 else 600_000
probe = sys.argv[3] if len(sys.argv) > 3 else "runs/hard/sin_probe.json"
m2.configure("v2.3")
EDGES = [10, 20, 40, 70, 100, 150, 200, 300]
rng, c, seen, n = random.Random(0), collections.Counter(), collections.Counter(), 0
while n < N:
    e, _, _ = mg.sample(6, rng)
    try:
        sc, _ = m2.solve(e, rng)
    except mg.Reject:
        continue
    n += 1
    for m in re.finditer(r"sin\((-?[0-9.]+)\)=", sc):
        v = float(m.group(1))
        if abs(v) <= 9.99:
            c["inside"] += 1
            continue
        seen[m.group(1)] += 1
        for a, b in zip(EDGES, EDGES[1:]):
            if a <= abs(v) < b:
                c[("+" if v > 0 else "-") + f"{a}-{b}"] += 1
        if abs(v) >= EDGES[-1]:
            c[("+" if v > 0 else "-") + "beyond"] += 1
out = {"per_million_composite_problems": {k: v * 1e6 / n for k, v in c.items()}, "edges": EDGES, "problems": n, "distinct_outside": len(seen)}
print({k: round(v) for k, v in sorted(out["per_million_composite_problems"].items())}, "distinct arguments outside the grid:", len(seen))
if os.path.exists(probe):  # probe points between 10 and 40: met in these training solutions or not
    R = json.load(open(probe))
    x = np.array(R["x"])
    pos = np.round(x * 100).astype(int) > 999
    met = np.array([t[4:-1] in seen for t in R["text"]])
    out["probe"] = {"n": int(pos.sum()), "met": int((pos & met).sum()), "models": {}}
    for name, M in R["models"].items():
        val = np.array([float(a) if re.fullmatch(r"-?\d+(\.\d+)?", a) else np.nan for a in M["ans"]])
        ok, t = np.array(M["ok"]), np.sin(x)
        st = {}
        for lab, m in (("met", pos & met), ("unmet", pos & ~met)):
            mm = m & ~np.isnan(val)
            st[lab] = {"exact": float(ok[m].mean()), "within": float(np.mean(np.abs(val[mm] - t[mm]) < 0.1)), "sign": float(np.mean(np.sign(val[mm]) == np.sign(t[mm])))}
        out["probe"]["models"][name] = st
        print(name, st)
json.dump(out, open(sys.argv[1], "w"))
