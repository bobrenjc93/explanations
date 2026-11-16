"""Test sets outside the original training pool, from the same generators (mathgen3.REGIME):
  stretch   every new tier's size knob one step past its training range; Newton runs longer than any training solution
  stretch2  two steps past (no Newton tier)
  gap       the sizes the wide pool leaves out in the middle of its range (L9 operations, L10 terms, L13 terms)
Usage: build_stretch.py [out.json] [problems per tier] [regime]"""
import json, random, sys
import mathgen as mg, mathgen2 as m2, mathgen3 as m3

OUT = sys.argv[1] if len(sys.argv) > 1 else "hard/stretch.json"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
REGIME = sys.argv[3] if len(sys.argv) > 3 else "stretch"
m2.configure("v2.3")
m3.REGIME = REGIME
LEVELS = {"stretch": range(m3.OLD, m3.NLEVELS), "stretch2": range(m3.OLD, m3.NLEVELS - 1), "gap": sorted(m3.GAPS)}[REGIME]
SEED, CAP = {"stretch": (7000, m3.S_T_MAX), "stretch2": (7100, 9000), "gap": (7200, m3.S_T_MAX)}[REGIME]
out = {"text": [], "truth": [], "kind": [], "level": [], "full_len": [], "ref": [], "size": []}
for lv in LEVELS:
    rng, seen, n, agree = random.Random(SEED + lv), set(), 0, 0
    while n < N:
        p = m3.sample(lv, rng)
        if p["text"] in seen:
            continue
        seen.add(p["text"])
        try:
            sc, ans = m3.solve(p)
        except mg.Reject:
            continue
        size = len(sc) + 2 + len(ans) + 1
        if size > CAP or (REGIME == "stretch" and lv == m3.NLEVELS - 1 and size <= m3.T_MAX):
            continue
        if lv == m3.NLEVELS - 1:
            p["kind"] += f", {sc.count(';x=')} iterations"
        for k, v in (("text", p["text"]), ("truth", p["truth"]), ("kind", p["kind"]), ("level", lv), ("full_len", size), ("ref", "[" + sc + "]" + ans),
                     ("size", p.get("size"))):
            out[k].append(v)
        n += 1
        agree += ans == p["truth"]
    L = out["full_len"][-N:]
    print(f"{m3.LEVELS[lv]:22s} ceiling {agree / N:.3f}  solution tokens mean {sum(L) / N:6.0f}  min {min(L):5d}  max {max(L):5d}  prompt max {max(len(t) for t in out['text'][-N:]) + 2}")
    print("     e.g.", out["text"][-1][:150], "=", out["truth"][-1])
json.dump(out, open(OUT, "w"))
