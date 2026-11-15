"""A test set beyond the training range: for every new tier, problems whose size knob is larger than anything in the
training pool (mathgen3.STRETCH). 300 per tier, reference solutions up to 6,000 tokens.
  L8 15-20 digit sums, 9-11 digit products ...   L9 10-14 operations      L10 9-13 terms
  L11 coefficients up to 400                     L12 3x3, two-digit       L13 13-24 terms
  L14 Newton runs that need more than the 2,304 tokens any training solution had
Usage: build_stretch.py [out.json] [problems per tier]"""
import json, random, sys
import mathgen as mg, mathgen2 as m2, mathgen3 as m3

OUT = sys.argv[1] if len(sys.argv) > 1 else "hard/stretch.json"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
m2.configure("v2.3")
m3.STRETCH = True
out = {"text": [], "truth": [], "kind": [], "level": [], "full_len": [], "ref": []}
for lv in range(m3.OLD, m3.NLEVELS):
    rng, seen, n, agree = random.Random(7000 + lv), set(), 0, 0
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
        if size > m3.S_T_MAX or (lv == m3.NLEVELS - 1 and size <= m3.T_MAX):
            continue
        if lv == m3.NLEVELS - 1:
            p["kind"] += f", {sc.count(';x=')} iterations"
        for k, v in (("text", p["text"]), ("truth", p["truth"]), ("kind", p["kind"]), ("level", lv), ("full_len", size), ("ref", "[" + sc + "]" + ans)):
            out[k].append(v)
        n += 1
        agree += ans == p["truth"]
    L = out["full_len"][-N:]
    print(f"{m3.LEVELS[lv]:22s} ceiling {agree / N:.3f}  solution tokens mean {sum(L) / N:6.0f}  min {min(L):5d}  max {max(L):5d}  prompt max {max(len(t) for t in out['text'][-N:]) + 2}")
    print("     e.g.", out["text"][-1][:150], "=", out["truth"][-1])
json.dump(out, open(OUT, "w"))
