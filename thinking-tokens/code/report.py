"""Text summary of every finished run."""
import glob, json, os, sys
import numpy as np
RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs/main"
rows = []
for f in sorted(glob.glob(f"{RUNS}/*.json")):
    if f.endswith("probes.json"):
        continue
    r = json.load(open(f))
    fin = r["final"]
    rows.append((r["name"], fin["mean_em"], np.mean(fin["em_gen"]), np.mean(fin["close"]), fin["tle"], fin["em_gen"], r["train_time"], r["model"]["params"], r.get("by_K")))
print(f"{'run':18s} {'EM':>6s} {'gen':>6s} {'~1%':>6s} {'TLE':>6s}  " + " ".join(f"L{i+1:<4d}" for i in range(7)) + "  min   params")
for n, em, gen, close, tle, lv, t, p, byk in rows:
    print(f"{n:18s} {em:6.3f} {gen:6.3f} {close:6.3f} {tle:6.1f}  " + " ".join(f"{x:.3f}" for x in lv) + f"  {t/60:4.1f}  {p/1e6:.2f}M")
    if byk and len(byk) > 1:
        print("     by K: " + "  ".join(f"K={k}: {v['mean_em']:.3f}" for k, v in byk.items()))
