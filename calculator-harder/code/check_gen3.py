"""The seven new tiers: agreement of the worked solutions with the true answers (the ceiling a perfect model could reach),
solution lengths, rejection rate, generation speed, and a re-check of every arithmetic step.
Usage: check_gen3.py [problems per tier]"""
import collections, random, sys, time
import numpy as np
import mathgen as mg, mathgen2 as m2, mathgen3 as m3
from stepcheck import check_step

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
print(f"{'tier':22s} agree reject  ms/prob  prompt max   len mean   p50   p90   p99    max  >{m3.T_MAX}  steps ok/bad")
for lv in range(m3.OLD, m3.NLEVELS):
    rng = random.Random(100 + lv)
    agree, lens, plen, rej, n_ok, n_bad, kinds = [], [], [], 0, 0, 0, collections.defaultdict(list)
    t0 = time.time()
    seen = set()
    while len(agree) < N:
        p = m3.sample(lv, rng)
        if p["text"] in seen:
            continue
        seen.add(p["text"])
        try:
            sc, ans = m3.solve(p)
        except mg.Reject:
            rej += 1
            continue
        m3.encode(p["text"] + "=" + "[" + sc + "]" + ans)
        assert all(c in m3.STOI for c in p["text"] + sc + ans + p["truth"]), p["text"]
        agree.append(ans == p["truth"]); kinds[p["kind"]].append(ans == p["truth"])
        lens.append(len(sc) + 2 + len(ans) + 1); plen.append(len(p["text"]) + 2)
        if ans != p["truth"] and sum(1 for a in agree if not a) <= 3:
            print("   mismatch:", p["text"], "truth", p["truth"], "chain", ans)
        for st in sc.split(";"):
            r = check_step(st)
            if r is not None:
                n_ok += r; n_bad += not r
                if not r and n_bad <= 3: print("   BAD STEP:", st, "in", p["text"])
    L = np.array(lens)
    print(f"{m3.LEVELS[lv]:22s} {np.mean(agree):.4f} {rej/(rej+N):.3f} {1000*(time.time()-t0)/(N+rej):7.2f}  {max(plen):6d}      {L.mean():8.1f} {int(np.percentile(L,50)):5d} {int(np.percentile(L,90)):5d} {int(np.percentile(L,99)):5d} {L.max():6d}  {(L > m3.T_MAX).mean():.3f}  {n_ok}/{n_bad}")
    for k in sorted(kinds, key=lambda k: (len(k), k)):
        print(f"      {k:20s} n={len(kinds[k]):5d}  agree {np.mean(kinds[k]):.3f}")
