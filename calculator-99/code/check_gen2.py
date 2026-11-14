"""v2 scratchpads on the 14k test problems: agreement with the true answers, length statistics, and an independent
re-check of every arithmetic claim inside the traces."""
import collections, math, re, sys
from decimal import Decimal as D
import numpy as np
import mathgen as mg, mathgen2 as m2
from subtypes import subtype

Dz = np.load("data.npz")
lv = Dz["test_level"]
exprs = [mg.decode(r)[:-1] for r in Dz["test_P"]]
answers = [mg.decode(r[: list(r).index(mg.EOS)]) for r in Dz["test_A"]]

agree = collections.defaultdict(list); rej = collections.Counter(); lens = collections.defaultdict(list)
bad_examples = collections.defaultdict(list); sub_agree = collections.defaultdict(list)
n_claims = collections.Counter(); n_bad = collections.Counter()
num = r"\(?-?[0-9.]+\)?"
val = lambda t: D(t.strip("()"))


def close6(a, b):
    """equal up to a couple of units in the 6th significant digit"""
    return a == b or abs(a - b) <= max(abs(a), abs(b)) * D("3e-5") or abs(a - b) < D("1e-8")


def check_step(st):
    m = re.fullmatch(rf"({num})([-+*/%])({num})(?::([0-9rp,]*))?(?:=(-?[0-9.]+))?", st)
    if not m:
        return None
    a, op, b, work, v = val(m.group(1)), m.group(2), val(m.group(3)), m.group(4), m.group(5)
    if op in "+-":
        return close6(a + b if op == "+" else a - b, D(v))
    if op == "*":
        ok = close6(a * b, D(v))
        if work:
            items = [int(x) for x in work.split(",")]
            (na, _), (nb, _) = m2.sig_int(a), m2.sig_int(b)
            ok &= items[-1] == na * nb
        return ok
    steps = []
    for t in work.split(",") if work else []:
        head, rem = t.split("r")
        d = int(head.split("p")[0])
        if "p" in head and int(head.split("p")[1]) != d * (int(b) if op == "%" else m2.sig_int(b)[0]):
            return False  # the written product must be digit * divisor
        steps.append((d, int(rem)))
    if op == "%":
        ok = steps[-1][1] == int(a) % int(b)
        q = int("".join(str(d) for d, _ in steps))
        return ok and q == int(a) // int(b)
    if not steps:
        return a == 0
    digs = "".join(str(d) for d, _ in steps)          # quotient digits; compare with the true ratio's leading digits
    true = abs(a / b)
    lead = int(digs)
    want = true.scaleb(-true.adjusted() + len(digs) - 1)   # scaled so it has len(digs) integer digits
    return abs(D(lead) - want) < 1 and close6(a / b, D(v))


for i, (e, a) in enumerate(zip(exprs, answers)):
    l = int(lv[i])
    try:
        sc, ans = m2.solve(e)
    except mg.Reject:
        rej[l] += 1; agree[l].append(False); sub_agree[(l, subtype(l, e))].append(False)
        continue
    agree[l].append(ans == a); sub_agree[(l, subtype(l, e))].append(ans == a)
    lens[l].append(len(sc) + 2 + len(ans) + 1)
    if ans != a and len(bad_examples[l]) < 4:
        bad_examples[l].append((e, a, ans, sc[-90:]))
    for st in sc.split(";") if sc else []:
        r = check_step(st)
        if r is not None:
            n_claims[st[re.search(r"[-+*/%]", st[1:]).start() + 1] if False else "arith"] += 1
            if not r:
                n_bad["arith"] += 1
                if n_bad["arith"] <= 5: print("BAD STEP:", st, "in", e)
print("arithmetic steps re-checked:", n_claims["arith"], " wrong:", n_bad["arith"])
print(f"{'level':18s} agree  reject   len mean  p50  p90  p99  max")
for l in range(mg.NLEVELS):
    L = np.array(lens[l])
    print(f"{mg.LEVELS[l]:18s} {np.mean(agree[l]):.4f} {rej[l]:5d}   {L.mean():7.1f} {int(np.percentile(L,50)):4d} {int(np.percentile(L,90)):4d} {int(np.percentile(L,99)):4d} {L.max():4d}")
print("mean agreement: %.4f" % np.mean([np.mean(agree[l]) for l in range(mg.NLEVELS)]))
allL = np.concatenate([lens[l] for l in range(mg.NLEVELS)])
print("all levels: mean %.1f  share <=32: %.3f <=64: %.3f <=128: %.3f <=256: %.3f <=384: %.3f <=512: %.3f" % (allL.mean(), *[(allL <= w).mean() for w in (32, 64, 128, 256, 384, 512)]))
print("\nsub-skills below 99% agreement:")
for k in sorted(sub_agree):
    if np.mean(sub_agree[k]) < 0.99:
        print(f"  {mg.LEVELS[k[0]][:2]} {k[1]:36s} n={len(sub_agree[k]):5d}  agree {np.mean(sub_agree[k]):.3f}")
for l in bad_examples:
    for x in bad_examples[l]:
        print("  mismatch", mg.LEVELS[l][:2], x)
