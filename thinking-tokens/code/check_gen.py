"""Cross-check: printed string re-parses to the same value; Python's own evaluator agrees."""
import math, random, re, collections
import mathgen as mg

rng = random.Random(123)
PY = {"sqrt": "math.sqrt", "asin": "math.asin", "acos": "math.acos", "atan": "math.atan", "sin": "math.sin",
      "cos": "math.cos", "tan": "math.tan", "log": "math.log10", "ln": "math.log", "abs": "abs", "gcd": "math.gcd",
      "lcm": "math.lcm", "mod": "(lambda a,b:a%b)", "pi": "math.pi", "e": "math.e"}
tok = re.compile("|".join(mg.NAMES) + r"|\d+\.?\d*|.")
n_round = n_py = 0
lens = collections.defaultdict(lambda: [0, 0, 0])
test_frac = collections.Counter()
for lv in range(mg.NLEVELS):
    for _ in range(40000):
        e, a, sc = mg.sample(lv, rng)
        assert mg.calc(e) == a, (e, a, mg.calc(e))
        n_round += 1
        L = lens[lv]
        L[0] = max(L[0], len(e) + 2); L[1] = max(L[1], len(a) + 1); L[2] = max(L[2], len(a) + 3 + sum(len(x) + 1 for x in sc))
        test_frac[lv] += mg.is_test(e)
        if "!" in e or "nCr" in e or "nPr" in e:
            continue
        py = "".join(PY.get(t, "**" if t == "^" else t) for t in tok.findall(e))
        py = re.sub(r"(?<![\d.])0+(\d)", r"\1", py)
        v = eval(py)
        want = float(a)
        assert abs(v - want) <= 5.01e-4 * max(abs(v), 1e-12) + 1e-12 or abs(v) < 1e-9, (e, a, v)
        n_py += 1
print("round-trip ok:", n_round, " python-eval ok:", n_py)
for lv in range(mg.NLEVELS):
    print(mg.LEVELS[lv], "max prompt/ans/cot len:", lens[lv], "test frac: %.3f" % (test_frac[lv] / 40000))
assert all(set(c) <= set(mg.CHARS) for c in ["sincostanlogsqrtabsgcdlcmmodpie", "nCrnPr"])
