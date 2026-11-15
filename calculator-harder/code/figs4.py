"""Figures for the third write-up: the 14-level benchmark (runs/hard)."""
import glob, json, math, os, re
import numpy as np
import mathgen as mg, mathgen3 as m3
import horizon as hz
from charts import *

GRAY, BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, VIOLET = "--base", "--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"
DIR, DATA = "runs/hard", "hard/full.npz"
SIDE = json.load(open(DATA.replace(".npz", ".test.json")))
LV, LEN, NL, OLD = np.array(SIDE["level"]), np.array(SIDE["full_len"]), m3.NLEVELS, m3.OLD
SHORT = [l.split()[0] for l in m3.LEVELS]
NAME = [l.split(" ", 1)[1] for l in m3.LEVELS]

R, LOG, OK = {}, {}, {}
for f in sorted(glob.glob(f"{DIR}/H_*.json")):
    n = os.path.basename(f)[:-5]
    if "." in n:
        continue
    R[n] = json.load(open(f))
    LOG[n] = [json.loads(l) for l in open(f"{DIR}/{n}.jsonl")]
    OK[n] = np.load(f"{DIR}/{n}.ok.npy")
RUNS = [  # (run, label, short label, colour)
    ("H_none_3m", "answer only, no thinking block", "answer only", GRAY),
    ("H_values_3m", "thinking block with values only", "values only", ORANGE),
    ("H_full_3m", "thinking block with full procedures", "full procedures", BLUE),
    ("H_full_3m_30k", "full procedures, 3x the steps", "3x steps", AQUA),
    ("H_full_11m", "full procedures, 3.4x the parameters", "3.4x parameters", VIOLET),
    ("H_um8_3m", "full procedures + 8 filler thinking tokens", "+ 8 thinking tokens", PINK),
]
MAIN = "H_full_3m"


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def have(*names):
    return all(n in R for n in names)


def em(n):
    return R[n]["final"]["em"]


def old_mean(n):
    return float(np.mean(em(n)[:OLD]))


def new_mean(n):
    return float(np.mean(em(n)[OLD:]))


def fpt(d):
    """Forward FLOPs for one token through one layer application (the matmuls: 12 d^2 multiply-adds)."""
    return 24.0 * d * d


def train_pflops(n):
    c = R[n]["compute"]
    return R[n]["args"]["steps"] * c["padded_positions_per_step"] * c["layer_applications"] * 3 * fpt(R[n]["model"]["d"]) / 1e15


def infer_gflops(n, level=None):
    v = R[n]["final"]["tle"] if level is None else R[n]["final"]["tle_level"][level]
    return v * fpt(R[n]["model"]["d"]) / 1e9


_REF = None


def refs():
    """Reference worked solutions of the test problems (the '[...]answer' text of the full version)."""
    global _REF
    if _REF is None:
        Z = np.load(DATA)
        _REF = [""] * len(LV)
        for b in range(len(Z["widths"])):
            for i, row in zip(Z[f"tI{b}"].tolist(), Z[f"tT{b}"].tolist()):
                _REF[i] = m3.decode(row[: row.index(mg.EOS)] if mg.EOS in row else row)
    return _REF


def ceiling():
    """Per level: share of test problems where the reference solution ends in the true answer."""
    agree = np.array([r.split("]")[-1] == t for r, t in zip(refs(), SIDE["truth"])])
    return [float(agree[LV == l].mean()) for l in range(NL)]


def example(l, lo=0.3, hi=0.6, max_text=44):
    """A typical test problem of a level: mid-length solution, short enough to print."""
    idx = np.where(LV == l)[0]
    a, b = np.quantile(LEN[idx], [lo, hi])
    cand = [i for i in idx if a <= LEN[i] <= b and len(SIDE["text"][i]) <= max_text] or list(idx)
    return cand[0]


def tier_table():
    ceil = ceiling()
    rows = []
    for l in range(NL):
        i = example(l)
        L = LEN[LV == l]
        rows.append([SHORT[l], NAME[l], f"<code>{esc(SIDE['text'][i])}</code> &rarr; <code>{esc(SIDE['truth'][i])}</code>", f"{L.mean():.0f}",
                     f"{int(np.percentile(L, 99)):,}", pct(ceil[l])])
    return table(["", "level", "a test problem", "solution tokens, mean", "99th percentile", "ceiling"], rows, num_from=3)


def worked_examples():
    out = []
    for l in range(OLD, NL):
        idx = np.where(LV == l)[0]
        i = min((j for j in idx if LEN[j] >= np.quantile(LEN[idx], 0.1)), key=lambda j: LEN[j])
        out.append(f"<details class=tv><summary>{esc(m3.LEVELS[l])}: <code>{esc(SIDE['text'][i])}</code> ({LEN[i]} tokens)</summary>"
                   f"<pre><code>{esc(SIDE['text'][i])}={esc(refs()[i]).replace(';', '; ')}</code></pre></details>")
    return "".join(out)


def fig_zero():
    z = jload(f"{DIR}/old_11m.hard.json")
    if not z:
        return ""
    items = [{"name": m3.LEVELS[l], "v": v, "c": GRAY if l < OLD else BLUE, "tip": [("problems", f"{z['n'][l]:,}")]} for l, v in enumerate(z["em"]) if v is not None]
    body = legend([("levels it was trained on", GRAY, "box"), ("new tiers in the same grammar, never trained on", BLUE, "box")]) + hbar_chart(items, label_w=150)
    return figure("The 99.1% model, zero-shot on the levels it can read",
                  "Exact match of the 10.7M-parameter model from the previous write-up. L10 to L14 use characters it has never seen (x, y, z, k) "
                  "and are left out.", body)


def run_heat(names=None, zero=True):
    names = [n for n, *_ in RUNS if n in R] if names is None else [n for n in names if n in R]
    lab = {n: l for n, l, *_ in RUNS}
    vals = [em(n) + [old_mean(n), new_mean(n)] for n in names]
    rows = [f"{lab[n]} ({R[n]['model']['params']/1e6:.1f}M, {R[n]['args']['steps']//1000}k steps)" for n in names]
    z = jload(f"{DIR}/old_11m.hard.json") if zero else None
    if z:
        rows.insert(0, "previous write-up's model, zero-shot (10.7M)")
        vals.insert(0, z["em"] + [float(np.mean(z["em"][:OLD])), None])
    return heat_table(rows, SHORT + ["L1\u20117", "L8\u201114"], vals, first_col="run")


def ans_curve(n):
    """[(step, [accuracy per level])] on the 200-per-level subset."""
    return [(r["step"], r["ans"]) for r in LOG.get(n, []) if "ans" in r]


def fig_saturation(n):
    if n not in LOG:
        return ""
    cur = ans_curve(n)
    xs = [s for s, _ in cur]
    cols = [BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, VIOLET]
    series = [{"name": "L1 to L7, mean", "short": "L1-L7", "c": GRAY, "y": [float(np.mean(v[:OLD])) for _, v in cur], "dash": True}]
    series += [{"name": m3.LEVELS[l], "short": SHORT[l], "c": cols[l - OLD], "y": [v[l] for _, v in cur]} for l in range(OLD, NL)]
    k = max(1, len(xs) // 6)
    return figure(f"Held-out accuracy during training, level by level ({R[n]['model']['params']/1e6:.1f}M parameters, {R[n]['args']['steps']//1000}k steps)",
                  "Exact match of the final answer on 200 held-out problems per level, generated greedily. The seven old levels are averaged into the dashed line.",
                  line_chart(series, xs, [f"step {s/1000:g}k" for s in xs], h=340, xtick_idx=list(range(k - 1, len(xs), k)), right=70))


def steps_to(n, thr):
    """First evaluation step at which each level's subset accuracy reaches thr (None if never)."""
    out = []
    for l in range(NL):
        hit = [s for s, v in ans_curve(n) if v[l] >= thr]
        out.append(hit[0] if hit else None)
    return out


def saturation_table(n, thrs=(0.5, 0.9, 0.95)):
    if n not in R:
        return ""
    ceil = ceiling()
    cols = [steps_to(n, t) for t in thrs]
    f = lambda s: "not reached" if s is None else f"{s:,}"
    rows = [[m3.LEVELS[l], pct(ceil[l]), pct(em(n)[l])] + [f(c[l]) for c in cols] for l in range(NL)]
    return table(["level", "ceiling", "final exact match"] + [f"steps to {100*t:.0f}%" for t in thrs], rows)


def bin_labels():
    return [f"{lo}-{hi - 1}" for lo, hi in zip(hz.EDGES, hz.EDGES[1:])]


def fig_horizon(names=None):
    names = [n for n, *_ in RUNS if n in OK and n != "H_um8_3m"] if names is None else [n for n in names if n in OK]
    if not names:
        return ""
    info = {n: (l, s, c) for n, l, s, c in RUNS}
    labs = bin_labels()
    series = []
    for n in names:
        acc = {lo: a for lo, hi, cnt, a in hz.bins(LEN, OK[n])}
        series.append({"name": info[n][0], "short": info[n][1], "c": info[n][2], "y": [acc.get(lo) for lo in hz.EDGES[:-1]]})
    count = {lo: cnt for lo, hi, cnt, a in hz.bins(LEN, np.ones(len(LEN)))}
    chart = line_chart(series, labs, labs, h=340, xtick_idx=list(range(1, len(labs), 2)), xlab="tokens in the reference worked solution", right=120,
                       tv_header="solution tokens")
    note = "Problems per bin: " + ", ".join(f"{l}: {count.get(lo, 0):,}" for l, lo in zip(labs, hz.EDGES[:-1]) if count.get(lo))
    return figure("Exact match against the length of the worked solution",
                  "All 28,000 test problems, binned by the number of tokens in their reference solution (half-octave bins). "
                  "For the answer-only and values-only models the x axis is still the full solution's length: it measures the problem, not what the model wrote.",
                  chart + f'<p class="cap" style="margin-top:8px">{note}</p>')


def horizons(n, qs=(0.9, 0.5)):
    return [hz.horizon(LEN, OK[n], q) for q in qs]


def fmt_h(h):
    if h is None:
        return "no fall-off"
    return f"{h:,.0f}" if h <= 2304 else "beyond 2,304"


def horizon_table(names=None):
    names = [n for n, *_ in RUNS if n in OK] if names is None else [n for n in names if n in OK]
    lab = {n: l for n, l, *_ in RUNS}
    rows = []
    for n in names:
        h90, h50 = horizons(n)
        rows.append([lab[n], f"{R[n]['model']['params']/1e6:.1f}M", f"{R[n]['args']['steps']//1000}k", pct(float(np.mean(em(n)))), fmt_h(h90), fmt_h(h50)])
    return table(["run", "parameters", "steps", "exact match, 14 levels", "90% horizon (tokens)", "50% horizon (tokens)"], rows)


def horizon_curve(n, q=0.5):
    """[(step, horizon)] over training, from the per-problem outcomes of the 200-per-level subset."""
    idx = hz.subset_index(LV, R[n]["args"]["gen_n"]) if n in R else hz.subset_index(LV, 200)
    out = []
    for r in LOG.get(n, []):
        if "ok" in r:
            ok = np.frombuffer(r["ok"].encode(), np.uint8) == ord("1")
            out.append((r["step"], hz.horizon(LEN[idx], ok, q)))
    return out


def fig_horizon_steps(names=("H_full_3m", "H_full_3m_30k", "H_full_11m", "H_values_3m"), q=0.9):
    names = [n for n in names if n in LOG]
    if not names:
        return ""
    info = {n: (l, s, c) for n, l, s, c in RUNS}
    xs = sorted({s for n in names for s, _ in horizon_curve(n, q)})
    series = []
    for n in names:
        cur = dict(horizon_curve(n, q))
        series.append({"name": info[n][0], "short": info[n][1], "c": info[n][2], "y": [min(max(cur[s], 8.0), 2304.0) if cur.get(s) else None for s in xs]})
    k = max(1, len(xs) // 6)
    return figure(f"The {100*q:.0f}% horizon during training",
                  f"Solution length (tokens, log scale) at which the fitted accuracy crosses {100*q:.0f}%, on the 200-per-level subset scored during training. "
                  "Capped at 2,304 tokens, the longest solution in the benchmark.",
                  line_chart(series, xs, [f"step {s/1000:g}k" for s in xs], h=300, ylog=True, ymin=8, ymax=3000, yticks=(10, 100, 1000),
                             yfmt=lambda v: f"{v:,.0f}", vfmt=lambda v: f"{v:,.0f}", xtick_idx=list(range(k - 1, len(xs), k)), right=130, vlines=()))


def errs(n):
    return jload(f"{DIR}/{n}.errs.json")


def err_table(n):
    e = errs(n)
    if not e:
        return ""
    rows = []
    for l in range(OLD, NL):
        d = e["levels"][m3.LEVELS[l]]
        top = sorted(d["kinds"].items(), key=lambda kv: -kv[1])[:3]
        rows.append([m3.LEVELS[l], f"{d['failed']:,} of {d['n']:,}", "; ".join(f"{k} ({pct(v / max(d['failed'], 1), 0)})" for k, v in top)])
    return table(["level", "wrong answers", "where the worked solution first goes wrong (share of the level's failures)"], rows, num_from=1)


def fig_err_kinds(n):
    e = errs(n)
    if not e:
        return ""
    tot = {}
    for l in range(OLD, NL):
        for k, v in e["levels"][m3.LEVELS[l]]["kinds"].items():
            tot[k] = tot.get(k, 0) + v
    s = sum(tot.values())
    items = [{"name": k, "v": v / s, "c": BLUE, "tip": [("failures", f"{v:,}")]} for k, v in sorted(tot.items(), key=lambda kv: -kv[1])]
    return figure("What goes wrong on the new tiers", f"First fault in the model's own worked solution, over the {s:,} wrong answers on L8 to L14.",
                  hbar_chart(items, label_w=300, xmax=max(i["v"] for i in items) * 1.15, xticks=[t / 10 for t in range(0, 11) if t / 10 <= max(i["v"] for i in items) * 1.15]))


def kind_table(names):
    names = [n for n in names if errs(n)]
    if not names:
        return ""
    lab = {n: s for n, l, s, c in RUNS}
    per = [{(k["level"], k["kind"]): k for k in errs(n)["kinds"]} for n in names]
    rows = [[lv, kind, f"{per[0][(lv, kind)]['n']:,}"] + [pct(p[(lv, kind)]["em"]) for p in per] for lv, kind in per[0]]
    return table(["level", "sub-type", "test problems"] + [lab[n] for n in names], rows, num_from=2)


def fig_variants(names=("H_none_3m", "H_values_3m", "H_full_3m")):
    names = [n for n in names if n in R]
    if len(names) < 2:
        return ""
    info = {n: (l, s, c) for n, l, s, c in RUNS}
    items = []
    for n in names:
        items.append({"name": info[n][0] + ": L1-7", "v": old_mean(n), "c": GRAY})
        items.append({"name": "L8-14", "v": new_mean(n), "c": BLUE})
    body = legend([("the seven old levels", GRAY, "box"), ("the seven new tiers", BLUE, "box")]) + hbar_chart(items, label_w=330)
    return figure("What the model writes before the answer, on old and new levels",
                  "Exact match, 3.2M parameters, 10,000 steps, 2,000 test problems per level.", body)


def compute_table():
    lab = {n: l for n, l, *_ in RUNS}
    rows = []
    for n, *_ in RUNS:
        if n not in R:
            continue
        c = R[n]["compute"]
        rows.append([lab[n], f"{R[n]['model']['params']/1e6:.1f}M", f"{R[n]['args']['steps']//1000}k", f"{c['mean_target_tokens']:.0f}",
                     f"{c['padded_positions_per_step']/1e3:.0f}k", f"{train_pflops(n):.0f}", f"{R[n]['train_time']/3600:.1f}", f"{infer_gflops(n):.2f}",
                     pct(old_mean(n)), pct(new_mean(n))])
    return table(["run", "parameters", "steps", "target tokens / row", "positions / step", "train PFLOP", "hours (shared GPU)", "infer GFLOP / problem",
                  "L1-7", "L8-14"], rows)


def level_compute_table(n):
    if n not in R:
        return ""
    f = R[n]["final"]
    rows = [[m3.LEVELS[l], f"{f['gen_tokens_level'][l]:.0f}", f"{infer_gflops(n, l):.2f}", pct(f["em"][l])] for l in range(NL)]
    return table(["level", "tokens written", "GFLOP per problem", "exact match"], rows)


# ------------------------------------------------------------------ stretch set: beyond the training range
STRETCH = jload("hard/stretch.json")
STRETCH_WHAT = [
    ("sums of 8-14 digit numbers, products of 4-8 digit numbers", "15-20 digit sums, 9-11 digit products, longer divisions"),
    ("4 to 9 operations", "10 to 14 operations"),
    ("2 to 7 terms", "9 to 13 terms"),
    ("coefficients up to about 40", "coefficients up to 400"),
    ("2x2, or 3x3 with one-digit coefficients", "3x3 with two-digit coefficients"),
    ("3 to 12 terms", "13 to 24 terms"),
    ("solutions up to 2,304 tokens (mostly 4 to 6 iterations)", "solutions of 2,306 to 4,037 tokens (6 to 8 iterations)"),
]


def stretch(n):
    return jload(f"{DIR}/{n}.stretch.json")


def stretch_em(n):
    s = stretch(n)
    return [s["em"][m3.LEVELS[l]] for l in range(OLD, NL)]


def stretch_def_table():
    if not STRETCH:
        return ""
    slv, slen = np.array(STRETCH["level"]), np.array(STRETCH["full_len"])
    rows = [[m3.LEVELS[l], STRETCH_WHAT[l - OLD][0], STRETCH_WHAT[l - OLD][1], f"{LEN[LV == l].mean():.0f} \u2192 {slen[slv == l].mean():.0f}"]
            for l in range(OLD, NL)]
    return table(["level", "in the training pool", "stretch set", "solution tokens, mean"], rows, num_from=3)


def stretch_table(names):
    names = [n for n in names if stretch(n)]
    if not names or not STRETCH:
        return ""
    lab = {n: f"{R[n]['model']['params']/1e6:.1f}M, {R[n]['args']['steps']//1000}k steps" for n in names}
    rows = []
    for l in range(OLD, NL):
        r = [m3.LEVELS[l]]
        for n in names:
            r += [pct(em(n)[l]), pct(stretch_em(n)[l - OLD])]
        rows.append(r)
    rows.append(["mean"] + [x for n in names for x in (pct(new_mean(n)), pct(stretch_mean(n)))])
    head = ["level"]
    for n in names:
        head += [f"{lab[n]}: in range", "stretch"]
    return table(head, rows)


E2 = hz.EDGES[:-1] + [2305, 3259, 4609, 6001]


def fig_stretch_len(n):
    s = stretch(n)
    if not s or n not in OK:
        return ""
    new = LV >= OLD
    sl, sok = np.array(STRETCH["full_len"]), np.array(s["ok"], float)
    labs = [f"{lo}-{hi - 1}" for lo, hi in zip(E2, E2[1:])]

    def binned(lens, ok):
        out = []
        for lo, hi in zip(E2, E2[1:]):
            m = (lens >= lo) & (lens < hi)
            out.append(float(ok[m].mean()) if m.sum() >= 15 else None)
        return out

    a, b = binned(LEN[new], OK[n][new].astype(float)), binned(sl, sok)
    keep = [i for i in range(len(labs)) if a[i] is not None or b[i] is not None]
    series = [{"name": "test problems inside the training range (L8-L14)", "short": "in range", "c": BLUE, "y": [a[i] for i in keep]},
              {"name": "stretch problems, beyond the training range", "short": "stretch", "c": ORANGE, "y": [b[i] for i in keep]}]
    xl = [labs[i] for i in keep]
    cut = next((j for j, i in enumerate(keep) if E2[i] >= 2305), None)
    return figure("Inside and beyond the training range, by solution length",
                  f"{R[n]['model']['params']/1e6:.1f}M parameters, {R[n]['args']['steps']//1000}k steps. Bins with fewer than 15 problems are left out. "
                  "No training solution is longer than 2,304 tokens.",
                  line_chart(series, xl, xl, h=320, xtick_idx=list(range(0, len(xl), 2)), xlab="tokens in the reference worked solution", right=80,
                             vlines=[(cut - 0.5, "longest in training")] if cut else (), tv_header="solution tokens"))


def stretch_mean(n):
    return float(np.mean(stretch_em(n)))


def fig_stretch_bars(names):
    names = [n for n in names if stretch(n)]
    if not names:
        return ""
    lab = {n: l for n, l, s, c in RUNS}
    items = []
    for n in names:
        items.append({"name": f"{lab[n]}: in range", "v": new_mean(n), "c": GRAY})
        items.append({"name": "one size up", "v": stretch_mean(n), "c": BLUE})
    body = legend([("test problems inside the training range", GRAY, "box"), ("stretch problems, one size beyond it", BLUE, "box")]) + hbar_chart(items, label_w=330)
    return figure("The seven new tiers, inside the training range and one size up", "Exact match, mean of L8 to L14. In range: 2,000 test problems per tier. "
                  "Stretch: 300 per tier.", body)


def stretch_kinds_table(n):
    s = stretch(n)
    if not s or "kinds" not in s:
        return ""
    slv = np.array(STRETCH["level"])
    rows = []
    for l in range(OLD, NL):
        d = s["kinds"][m3.LEVELS[l]]
        tot = sum(d.values())
        top = sorted(d.items(), key=lambda kv: -kv[1])[:3]
        pos, frac = stretch_fault_stats(n, l)
        rows.append([m3.LEVELS[l], f"{int(round((1 - s['em'][m3.LEVELS[l]]) * (slv == l).sum()))} of {(slv == l).sum()}",
                     f"{100 * float(np.median(frac)):.0f}%" if len(frac) else "", "; ".join(f"{k} ({pct(v / max(tot, 1), 0)})" for k, v in top)])
    return table(["level", "wrong answers", "first fault, share of the way through (median)", "what the first fault is (share of the level's failures)"],
                 rows, num_from=1)


def stretch_fault_stats(n, l):
    """Token offsets of the first fault in the wrong stretch solutions of a level, and the same as a fraction of the reference length."""
    s = stretch(n)
    slv, sl = np.array(STRETCH["level"]), np.array(STRETCH["full_len"])
    idx = [j for j in np.where(slv == l)[0] if s.get("fault_pos") and s["fault_pos"][j] is not None]
    pos = np.array([s["fault_pos"][j] for j in idx], float)
    return pos, pos / sl[idx] if len(idx) else pos


def stretch_formats_table(names):
    names = [n for n in names if stretch(n)]
    if len(names) < 2:
        return ""
    lab = {n: s for n, l, s, c in RUNS}
    rows = [[m3.LEVELS[l]] + [pct(stretch_em(n)[l - OLD]) for n in names] for l in range(OLD, NL)]
    rows.append(["mean"] + [pct(stretch_mean(n)) for n in names])
    return table(["stretch set, exact match"] + [lab[n] for n in names], rows)


KNOBS = [(8, "operations"), (9, "terms (a bracketed term counts as two)"), (12, "terms"), (13, "Newton iterations")]


def knob_acc(kinds, levels, ok, l, min_n=15):
    """{knob value: accuracy} for one level, from sub-type labels that carry a number ('7 operations', 'cubic, 5 iterations')."""
    out = {}
    for k, lv_, o in zip(kinds, levels, ok):
        if lv_ == l:
            out.setdefault(int(re.findall(r"\d+", k)[-1]), []).append(bool(o))
    return {v: float(np.mean(x)) for v, x in out.items() if len(x) >= min_n}


def fig_knobs(n):
    s = stretch(n)
    if not s or n not in OK:
        return ""
    panels, rows = [], []
    for l, unit in KNOBS:
        a = knob_acc(SIDE["kind"], SIDE["level"], OK[n], l)
        b = knob_acc(STRETCH["kind"], STRETCH["level"], s["ok"], l)
        xs = sorted(set(a) | set(b))
        series = [{"name": "inside the training range", "c": BLUE, "y": [a.get(x) for x in xs]}, {"name": "stretch", "c": ORANGE, "y": [b.get(x) for x in xs]}]
        chart = line_chart(series, xs, [str(x) for x in xs], w=400, h=190, end_labels=False, show_legend=False, tv=False, xlab=unit,
                           xtick_idx=list(range(0, len(xs), 2 if len(xs) > 9 else 1)))
        panels.append(f'<div><div style="font-size:.85em;color:var(--ink2);margin-top:10px"><b>{esc(m3.LEVELS[l])}</b></div>{chart}</div>')
        rows += [[m3.LEVELS[l], f"{x} {unit}", "" if a.get(x) is None else pct(a[x]), "" if b.get(x) is None else pct(b[x])] for x in xs]
    body = legend([("inside the training range", BLUE, "line"), ("stretch", ORANGE, "line")]) + '<div class="grid2">' + "".join(panels) + "</div>" + \
        table_view(["level", "size", "in range", "stretch"], rows)
    return figure("Accuracy against each tier's size knob",
                  f"{R[n]['model']['params']/1e6:.1f}M parameters, {R[n]['args']['steps']//1000}k steps. Blue: held-out problems of the sizes in the training pool. "
                  "Orange: sizes the pool never contained. Newton stretch problems are the runs whose solution exceeds the 2,304-token limit of the "
                  "pool, so the two series overlap in iteration count. Sizes with fewer than 15 problems are left out.", body)


def sum_stop_stats(n):
    """Stretch sums: share of the failures that are a correct prefix cut short, and how many terms those failures wrote (median)
    against how many the problem has (median)."""
    s = stretch(n)
    slv, sl = np.array(STRETCH["level"]), np.array(STRETCH["full_len"])
    m = slv == 12
    ok, gen = np.array(s["ok"], bool)[m], np.array(s["gen_len"])[m]
    terms = np.array([int(k.split()[0]) for k in np.array(STRETCH["kind"])[m]])
    written = terms * gen / sl[m]
    kinds = s["kinds"][m3.LEVELS[12]]
    return kinds.get("departs at extra or missing steps", 0) / max(sum(kinds.values()), 1), float(np.median(written[~ok])), float(np.median(terms[~ok]))


def err_shares(n):
    """Shares of the first-fault kinds over the wrong answers on the new tiers: (arithmetic, wrong operands, recalled value, reference rounding)."""
    e, tot = errs(n), {}
    for l in range(OLD, NL):
        for k, v in e["levels"][m3.LEVELS[l]]["kinds"].items():
            tot[k] = tot.get(k, 0) + v
    s = sum(tot.values())
    pick = lambda key: sum(v for k, v in tot.items() if key in k) / s
    return pick("wrong arithmetic"), pick("operands of a step"), pick("looked-up"), pick("follows the reference")


# ------------------------------------------------------------------ sine probe
SIN = jload(f"{DIR}/sin_probe.json")
SIN_MODELS = [("old", "previous write-up's model (10.7M, old 7 levels)", "old model", GRAY), ("small30k", "3.2M, 30k steps", "3.2M", AQUA),
              ("big", "10.7M, 10k steps", "10.7M", VIOLET)]


def _num(a):
    try:
        return float(a)
    except ValueError:
        return None


def sin_stats(name="big"):
    """In-range and out-of-range behaviour of one model on sin(x), x on the 0.01 grid in [-40, 40]."""
    x, truth = np.array(SIN["x"]), SIN["truth"]
    k = np.round(x * 100).astype(int)
    ok = np.array(SIN["models"][name]["ok"])
    val = np.array([np.nan if _num(a) is None else _num(a) for a in SIN["models"][name]["ans"]])
    valid = np.array([t is not None for t in truth])
    out = np.abs(k) > 999
    err = np.abs(val - np.sin(x))
    grid = set(t for t, kk in zip(truth, k) if abs(kk) <= 999)
    return {"in": float(ok[~out & valid].mean()), "n_in": int((~out).sum()), "out": float(ok[out & valid].mean()), "n_out": int(out.sum()),
            "near": float(ok[out & (np.abs(k) <= 2000) & valid].mean()), "far": float(ok[(np.abs(k) > 2000) & valid].mean()),
            "ints": float(ok[out & (k % 100 == 0) & valid].mean()), "n_ints": int((out & (k % 100 == 0)).sum()),
            "non_ints": float(ok[out & (k % 100 != 0) & valid].mean()),
            "err_in": float(np.nanmean(err[~out])), "err_out": float(np.nanmean(err[out])),
            "bounded": float(np.nanmean(np.abs(val[out]) <= 1)), "table": float(np.mean([a in grid for a in np.array(SIN["models"][name]["ans"])[out]]))}


def fig_sin_curve(name="big", lim=30, step=10):
    """The model's answers to sin(x) as dots over the true curve; the range asked for directly in training is shaded."""
    if not SIN:
        return ""
    x, truth, ans = SIN["x"], SIN["truth"], SIN["models"][name]["ans"]
    idx = [i for i in range(len(x)) if abs(x[i]) <= lim and round(x[i] * 100) % step == 0]
    w, h = 760, 330
    L, R_, T, B = 40, 14, 24, 34
    pw, ph = w - L - R_, h - T - B
    X = lambda v: L + pw * (v + lim) / (2 * lim)
    Y = lambda v: T + ph * (1 - (max(-1.15, min(1.15, v)) + 1.15) / 2.3)
    col = dict((m[0], m[3]) for m in SIN_MODELS)[name]
    o = [f'<rect x="{X(-9.99):.1f}" y="{T}" width="{X(9.99) - X(-9.99):.1f}" height="{ph}" fill="var(--grid)" opacity=".45"/>',
         f'<text x="{X(0):.1f}" y="{T - 8}" text-anchor="middle" class="lab">asked for directly in training: -9.99 to 9.99</text>']
    for t in (-1, -0.5, 0, 0.5, 1):
        o.append(f'<line class="grid" x1="{L}" x2="{L + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text x="{L - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in range(-lim, lim + 1, 10):
        o.append(f'<text x="{X(t):.1f}" y="{T + ph + 15}" text-anchor="middle">{t}</text>')
    o.append(f'<text x="{L + pw / 2:.1f}" y="{h - 3}" text-anchor="middle" class="lab">x</text>')
    fine = [-lim + j * 0.05 for j in range(int(2 * lim / 0.05) + 1)]
    o.append(f'<path class="ln" d="M{" L".join(f"{X(v):.1f},{Y(math.sin(v)):.1f}" for v in fine)}" stroke="var({GRAY})"/>')
    for i in idx:
        v = _num(ans[i])
        if v is not None:
            o.append(f'<circle cx="{X(x[i]):.1f}" cy="{Y(v):.1f}" r="2.6" fill="var({col})" stroke="var(--surface)" stroke-width="1"/>')
    o.append(f'<line class="xh" x1="0" x2="0" y1="{T}" y2="{T + ph}"/>')
    data = {"w": w, "px": [round(X(x[i]), 1) for i in idx], "xl": [SIN["text"][i] for i in idx],
            "s": [{"n": "true value", "c": GRAY, "v": [truth[i] or "0" for i in idx]}, {"n": "the model's answer", "c": col, "v": [ans[i] for i in idx]}]}
    svg = f'<svg viewBox="0 0 {w} {h}" role="img" data-xh="{esc(json.dumps(data))}">' + "".join(o) + "</svg>"
    xa, k = np.array(x), np.round(np.array(x) * 100).astype(int)
    ok = np.array(SIN["models"][name]["ok"])
    val = np.array([np.nan if _num(a) is None else _num(a) for a in ans])
    err = np.abs(val - np.sin(xa))
    rows = []
    for lo in range(0, 40, 5):
        m = (np.abs(k) >= lo * 100) & (np.abs(k) < (lo + 5) * 100) if lo else (np.abs(k) < 500)
        rows.append([f"{lo} to {lo + 5}", f"{int(m.sum()):,}", pct(float(ok[m].mean())), f"{float(np.nanmean(err[m])):.4f}"])
    lab = dict((m[0], m[1]) for m in SIN_MODELS)[name]
    return figure("The model's sin(x) against the real one",
                  f"{lab}. One dot per 0.{step:02d} of x. A right answer sits on the curve. Hover for the values.",
                  legend([("sin(x)", GRAY, "line"), ("the model's answer", col, "box")]) + svg +
                  table_view(["|x|", "problems (0.01 grid)", "exact match", "mean |answer - sin x|"], rows))


SINW = jload(f"{DIR}/sin_probe_wide.json")
SIND = jload(f"{DIR}/sin_density.json")
SIN_EDGES = [10, 20, 40, 70, 100, 150, 200, 300]


def _sin_arrays(S, name):
    x = np.array(S["x"])
    val = np.array([np.nan if _num(a) is None else _num(a) for a in S["models"][name]["ans"]])
    return x, val, np.array(S["models"][name]["ok"]), np.array([t is not None for t in S["truth"]])


def sin_bins(name):
    """Per range of x, from far negative to far positive: (label, key, exact, within 0.1, sign right, n).
    The inside bin uses the 0.01 grid; the outside bins the wide probe (every 0.07 out to +-300)."""
    out = []
    xw, vw, okw, validw = _sin_arrays(SINW, name)
    tw = np.sin(xw)

    def stat(m, x, v, t, ok):
        mm = m & ~np.isnan(v)
        return float(ok[m].mean()), float(np.mean(np.abs(v[mm] - t[mm]) < 0.1)), float(np.mean(np.sign(v[mm]) == np.sign(t[mm]))), int(m.sum())

    pairs = list(zip(SIN_EDGES, SIN_EDGES[1:]))
    for a, b in reversed(pairs):
        out.append((f"-{b} to -{a}", f"-{a}-{b}") + stat((xw <= -a) & (xw > -b) & validw, xw, vw, tw, okw))
    x, v, ok, valid = _sin_arrays(SIN, name)
    out.append(("-9.99 to 9.99", "inside") + stat((np.abs(x) < 9.995) & valid, x, v, np.sin(x), ok))
    for a, b in pairs:
        out.append((f"{a} to {b}", f"+{a}-{b}") + stat((xw >= a) & (xw < b) & validw, xw, vw, tw, okw))
    return out


def fig_sin_within():
    if not SIN or not SINW:
        return ""
    series, labs = [], None
    for name, lab, short, c in SIN_MODELS:
        b = sin_bins(name)
        labs = [r[0] for r in b]
        series.append({"name": lab, "short": short, "c": c, "y": [r[3] for r in b]})
    mid = [i for i, l in enumerate(labs) if l.startswith("-9.99")][0]
    return figure("How close the answer is, by where x is",
                  "Share of answers within 0.1 of sin(x). Answers unrelated to x would score about 12%. The middle bin is the range asked for "
                  "directly in training; the bins get wider away from it.",
                  line_chart(series, labs, labs, h=300, xtick_idx=[0, 2, 4, mid, 10, 12, 14], xlab="x", right=96, tv_header="x"))


def sin_table(name="big"):
    if not SIN or not SINW:
        return ""
    dens = (SIND or {}).get("per_million_composite_problems", {})
    rows = []
    for lab, key, exact, within, sign, n in sin_bins(name):
        d = "every grid point, many times" if key == "inside" else f"{dens.get(key, 0):,.0f}"
        rows.append([lab, d, pct(exact), pct(within, 0), pct(sign, 0)])
    return table(["x", "times sin(x) is recalled per 1M composite training problems", "exact (4 digits)", "within 0.1", "right sign"], rows)


def sin_flip(name, a=10, b=20):
    """For x in (-b, -a]: share of answers within 0.1 of sin(|x|), the answer with the minus sign ignored."""
    x, v, _, _ = _sin_arrays(SINW, name)
    m = (x <= -a) & (x > -b) & ~np.isnan(v)
    return float(np.mean(np.abs(v[m] + np.sin(x[m])) < 0.1))
