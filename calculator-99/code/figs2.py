"""Figures for the second write-up (scaling the calculator past 95%)."""
import glob, json, os
import numpy as np
import mathgen as mg, mathgen2 as m2
import pagedata as pd  # v1 runs (runs/main) and the shared test-set metadata
from charts import *

GRAY, BLUE, ORANGE, AQUA = "--base", "--s1", "--s2", "--s3"


def load_dir(d):
    R, LOG, OK = {}, {}, {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        name = os.path.basename(f)[:-5]
        if "." in name:
            continue
        R[name] = json.load(open(f))
        if os.path.exists(f"{d}/{name}.jsonl"):
            LOG[name] = [json.loads(l) for l in open(f"{d}/{name}.jsonl")]
        if os.path.exists(f"{d}/{name}.ok.npy"):
            OK[name] = np.load(f"{d}/{name}.ok.npy")
    return R, LOG, OK


V2, V2LOG, V2OK = load_dir("runs/v2")
ARCH, _, _ = load_dir("runs/arch")
ABL, ABLLOG, _ = load_dir("runs/v2abl")


def em(R, name):
    return float(np.mean(R[name]["final"]["em_gen"]))


def close(R, name):
    return float(np.mean(R[name]["final"]["close"]))


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def sub_acc(ok):
    return [float(ok[pd.SUB_IDX[k]].mean()) for k in pd.SUBKEYS]


def fig_progress(rows):
    """rows: [(label, exact, within1, gray?)]"""
    items = [{"name": lab, "v": e, "c": GRAY if g else BLUE, "tip": [("within 1%", pct(c))]} for lab, e, c, g in rows]
    return figure("Exact match on the same 14,000 held-out problems", "Level-averaged. Gray: the first write-up's models. Blue: the new worked-solution scratchpad.",
                  hbar_chart(items, label_w=330)) + table(["model", "exact", "within 1%"], [[lab, pct(e), pct(c)] for lab, e, c, _ in rows])


def worked_examples():
    ex = ["8-4818-96", "440*223", "51/437", "mod(9278,85)", "sqrt(1042)", "27^4", "gcd(651,687)", "ln(45.6)", "10^5.33", "9nCr4",
          "cos(30*pi/180)", "sqrt(9^2+31^2)"]
    rows = []
    for e in ex:
        sc, a = m2.solve(e)
        rows.append([f"<code>{esc(e)}</code>", '<code style="word-break:break-all">' + esc(sc).replace(";", ";<wbr>") + "</code>", f"<code>{esc(a)}</code>"])
    return table(["problem", "worked solution the model writes", "answer"], rows, num_from=9)


def fig_subskills(cols):
    """cols: [(short label, per-example ok array)]"""
    vals = [sub_acc(ok) for _, ok in cols]
    rows, groups, notes, data, prev = [], [], [], [], None
    for i, k in enumerate(pd.SUBKEYS):
        if pd.SUB_N[k] < 30:
            continue
        rows.append(k[1]); groups.append(mg.LEVELS[k[0]] if prev != k[0] else None); prev = k[0]
        notes.append(pd.SUB_N[k]); data.append([v[i] for v in vals])
    return figure("Exact match by sub-skill (%)", "Same rows as the first write-up. n = held-out problems of that type.",
                  heat_table(rows, [c for c, _ in cols], data, groups, first_col="sub-skill", notes=notes))


def fig_curves(names):
    """names: [(run, label, color)] -> teacher-forced exact match of the whole worked solution over training."""
    steps = sorted({r["step"] for n, _, _ in names for r in V2LOG.get(n, [])})
    series = []
    for n, lab, c in names:
        if n in V2LOG:
            by = {r["step"]: float(np.mean(r["em"])) for r in V2LOG[n]}
            series.append({"name": lab, "c": c, "y": [by.get(s) for s in steps]})
    if not series:
        return ""
    return figure("Learning curves", "Share of test problems whose entire worked solution is reproduced token for token (teacher-forced), level average.",
                  line_chart(series, steps, [f"step {s//1000}k" for s in steps], h=280, xtick_idx=list(range(1, len(steps), max(1, len(steps) // 6))), right=150))


def fig_ptq(runs):
    """runs: [(quant json path, label, color)]"""
    bits = [8, 6, 5, 4, 3, 2]
    series = []
    for path, lab, c in runs:
        q = jload(path)
        if q:
            series.append({"name": lab, "c": c, "y": [float(np.mean(q[str(b)])) for b in bits], "fp": float(np.mean(q["fp"]))})
    if not series:
        return ""
    cap = "Block weights rounded to b bits after training, no retraining (per-row scale, clipping chosen to minimise squared error). "
    cap += " ".join(f"Full precision: {pct(s['fp'])} ({s['name']})." for s in series)
    return figure("Post-training quantisation", cap, line_chart(series, bits, [f"{b} bits" if b > 2 else "ternary" for b in bits], h=260, right=140))


def abl_table(order):
    rows = []
    for name, lab in order:
        if name in ABL:
            f = ABL[name]["final"]
            tf = float(np.mean(ABLLOG[name][-1]["em"])) if name in ABLLOG else float("nan")
            rows.append([lab, pct(em(ABL, name)), pct(close(ABL, name)), pct(tf)] + [pct(x, 0) for x in f["em_gen"]])
    return table(["variant (10k steps)", "exact", "within 1%", "whole solution"] + [f"L{i+1}" for i in range(7)], rows)


def fig_abl(order, title, cap):
    items = [{"name": lab, "v": em(ABL, name), "c": GRAY if i == 0 else BLUE, "tip": [("within 1%", pct(close(ABL, name)))]}
             for i, (name, lab) in enumerate(order) if name in ABL]
    return figure(title, cap, hbar_chart(items, label_w=300)) if items else ""


def err_table(path):
    e = jload(path)
    if not e:
        return ""
    rows = [[k, n, e["total"][k], pct(n / e["total"][k])] for k, n in sorted(e["first"].items(), key=lambda kv: -kv[1])]
    return table(["first wrong step", "solutions lost", "steps of this kind", "error rate"], rows)
