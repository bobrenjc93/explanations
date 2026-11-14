"""Figures for the write-up, built from whatever runs have finished."""
import random
import numpy as np
import mathgen as mg
import pagedata as pd
from charts import *

GRAY, BLUE, ORANGE, AQUA = "--base", "--s1", "--s2", "--s3"
NAMES = {
    "none": "no thinking", "um8": '8 "um" tokens', "disc8": "8 discrete thinking tokens", "soft8": "8 soft (probability-weighted) tokens",
    "direct8": "8 continuous thoughts, raw hidden state", "cont8": "8 continuous thoughts, dedicated head",
    "cont2": "2 continuous thoughts", "cont4": "4 continuous thoughts", "cont16": "16 continuous thoughts",
    "um2": '2 "um" tokens', "um4": '4 "um" tokens', "um16": '16 "um" tokens', "disc16": "16 discrete thinking tokens",
    "mid8": "8 thoughts, middle layers only", "mid16": "16 thoughts, middle layers only",
    "ut2": "looped transformer, middle block x2", "ut3": "looped transformer, middle block x3",
    "win1": "8 thoughts, 1 memory slot", "win2": "8 thoughts, 2 memory slots", "win4": "8 thoughts, 4 memory slots",
    "slots2": "workspace: 2 vectors x 4 rounds", "slots4": "workspace: 4 vectors x 2 rounds",
    "cot": "written scratchpad (supervised)", "noneL6": "no thinking, 6 layers (+50% params)",
    "cot_ut2": "written scratchpad + looped middle block x2", "cot_cont8": "written scratchpad + 8 continuous thoughts",
    "nocurr_none": "no thinking, no curriculum", "nocurr_cont8": "8 continuous thoughts, no curriculum",
    "nocurr_ut2": "looped transformer, no curriculum", "nocurr_cot": "written scratchpad, no curriculum",
    "coco_none": "scratchpad, then removed", "coco_um8": 'scratchpad distilled into 8 "um" tokens',
    "coco_cont8": "scratchpad distilled into 8 continuous thoughts",
    "krand_cont": "continuous thoughts, K varied in training", "krand_um": '"um" tokens, K varied in training',
}
INTRO = [0, 1250, 2500, 5000, 7500, 10000, 13750]


def level_table():
    rng = random.Random(7)
    rows = []
    blurb = ["1-3 digit operands", "up to 8 digits, or 3-4 operands", "up to 4x3 digits; division to 4 significant digits; mod",
             "2-4 operations with precedence and parentheses", "decimals, negatives, powers, roots, reciprocals",
             "sin cos tan, inverses, ln, log, e^x, 10^x, n!, nCr, nPr, gcd, lcm, pi, e",
             "hypotenuse, compound interest, quadratic formula, identities, degrees, mixed"]
    for l in range(mg.NLEVELS):
        ex = []
        while len(ex) < 3:
            e, a, _ = mg.sample(l, rng)
            if len(e) <= 22 or l == 6:
                ex.append(f"<code>{esc(e)} = {esc(a)}</code>")
        rows.append([mg.LEVELS[l], blurb[l], "<span>" + "<br>".join(ex) + "</span>"])
    return table(["level", "what it adds", "examples (answer shown as the model must print it)"], rows, num_from=9)


def fig_curriculum():
    runs = [(n, lab, c) for n, lab, c in [("none_s0", "no thinking", GRAY), ("um8_s0", '8 "um" tokens', ORANGE), ("cont8_s0", "8 continuous thoughts", BLUE)] if n in pd.R]
    steps = pd.curve(runs[0][0])[0]
    panels = []
    for l in range(mg.NLEVELS):
        series = [{"name": lab, "c": c, "y": [e[l] for e in pd.curve(n)[1]]} for n, lab, c in runs]
        fin = " / ".join(pct(s["y"][-1], 0) for s in series)
        chart = line_chart(series, steps, [f"step {s//1000}k" for s in steps], w=400, h=170, end_labels=False, xtick_idx=[4, 9, 14, 19],
                           vlines=[(INTRO[l] / 1000 - 1, "introduced")] if l else (), show_legend=False, tv=False)
        panels.append(f'<div><div class="lab" style="font-size:.85em;color:var(--ink2);margin-top:8px"><b>{esc(mg.LEVELS[l])}</b> &nbsp;final {fin}</div>{chart}</div>')
    hdr = ["level"] + [lab for _, lab, _ in runs]
    rows = [[mg.LEVELS[l]] + [pct(pd.curve(n)[1][-1][l]) for n, _, _ in runs] for l in range(mg.NLEVELS)]
    body = legend([(lab, c, "line") for _, lab, c in runs]) + '<div class="grid2">' + "".join(panels) + "</div>" + table_view(hdr, rows)
    return figure("Held-out accuracy per level while training (seed 0)",
                  "Each panel is one level; the vertical rule marks the step where it enters the curriculum. "
                  "Exact match on 500 held-out problems per level, checked every 1,000 steps. Final values are listed in panel order of the legend.", body)


def _bars(order, metric="em", label_w=270):
    items = []
    for p in order:
        if pd.has(p):
            v = [float(np.mean(pd.R[n]["final"]["em_gen" if metric == "em" else "close"])) for n in pd.seeds(p)]
            items.append({"name": NAMES[p], "v": float(np.mean(v)), "c": GRAY if p in ("none", "noneL6", "nocurr_none") else BLUE,
                          "dots": v if len(v) > 1 else [], "tip": [("seeds", str(len(v))), ("token-layer evals", f"{pd.tle(p):.0f}")]})
    return hbar_chart(items, label_w=label_w)


def variant_table(order):
    rows = [[NAMES[p], len(pd.seeds(p)), pct(pd.em(p)[0]), pct(pd.close(p)), f"{pd.tle(p):.0f}"] + [pct(x, 0) for x in pd.levels(p)] for p in order if pd.has(p)]
    return table(["variant", "seeds", "exact", "within 1%", "TLE"] + [f"L{i+1}" for i in range(7)], rows)


def fig_media():
    order = ["none", "um8", "disc8", "soft8", "direct8", "cont8"]
    a = figure("Exact match at 8 thinking steps, by what the thought is made of",
               "Averaged over the seven levels (2,000 held-out problems each). Bars are the mean over seeds; dots are the individual seeds. "
               "Same transformer in every row; only what is fed back differs.", _bars(order, "em"))
    b = figure("Within 1% of the true value, same models",
               "Share of problems where the printed number is within 1% of the right one (exact answers included).", _bars(order, "close"))
    return a, b, variant_table(order)


def fig_compute():
    fams = []
    for key, lab, c, ks in [("um", '"um" tokens', ORANGE, [2, 4, 8, 16]), ("cont", "continuous thoughts", BLUE, [2, 4, 8, 16]), ("mid", "thoughts through middle layers only", AQUA, [8, 16])]:
        pts = [(pd.tle("none"), pd.em("none")[0], "K=0")] if pd.has("none") else []
        pts += [(pd.tle(f"{key}{k}"), pd.em(f"{key}{k}")[0], f"K={k}") for k in ks if pd.has(f"{key}{k}")]
        if len(pts) > 1:
            fams.append({"name": lab, "c": c, "pts": pts})
    pts = []
    for p, short, side, dy in [("ut2", "looped x2", "r", 0), ("ut3", "looped x3", "r", 0), ("noneL6", "6 layers", "r", 0), ("cot", "written scratchpad", "l", 0)]:
        if pd.has(p):
            pts.append({"name": NAMES[p], "short": short, "c": GRAY, "x": pd.tle(p), "y": pd.em(p)[0], "side": side, "dy": dy})
    allx = [q[0] for f in fams for q in f["pts"]] + [q["x"] for q in pts]
    ally = [q[1] for f in fams for q in f["pts"]] + [q["y"] for q in pts]
    lo, hi = np.floor(min(ally) * 20) / 20, np.ceil(max(ally) * 20) / 20
    xt = [t for t in range(50, 400, 25) if min(allx) - 5 <= t <= max(allx) + 5]
    body = xy_chart(fams, pts, xlab="token-layer evaluations per problem (inference cost)", ylab="exact match", ymin=lo, ymax=hi,
                    yticks=[round(t, 2) for t in np.arange(lo, hi + 1e-9, 0.05)], xticks=xt, xmin=min(allx) - 6, xmax=max(allx) + 6, right=150)
    return figure("Accuracy against inference compute",
                  "x: transformer-layer applications needed to answer one problem with a KV cache (prompt + thinking + answer characters), the FLOPs proxy. "
                  "Lines start at the no-thinking model (K=0) and add thinking steps. Gray squares are the reference points.", body)


SUB_COLS = ["none", "um8", "cont8", "win1", "ut2", "noneL6", "cot", "coco_cont8"]
SUB_SHORT = {"none": "none", "um8": "um x8", "cont8": "cont x8", "cont16": "cont x16", "mid16": "mid x16", "win1": "1 slot", "noneL6": "6 layers",
             "cot": "scratchpad", "coco_cont8": "distilled", "ut2": "looped x2", "coco_none": "removed", "direct8": "direct x8", "disc8": "disc x8", "soft8": "soft x8"}


def fig_subskills(cols=None):
    cols = [c for c in (cols or SUB_COLS) if pd.has(c) and all(n in pd.OK for n in pd.seeds(c))]
    vals = [pd.sub(c) for c in cols]
    rows, groups, notes, data = [], [], [], []
    for i, k in enumerate(pd.SUBKEYS):
        if pd.SUB_N[k] < 30:
            continue
        rows.append(k[1])
        groups.append(mg.LEVELS[k[0]] if not rows[:-1] or prev != k[0] else None)
        prev = k[0]
        notes.append(pd.SUB_N[k])
        data.append([v[i] for v in vals])
    return figure("Exact match by sub-skill (%)",
                  "Rows are problem types inside each level (n = held-out problems of that type; types with fewer than 30 are omitted). "
                  "Darker is better. Columns with several seeds show the seed average.",
                  heat_table(rows, [SUB_SHORT[c] for c in cols], data, groups, first_col="sub-skill", notes=notes))


def fig_bars(title, cap, order, label_w=300, metric="em"):
    return figure(title, cap, _bars(order, metric, label_w)) + variant_table(order)


def fig_ksweep():
    """Accuracy when the number of thoughts at inference differs from training."""
    Ks = [0, 1, 2, 4, 8, 16, 24, 32]
    series = []
    for p, lab, c in [("krand_cont", "continuous thoughts, trained with K in {0,1,2,4,8,16}", BLUE), ("krand_um", '"um" tokens, trained with K in {0,1,2,4,8,16}', ORANGE)]:
        if pd.has(p):
            byk = pd.R[pd.seeds(p)[0]]["by_K"]
            series.append({"name": lab, "short": "varied K" if "cont" in p else "um, varied K", "c": c, "y": [byk[str(k)]["mean_em"] if str(k) in byk else None for k in Ks]})
    ks = pd.PROBES.get("ksweep", {})
    if "cont8_s0" in ks:
        series.append({"name": "continuous thoughts, trained only at K=8", "short": "fixed K=8", "c": AQUA,
                       "y": [ks["cont8_s0"][str(k)]["mean"] if str(k) in ks["cont8_s0"] else None for k in Ks]})
    if not series:
        return ""
    return figure("One model, different amounts of thinking at inference",
                  "x: thinking steps used at test time. One model saw a random K in {0, 1, 2, 4, 8, 16} at every training step; the other only ever saw K=8."
                  if len(series) == 2 else "x: thinking steps used at test time. The varied-K models saw a random K each training step; the fixed one only ever saw K=8.",
                  line_chart(series, Ks, [str(k) for k in Ks], xlab="thinking steps at inference", h=280, ymin=0.0, ymax=0.75, yticks=(0, .25, .5, .75), right=150))


def knockout_table(name="cont8_s0"):
    k = pd.PROBES.get("knockout", {}).get(name)
    if not k:
        return ""
    lab = [("full", "nothing (as trained)"), ("window4", "answer and thoughts see only the newest 4 thoughts"), ("window2", "only the newest 2"),
           ("window1", "only the newest 1"), ("shuffle", "thoughts swapped with another problem's"), ("noise", "thoughts replaced by noise of the same scale"),
           ("zero", "thoughts replaced by zeros")]
    rows = [[b, pct(k[a]["mean"])] + [pct(x, 0) for x in k[a]["levels"]] for a, b in lab if a in k]
    return table(["what is changed at inference", "exact"] + [f"L{i+1}" for i in range(7)], rows)


def fig_lens():
    lens = pd.PROBES.get("lens")
    if not lens:
        return ""
    K = len(lens["match_char"][0])
    series = [{"name": f"answer character {j+1}", "short": f"char {j+1}", "c": c, "y": lens["match_char"][j]} for j, c in zip(range(3), [BLUE, ORANGE, AQUA])]
    chart = figure("What each thought would say if it were decoded as a character",
                   "Each thought's hidden state is pushed through the ordinary output head (it was never trained to mean anything there). "
                   "y: share of test problems where the top character equals the answer's 1st, 2nd or 3rd character.",
                   line_chart(series, list(range(1, K + 1)), [f"thought {i}" for i in range(1, K + 1)], h=260, right=100))
    rows = [[mg.LEVELS[e["level"]][:2], f"<code>{esc(e['expr'])}</code>", f"<code>{esc(e['answer'])}</code>",
             "<code>" + " ".join(esc(t) for t in e["thoughts"]) + "</code>"] for e in lens["examples"]]
    return chart + table(["", "problem", "answer", "thoughts 1-8, decoded"], rows, num_from=9)


def halting_table(name="krand_cont_s0"):
    h = pd.PROBES.get("halting", {}).get(name)
    if not h:
        return ""
    rows = [[f"always {f['K']} thoughts", f"{f['K']:.1f}", pct(f["acc"])] for f in h["fixed"]]
    best_fixed = max(f["acc"] for f in h["fixed"])
    for c in h["curve"]:
        if c["tau"] in (0.5, 0.9, 0.99, 0.999):
            rows.append([f"stop once the answer's probability reaches {c['tau']}", f"{c['mean_K']:.1f}", pct(c["acc"])])
    rows.append(["oracle: fewest thoughts that give a correct answer", f"{h['oracle_mean_K']:.1f}", pct(h["oracle_acc"])])
    return table(["policy", "mean thoughts", "exact"], rows)


def halting_levels(name="krand_cont_s0"):
    h = pd.PROBES.get("halting", {}).get(name)
    if not h:
        return ""
    c = [c for c in h["curve"] if c["tau"] == 0.99][0]
    rows = [[mg.LEVELS[l]] + [pct(x, 0) for x in h["acc_by_level_K"][l]] + [f"{c['K_by_level'][l]:.1f}"] for l in range(mg.NLEVELS)]
    return table(["level"] + [f"K={k}" for k in (0, 1, 2, 4, 8, 16)] + ["thoughts used at 0.99"], rows)


def calculator_table(name, n_per_level=4, seed=3):
    """A calculator session from the saved predictions of a run (every 50th test problem was kept)."""
    preds = pd.R[name]["preds"]
    step = len(pd.EXPRS) // 280
    idx = list(range(0, len(pd.EXPRS), step))[: len(preds)]
    rng = random.Random(seed)
    rows = []
    for l in range(mg.NLEVELS):
        cand = [(i, p) for i, p in zip(idx, preds) if pd.TEST_LV[i] == l]
        for i, p in rng.sample(cand, n_per_level):
            ok = p == pd.ANSWERS[i]
            rows.append([mg.LEVELS[l][:2], f"<code>{esc(pd.EXPRS[i])}</code>", f"<code>{esc(p)}</code>", f"<code>{esc(pd.ANSWERS[i])}</code>",
                         "<span>✓</span>" if ok else "<span>✗</span>"])
    return table(["", "typed in", "model prints", "calculator prints", ""], rows, num_from=9)


def fig_distill():
    """Level-average exact match over training for the scratchpad model and the distilled one."""
    if "cot_s0" not in pd.R or not pd.has("coco_cont8"):
        return ""
    name = pd.seeds("coco_cont8")[0]
    steps = pd.curve("cot_s0")[0]
    series = [{"name": "written scratchpad kept", "short": "kept", "c": GRAY, "y": [float(np.mean(e)) for e in pd.curve("cot_s0")[1]]},
              {"name": "scratchpad removed in 4 stages, 2 continuous thoughts added per stage", "short": "distilled", "c": BLUE,
               "y": [float(np.mean(e)) for e in pd.curve(name)[1]]}]
    vl = [(11, "-1"), (13, "-2"), (15, "-3"), (17, "all")]
    return figure("Distilling the scratchpad into thoughts, stage by stage",
                  "Level-average exact match on the full target (scratchpad items still present, then the answer) during training. "
                  "The distilled run compresses the level curriculum into its first 12,000 steps, then removes one more scratchpad item every "
                  "2,000 steps (rules marked -1, -2, -3, all) while adding two continuous thoughts.",
                  line_chart(series, steps, [f"step {s//1000}k" for s in steps], h=280, xtick_idx=[4, 9, 14, 19], vlines=vl, right=110))
