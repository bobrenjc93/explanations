"""Figures for the scaling write-up: error against parameters, training length, compute and unique data."""
import json, math, os
import numpy as np
import mathgen3 as m3
import figs4 as G
from charts import *

GRAY, BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, VIOLET = "--base", "--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"
OLD, NL = m3.OLD, m3.NLEVELS
POOL_ROWS = 28_000_000
# every run of the study: name -> directory. The three H_ runs are from the previous write-ups.
WHERE = {"H_full_3m": "runs/hard", "H_full_3m_30k": "runs/hard", "H_full_11m": "runs/hard"}
SIZE = ["S_n02", "S_n08", "S_n18", "H_full_3m", "H_full_11m", "S_n19"]           # 10k steps, full pool, parameters vary
STEPS = {"3.2M": ["S_t2k5", "S_t5k", "H_full_3m", "H_full_3m_30k"], "0.8M": ["S_n08", "S_n08_30k"]}  # training length varies
DATA = ["S_d056k", "S_d560k", "S_d5m6", "H_full_3m"]                             # 3.2M, 10k steps, unique rows vary
ALL = list(dict.fromkeys(SIZE + sum(STEPS.values(), []) + DATA))


def _dir(n):
    return WHERE.get(n, "runs/scale")


def have(n):
    return os.path.exists(f"{_dir(n)}/{n}.json")


_M = {}


def m(n):
    """Everything measured for one run."""
    if n in _M:
        return _M[n]
    R = json.load(open(f"{_dir(n)}/{n}.json"))
    em = R["final"]["em"]
    loss = R.get("loss") or G.jload(f"{_dir(n)}/{n}.loss.json") or {}
    s = G.jload(f"{_dir(n)}/{n}.stretch100.json")
    if s:
        stretch = [s["em"][m3.LEVELS[l]] for l in range(OLD, NL)]
    else:  # an older run scored on all 300 per tier: take the same first 100
        s = G.jload(f"{_dir(n)}/{n}.stretch.json")
        lv = np.array(G.STRETCH["level"])
        stretch = [float(np.mean(np.array(s["ok"])[lv == l][:100])) for l in range(OLD, NL)] if s else None
    c = R["compute"]
    steps, d = R["args"]["steps"], R["model"]["d"]
    seen = steps * R["args"]["bs"]
    frac = R["args"].get("data_frac", 1.0)
    unique = c.get("unique_rows", POOL_ROWS) if frac < 1 else POOL_ROWS * (1 - math.exp(-seen / POOL_ROWS))
    lvl_loss = loss.get("test_level")
    _M[n] = {"name": n, "N": R["model"]["params"], "d": d, "L": R["model"]["L"], "steps": steps, "seen": seen, "unique": unique, "frac": frac,
             "pflop": steps * c["padded_positions_per_step"] * c["layer_applications"] * 3 * G.fpt(d) / 1e15,
             "gflop": R["final"]["tle"] * G.fpt(d) / 1e9, "em": em, "acc": float(np.mean(em)), "old": float(np.mean(em[:OLD])), "new": float(np.mean(em[OLD:])),
             "loss": loss.get("test"), "loss_level": lvl_loss, "train_loss": loss.get("train"), "train_tf": loss.get("train_tf_exact"),
             "test_tf": loss.get("test_tf_exact"), "stretch": stretch, "stretch_mean": float(np.mean(stretch)) if stretch else None,
             "hours": R["train_time"] / 3600}
    return _M[n]


def runs(names):
    return [m(n) for n in names if have(n)]


CEIL = None


def ceiling():
    """Mean over the 14 levels of the share of test problems whose reference solution ends in the true answer."""
    global CEIL
    if CEIL is None:
        CEIL = float(np.mean(G.ceiling()))
    return CEIL


def nfmt(n):
    return f"{n/1e6:.1f}M" if n >= 1e5 else f"{n/1e3:.0f}k"


FIT_MIN = 0.5  # runs that get fewer than half the problems right are before the power-law regime (error cannot exceed 100%)


def fit_power(xs, ys):
    """Least squares in log-log: y = a * x^(-alpha). -> (a, alpha)"""
    b, loga = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(np.exp(loga)), float(-b)


def excess(r):
    """In-range error above the floor a perfect solver has."""
    return max(1 - r["acc"] - (1 - ceiling()), 1e-4)


LOGT = lambda v: math.log10(v)


def _loglog(fams, xlab, ylab, xt, yt, xfmt, yfmt, right=110, h=340, pts=()):
    """Scatter/line chart with both axes logarithmic: points are given in data units."""
    conv = [dict(f, pts=[(LOGT(x), LOGT(y), lab) for x, y, lab in f["pts"]]) for f in fams]
    allx = [p[0] for f in conv for p in f["pts"]]
    ally = [p[1] for f in conv for p in f["pts"]]
    return xy_chart(conv, pts, xlab=xlab, ylab=ylab, xticks=[LOGT(t) for t in xt], yticks=[LOGT(t) for t in yt], xmin=min(allx + [LOGT(xt[0])]) - 0.05,
                    xmax=max(allx + [LOGT(xt[-1])]) + 0.05, ymin=min(ally + [LOGT(yt[0])]) - 0.05, ymax=max(ally + [LOGT(yt[-1])]) + 0.05,
                    xfmt=lambda v: xfmt(10 ** v), yfmt=lambda v: yfmt(10 ** v), vfmt=lambda v: yfmt(10 ** v), right=right, h=h)


PCT = lambda v: f"{100*v:.2g}%" if v < 0.1 else f"{100*v:.0f}%"
ERR_TICKS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_TICKS = [1e5, 3e5, 1e6, 3e6, 1e7, 3e7]


def run_table(names=None):
    rows = []
    for r in runs(names or ALL):
        rows.append([r["name"], nfmt(r["N"]), f"{r['steps']//1000 if r['steps'] % 1000 == 0 else r['steps']/1000:g}k", nfmt(r["unique"]), f"{r['pflop']:.0f}",
                     f"{r['hours']:.1f}", f"{r['loss']:.4f}" if r["loss"] else "", pct(r["old"]), pct(r["new"]), pct(r["stretch_mean"]) if r["stretch_mean"] is not None else ""])
    return table(["run", "parameters", "steps", "distinct problems trained on", "train PFLOP", "hours", "held-out loss / token", "L1‑7", "L8‑14",
                  "one size up"], rows)


# ------------------------------------------------------------------ parameters
def size_fit(upto=None):
    """Power law of the excess in-range error against parameters over the size sweep (optionally only the runs up to a size)."""
    rs = [r for r in runs(SIZE) if (upto is None or r["N"] <= upto) and r["acc"] >= FIT_MIN]
    return fit_power([r["N"] for r in rs], [excess(r) for r in rs]), rs


def fig_size():
    rs = runs(SIZE)
    if len(rs) < 3:
        return ""
    fams = [{"name": "L1 to L7", "c": GRAY, "pts": [(r["N"], max(1 - r["old"], 1e-4), nfmt(r["N"])) for r in rs]},
            {"name": "L8 to L14", "c": ORANGE, "pts": [(r["N"], max(1 - r["new"], 1e-4), nfmt(r["N"])) for r in rs]},
            {"name": "all 14 levels", "c": BLUE, "pts": [(r["N"], 1 - r["acc"], nfmt(r["N"])) for r in rs]}]
    floor = 1 - ceiling()
    fams.append({"name": "floor: a perfect solver", "c": GRAY, "dash": True, "nodots": True, "pts": [(rs[0]["N"], floor, ""), (rs[-1]["N"], floor, "")]})
    return figure("In-range error against parameters",
                  "Share of test problems answered wrong (log scale) against the number of parameters (log scale). Every model trained for 10,000 "
                  "steps on the same 28M-problem pool. Dashed: the error a solver would have that followed the worked solutions perfectly.",
                  _loglog(fams, "parameters", "error", [n for n in N_TICKS if rs[0]["N"] * 0.6 <= n <= rs[-1]["N"] * 1.6], ERR_TICKS, nfmt, PCT, right=150))


def fig_size_levels():
    rs = runs(SIZE)
    if len(rs) < 3:
        return ""
    labs = [nfmt(r["N"]) for r in rs]
    vals = [r["em"] for r in rs]
    return heat_table([f"{l} parameters" for l in labs], G.SHORT, vals, first_col="exact match, 10k steps")


def fig_loss_size():
    rs = [r for r in runs(SIZE) if r["loss"]]
    if len(rs) < 3:
        return ""
    clean = [l for l in range(NL) if l in (0, 1, 2, 3, 7, 9, 11, 13)]  # levels whose solutions have no jittered root estimate
    fams = [{"name": "all levels", "c": GRAY, "pts": [(r["N"], r["loss"], nfmt(r["N"])) for r in rs]},
            {"name": "levels without root estimates", "c": BLUE, "pts": [(r["N"], float(np.mean([r["loss_level"][l] for l in clean])), nfmt(r["N"])) for r in rs]}]
    return figure("Held-out loss against parameters",
                  "Cross-entropy per token on the test set's reference solutions (log-log). Square and cube roots start from an estimate that was "
                  "jittered in training, so a few tokens per problem cannot be predicted and put a floor under the gray line; the blue line leaves "
                  "those levels out.",
                  _loglog(fams, "parameters", "nats per token", [n for n in N_TICKS if rs[0]["N"] * 0.6 <= n <= rs[-1]["N"] * 1.6],
                          [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3], nfmt, lambda v: f"{v:g}", right=190))


# ------------------------------------------------------------------ training length and compute
FAM_COL = {"0.2M": GRAY, "0.8M": ORANGE, "1.8M": YELLOW, "3.2M": BLUE, "10.7M": VIOLET, "19.0M": AQUA}


def fig_compute():
    """Error against training compute, one line per model size (points: training lengths)."""
    by = {}
    for r in runs(ALL):
        if r["frac"] >= 1:
            by.setdefault(nfmt(r["N"]), []).append(r)
    fams = []
    for k in sorted(by, key=lambda k: by[k][0]["N"]):
        pts = sorted(by[k], key=lambda r: r["pflop"])
        fams.append({"name": f"{k} parameters", "c": FAM_COL.get(k, GRAY), "pts": [(r["pflop"], 1 - r["acc"], f"{r['steps']/1000:g}k steps") for r in pts]})
    if len(fams) < 2:
        return ""
    floor = 1 - ceiling()
    xs = [p[0] for f in fams for p in f["pts"]]
    fams.append({"name": "floor", "c": GRAY, "dash": True, "nodots": True, "pts": [(min(xs), floor, ""), (max(xs), floor, "")]})
    return figure("In-range error against training compute",
                  "Error over the 14 levels (log scale) against forward-plus-backward FLOPs of the run (log scale; PFLOP = 10^15). One line per "
                  "model size; the points along a line are training lengths.",
                  _loglog(fams, "training compute, PFLOP", "error", [3, 10, 30, 100, 300, 1000, 3000], ERR_TICKS, lambda v: f"{v:g}", PCT, right=150))


def steps_fit(key="3.2M"):
    rs = runs(STEPS[key])
    ok = [r for r in rs if r["acc"] >= FIT_MIN]
    return (fit_power([r["steps"] for r in ok], [excess(r) for r in ok]), rs) if len(ok) >= 3 else (None, rs)


# ------------------------------------------------------------------ data
def fig_data():
    rs = sorted(runs(DATA), key=lambda r: r["unique"])
    if len(rs) < 3:
        return ""
    fams = [{"name": "held-out problems", "c": BLUE, "pts": [(r["unique"], max(1 - r["acc"], 1e-4), nfmt(r["unique"])) for r in rs]}]
    tr = [(r["unique"], max(1 - r["train_tf"], 1e-4), nfmt(r["unique"])) for r in rs if r["train_tf"] is not None]
    te = [(r["unique"], max(1 - r["test_tf"], 1e-4), nfmt(r["unique"])) for r in rs if r["test_tf"] is not None]
    if len(tr) >= 2:
        fams.append({"name": "the rows it trained on, reproduced token for token", "c": GRAY, "pts": tr})
        fams.append({"name": "held-out problems, reproduced token for token", "c": ORANGE, "pts": te})
    return figure("Error against the number of distinct training problems",
                  "3.2M parameters, 10,000 steps of batch 2,048 (20.5M problems drawn) in every run; only the size of the pool they are drawn from "
                  "changes. Blue: wrong final answers on the test set. Gray and orange: worked solutions not reproduced exactly under teacher forcing, "
                  "on training rows and on test problems.",
                  _loglog(fams, "distinct problems in the pool", "error", [3e4, 1e5, 1e6, 1e7, 3e7], ERR_TICKS, nfmt, PCT, right=40, h=320))


def data_table():
    rs = sorted(runs(DATA), key=lambda r: r["unique"])
    rows = [[nfmt(r["unique"]), f"{r['seen'] / r['unique']:.1f}", pct(r["acc"]), pct(r["new"]), f"{r['loss']:.4f}" if r["loss"] else "",
             f"{r['train_loss']:.4f}" if r["train_loss"] is not None else "", pct(r["stretch_mean"]) if r["stretch_mean"] is not None else ""] for r in rs]
    return table(["distinct problems", "times each is seen", "exact match, 14 levels", "L8‑14", "held-out loss", "loss on its training rows", "one size up"], rows)


# ------------------------------------------------------------------ one size up
def fig_stretch_scale():
    rs = [r for r in runs(ALL) if r["frac"] >= 1 and r["stretch_mean"] is not None]
    if len(rs) < 3:
        return ""
    by = {}
    for r in rs:
        by.setdefault(nfmt(r["N"]), []).append(r)
    fams = []
    for k in sorted(by, key=lambda k: by[k][0]["N"]):
        pts = sorted(by[k], key=lambda r: r["pflop"])
        fams.append({"name": f"{k} parameters", "c": FAM_COL.get(k, GRAY), "pts": [(LOGT(r["pflop"]), r["stretch_mean"], f"{r['steps']/1000:g}k steps") for r in pts]})
    xt = [3, 10, 30, 100, 300, 1000, 3000]
    xs = [p[0] for f in fams for p in f["pts"]]
    return figure("One size up against training compute",
                  "Exact match on the stretch problems (first 100 per tier, mean of L8 to L14) against the run's training compute (log scale). "
                  "None of these sizes was trained on.",
                  xy_chart(fams, [], xlab="training compute, PFLOP", ylab="exact match, one size up", xticks=[LOGT(t) for t in xt if LOGT(t) <= max(xs) + 0.1],
                           yticks=[0, .25, .5, .75, 1.0], xmin=min(xs) - 0.1, xmax=max(xs) + 0.1, ymin=0.0, ymax=1.0, xfmt=lambda v: f"{10**v:g}", right=150))


def stretch_levels_table():
    rs = [r for r in runs(SIZE) if r["stretch"] is not None]
    if not rs:
        return ""
    rows = [[m3.LEVELS[l]] + [pct(r["stretch"][l - OLD], 0) for r in rs] for l in range(OLD, NL)]
    rows.append(["mean"] + [pct(r["stretch_mean"]) for r in rs])
    return table(["one size up, 10k steps"] + [nfmt(r["N"]) for r in rs], rows)


# ------------------------------------------------------------------ per-token view
REF_LEN = [float(G.LEN[G.LV == l].mean()) for l in range(NL)]


def slips(r, levels=range(NL)):
    """Wrong answers per million solution tokens: sum over levels of -ln(accuracy) divided by the tokens a reference solution has.
    (If every token went wrong independently with probability p, a solution of T tokens would be right with probability (1-p)^T.)"""
    num = sum(-math.log(max(r["em"][l], 1e-3)) for l in levels)
    return 1e6 * num / sum(REF_LEN[l] for l in levels)


def slip_table():
    rs = runs(SIZE)
    if len(rs) < 3:
        return ""
    rows = []
    for r in rs:
        rows.append([nfmt(r["N"]), f"{slips(r):,.0f}", f"{slips(r, range(OLD)):,.0f}", f"{slips(r, range(OLD, NL)):,.0f}", pct(r["em"][NL - 1]),
                     f"{1e6 * -math.log(max(r['em'][NL - 1], 1e-3)) / REF_LEN[NL - 1]:,.0f}"])
    return table(["parameters", "fatal slips per million tokens, all levels", "L1‑7", "L8‑14", "L14 Newton (1,700 tokens): exact match", "its slips per million tokens"], rows)


def compute_fit():
    """Power law of the excess in-range error against training compute over every full-data run, whatever its size and length.
    -> ((a, gamma), runs, largest factor by which a run misses the line)"""
    rs = [r for r in runs(ALL) if r["frac"] >= 1 and r["acc"] >= FIT_MIN]
    a, g = fit_power([r["pflop"] for r in rs], [excess(r) for r in rs])
    miss = max(abs(math.log(excess(r) / (a * r["pflop"] ** (-g)))) for r in rs)
    return (a, g), rs, math.exp(miss)
