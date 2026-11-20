"""make6.py [--site DIR]: the page on the second half of the experiment (three more hours for each side): the six-hour
chart, the gap over time, the second half's steps, and the checks. Reads what make.py reads, for both halves."""
import glob, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make as M  # the logs, the tokens and the checks of every recorded row, for both halves
import appendix as A1
import appendix6 as A

C, H, OUT, esc, load = M.C, M.H, M.OUT, M.esc, M.load
NAME, COLOR = M.NAME, M.COLOR
points = M.points
EAGER = M.EAGER
STRUCK = {("tc", 17)}  # measured with a stale first-convolution weight
BUGGY = {("tc", i) for i in (7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18)}  # states that still had the cache bug
for side in points:
    for p in points[side]:
        key = (side, p["i"])
        p["struck"] = key in STRUCK
        p["buggy"] = key in BUGGY
        if p["i"] in A.SHORT[side]:
            p["short"] = A.SHORT[side][p["i"]]
        p["passes"] = bool(p["follows"] and p["infer"] and abs(p["infer"]["compiled_bfloat16"]["mAP@0.5"] - p["infer"]["eager_bfloat16"]["mAP@0.5"]) < 0.01
                           and not p["buggy"] and not p["struck"])
        same = {("tc", 15): 13}.get(key, p["i"])  # the same state, measured twice
        extra = load(f"{OUT}/probe2_{side}_{same:02d}.json")
        p["slope"] = None if not extra else ("error" if "raised" in list(extra["probes"].values())[0] else list(extra["probes"].values())[0]["followed"])
HALF = {side: next(p["minutes"] for p in points[side] if p["half"] == 2) for side in points}
SPLIT = {side: max(p["minutes"] for p in points[side] if p["half"] == 1) for side in points}
COST_SPLIT = {side: max(p["cost"] for p in points[side] if p["half"] == 1) for side in points}


def best_at(side, key, x, passing=False):
    best = 0.0
    for p in points[side]:
        if p[key] <= x and p["eligible"] and not p["struck"] and (p["passes"] or not passing):
            best = max(best, p["qps"])
    return best


def final_best(side, passing=False, half=None):
    return max(p["qps"] for p in points[side] if p["eligible"] and not p["struck"] and (p["passes"] or not passing) and (half is None or p["half"] <= half))


def chart(xkey, xlab, xmax, xticks, xfmt, w=760, h=400, ymax=5000, labels=()):
    L, R, T, B = 46, 262, 30, 38
    pw, ph = w - L - R, h - T - B
    X = lambda v: L + pw * v / xmax
    Y = lambda v: T + ph * (1 - v / ymax)
    o, rows = [], []
    for t in range(0, ymax + 1, 1000):
        o.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text x="{L-6}" y="{Y(t)+4:.1f}" text-anchor="end">{t:,}</text>')
    for t in xticks:
        o.append(f'<text x="{X(t):.1f}" y="{T+ph+15}" text-anchor="middle">{xfmt(t)}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{L+pw}" y1="{T+ph}" y2="{T+ph}"/>')
    o.append(f'<text x="{L+pw/2:.1f}" y="{h-4}" text-anchor="middle" class="lab">{esc(xlab)}</text>')
    o.append(f'<text x="{L-40}" y="{T-16}" class="lab">images/s</text>')
    if xkey == "minutes":
        x = X(182)
        o.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T+ph}" stroke-dasharray="2 3"/>')
        o.append(f'<text x="{x+5:.1f}" y="{T+ph-8}">second half</text>')
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(EAGER):.1f}" y2="{Y(EAGER):.1f}" stroke="var(--base)" stroke-width="2" stroke-dasharray="5 4"/>')
    ends = [((L + pw, Y(EAGER)), "eager", f"{EAGER:,.0f}", "--base")]

    def line(side, passing):
        pts = [p for p in points[side] if p["eligible"] and not p["struck"] and (p["passes"] or not passing)]
        path, level = [], None
        for p in pts:
            new = max(level or 0, p["qps"])
            if level is None:
                path.append(f"M{X(p[xkey]):.1f},{Y(new):.1f}")
            elif new != level:
                path.append(f"L{X(p[xkey]):.1f},{Y(level):.1f} L{X(p[xkey]):.1f},{Y(new):.1f}")
            level = new
        last_x = X(max(p[xkey] for p in points[side]))
        path.append(f"L{last_x:.1f},{Y(level):.1f}")
        dash = ' stroke-dasharray="6 4"' if passing else ""
        o.append(f'<path class="ln" d="{" ".join(path)}" stroke="var({COLOR[side]})"{dash}/>')
        return (last_x, Y(level)), level

    for side in ("tc", "vm"):
        c = COLOR[side]
        end, level = line(side, False)
        ends.append((end, NAME[side], f"{level:,.0f}", c))
        if any(not p["passes"] for p in points[side] if p["eligible"]):
            end, level = line(side, True)
            ends.append((end, f"{NAME[side]}, passing every check", f"{level:,.0f}", c))
        for p in points[side]:
            cx, cy = X(p[xkey]), Y(p["qps"])
            state = "struck: measured with a stale weight" if p["struck"] else ("passes every check" if p["passes"] else "does not pass every check")
            tip = {"t": f"{NAME[side]}: {p['short']}", "r": [
                {"c": c, "n": "images/s", "v": f"{p['qps']:,.0f}"}, {"n": "agent time", "v": f"{p['minutes']:.0f} min"},
                {"n": "agent tokens, list price", "v": f"${p['cost']:.2f}"}, {"n": "checks", "v": state}]}
            o.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="10" tabindex="0" data-tip="{esc(json.dumps(tip))}"/>')
            if p["struck"]:
                o.append(f'<path d="M{cx-4:.1f},{cy-4:.1f} L{cx+4:.1f},{cy+4:.1f} M{cx-4:.1f},{cy+4:.1f} L{cx+4:.1f},{cy-4:.1f}" stroke="var({c})" stroke-width="2" fill="none" pointer-events="none"/>')
            elif p["passes"]:
                o.append(f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="var({c})" pointer-events="none"/>')
            else:
                o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="var(--surface)" stroke="var({c})" stroke-width="2" pointer-events="none"/>')
            rows.append([NAME[side], p["short"], f"{p['minutes']:.0f}", f"{p['cost']:.2f}", f"{p['qps']:,.0f}", state])
    ys = C._spread([e[0][1] for e in ends], gap=15)
    for (pt, name, value, c), y in zip(ends, ys):
        lx = L + pw + 8
        if abs(pt[0] - (L + pw)) > 2 or abs(y - pt[1]) > 3:
            o.append(f'<line class="axis" x1="{pt[0]+6:.1f}" y1="{pt[1]:.1f}" x2="{lx-2:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="lab" x="{lx:.1f}" y="{y+4:.1f}">{esc(name)} <tspan class="val">{value}</tspan></text>')
    for side, i, text, dx, dy, anchor, leader in labels:
        p = points[side][i]
        if leader:
            o.append(f'<line class="axis" x1="{X(p[xkey]):.1f}" y1="{Y(p["qps"])+(7 if dy > 0 else -7):.1f}" x2="{X(p[xkey]):.1f}" y2="{Y(p["qps"])+dy+(-12 if dy > 0 else 5):.1f}"/>')
        o.append(f'<text class="lab" x="{X(p[xkey])+dx:.1f}" y="{Y(p["qps"])+dy:.1f}" text-anchor="{anchor}">{esc(text)}</text>')
    svg = f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(o) + "</svg>"
    return svg + C.table_view(["side", "step", "agent minutes", "agent tokens, $", "images/s", "checks"], rows)


LEGEND = ('<div class="legend">'
          '<span><i style="border-top-color:var(--s1)"></i>torch.vm, best so far</span><span><i style="border-top-color:var(--s2)"></i>torch.compile, best so far</span>'
          '<span><i class="dash" style="border-top-color:var(--s2)"></i>torch.compile, best so far that passes every check</span>'
          '<span><i class="dash" style="border-top-color:var(--base)"></i>eager</span>'
          '<span><svg width="12" height="12" style="width:12px;display:inline"><circle cx="6" cy="6" r="4" fill="var(--ink2)"/></svg>measurement of a state that passes every check</span>'
          '<span><svg width="12" height="12" style="width:12px;display:inline"><circle cx="6" cy="6" r="3.5" fill="none" stroke="var(--ink2)" stroke-width="2"/></svg>of one that does not</span>'
          '<span><svg width="12" height="12" style="width:12px;display:inline"><path d="M2,2 L10,10 M2,10 L10,2" stroke="var(--ink2)" stroke-width="2"/></svg>struck</span>'
          '</div>')


# The test suites on each side's final source of the second half (acc/out/tests2_*.log).
TESTS = [
    ["torch.vm: test/test_vm.py on the CPU", "408", "pass"], ["torch.vm: test/test_vm.py on the GPU", "425", "pass"],
    ["torch.vm: the agent's own tests of its changes", "22", "pass"],
    ["torch.compile: test/dynamo/test_modules.py", "211", "pass"],
    ["torch.compile: test/dynamo/test_misc.py", "1,013", "1 fails, the same way on the starting tree (its subprocess imports a checkout without libraries: the experiment's tree setup)"],
    ["torch.compile: test/dynamo/test_guard_manager.py and test_guard_serialization.py", "399", "pass"],
    ["torch.compile: test/inductor/test_torchinductor.py on the GPU, tests of convolutions, max pools and cat", "470", "pass"],
]


def gap_chart():
    xs = list(range(0, 366, 5))
    def ratio(passing):
        out = []
        for x in xs:
            a, b = best_at("vm", "minutes", x, passing), best_at("tc", "minutes", x, passing)
            out.append(a / b if a and b else None)
        return out
    series = [{"name": "best recorded", "c": "--ink2", "y": ratio(False)},
              {"name": "best recorded that passes every check", "short": "passing every check", "c": "--s7", "y": ratio(True)}]
    return C.line_chart(series, xs, xlabels=[f"{x} min" for x in xs], ymax=2.0, ymin=0.0, yticks=[0, 0.5, 1.0, 1.5, 2.0], xtick_idx=[i for i, x in enumerate(xs) if x % 60 == 0],
                        xlab="minutes of agent time", right=190, yfmt=lambda v: f"{v:.1f}x", vfmt=lambda v: f"{v:.2f}x", vlines=[(182 / 5, "second half")], tv_header="agent time")


def steps_table(side):
    rows = []
    for p in points[side]:
        if p["half"] != 2:
            continue
        t = p["tokens"]
        acc = f"{p['infer']['compiled_bfloat16']['mAP@0.5']:.3f}" if p["infer"] else ""
        name = f'<a href="#{A.ANCHOR[side][p["i"]]}">{esc(p["short"])}</a>'
        checks = "struck" if p["struck"] else (M.probed(p["probe"]) + ("; fails the cache check" if p["buggy"] else ""))
        rows.append([p["utc"][11:16], f"{p['minutes']:.0f}", f"{p['cost']:.2f}", f"{p['qps']:,.0f}", name, acc, checks])
    return C.table(["UTC", "min", "$", "images/s", "step", "mAP@0.5, real weights", "probes, real weights"], rows, num_from=9)


def probe_table():
    cols = [("tc", 15, "torch.compile, fastest (3,887)"), ("tc", 19, "torch.compile, every guard (3,404)"), ("tc", 24, "torch.compile, final (3,634)"),
            ("vm", 0, "torch.vm, start (784)"), ("vm", 8, "torch.vm, one graph (3,261)"), ("vm", 24, "torch.vm, final (4,606)")]
    got = [(name, points[side][i]) for side, i, name in cols if points[side][i]["probe"]]
    if not got:
        return ""
    names = [k for k in got[0][1]["probe"]["probes"] if k != "the timed shape again"]
    word = lambda v: "raises an error" if "raised" in v else ("followed" if v.get("followed", v.get("right")) else "not followed") if "followed" in v else ("right" if v.get("right") else "wrong")
    rows = [[k] + [word(p["probe"]["probes"][k]) for _, p in got] for k in names]
    slope = {None: "", True: "followed", False: "not followed", "error": "raises an error"}
    rows.append(["a LeakyReLU's slope edited (added for this page)"] + [slope[p["slope"]] for _, p in got])
    return C.table(["after the callable is warm"] + [n for n, _ in got], rows, num_from=9)


def after_table():
    cols = [("tc_15", "torch.compile, end of first half"), ("tc_24", "torch.compile, final"), ("vm_17", "torch.vm, end of first half"), ("vm_24", "torch.vm, final")]
    data = [(name, load(f"{OUT}/after_{key}.json")) for key, name in cols]
    data = [(n, d) for n, d in data if d]
    if not data:
        return ""
    keys = ("at first", "after a batch of 3 images", "after images of another size", "after a weight was updated in place")
    return C.table(["throughput at the timed shape, images/s"] + [n for n, _ in data], [[k] + [f"{d[k]:,.0f}" for _, d in data] for k in keys])


def cache_table():
    rows = []
    for key, name in (("tc_first_half_final", "torch.compile, end of first half (3,887)"), ("tc_17", "torch.compile, the struck row (3,413)"), ("tc_18", "torch.compile, 3,126 row"),
                      ("tc_19", "torch.compile, first row on the corrected source (3,404)"), ("tc_second_half_final", "torch.compile, final (3,634)"), ("vm_second_half_final", "torch.vm, final (4,606)")):
        r = load(f"{OUT}/xcache_{key}_pretrained.json")
        if not r:
            continue
        d = r["compiled_against_float32"]
        rows.append([name, f"{r['compiled_bfloat16']['mAP@0.5']:.3f}", f"{r['eager_bfloat16']['mAP@0.5']:.3f}", f"{d['raw_output']['rmse']:.3f}",
                     f"{d['detections_iou_0.5']['found']} of {d['detections_iou_0.5']['detections']}"])
    return C.table(["state, compiled once for the random-weight model and then for the pretrained one", "mAP@0.5", "eager's", "error against float32", "float32's detections found"], rows) if rows else ""


def train_table():
    rows = []
    for prefix, label in (("det_", "deterministic kernels, float32"), ("", "default kernels, float32"), ("amp_", "default kernels, autocast to bfloat16")):
        ref = load(f"{OUT}/train_{prefix}eager_a.json")
        if not ref:
            continue
        for key, name in (("eager_a", "eager"), ("eager_b", "eager, run again"), ("eager_nudge_1e-6", "eager, one layer's weights x (1 + 1e-6)"), ("tc_final", "torch.compile, end of first half"),
                          ("tc_final2", "torch.compile, final"), ("vm_final", "torch.vm, end of first half"), ("vm_final2", "torch.vm, final")):
            r = load(f"{OUT}/train_{prefix}{key}.json")
            if not r:
                continue
            n = min(r["steps"], ref["steps"])
            d = [abs(a - b) / abs(b) for a, b in zip(r["loss"][:n], ref["loss"][:n])]
            comp = r.get("compiled") or {}
            graphs = comp.get("unique_graphs", comp.get("variants", ""))
            rows.append([label, name, f"{r['loss'][n-1]:.3f}", "" if key == "eager_a" else f"{100*statistics.mean(d):.2f}%", f"{r['ms_per_step_last_50']:.0f}", graphs])
    return C.table(["settings", "forward and backward through", "loss, step 300", "mean distance from eager's loss", "ms per step", "graphs"], rows, num_from=2) if rows else ""


def loss_figure():
    runs = {n: load(f"{OUT}/train_det_{n}.json") for n in ("eager_a", "eager_nudge_1e-6", "tc_final2", "vm_final2")}
    if not all(runs.values()):
        return ""
    per = 8
    passes = min(r["steps"] for r in runs.values()) // per
    mean = lambda r: [sum(r["loss"][i * per:(i + 1) * per]) / per for i in range(passes)]
    series = [{"name": "eager", "c": "--ink2", "y": mean(runs["eager_a"])},
              {"name": "eager, one layer's weights changed in the 6th digit", "short": "eager, nudged", "c": "--base", "y": mean(runs["eager_nudge_1e-6"]), "dash": True},
              {"name": "torch.compile, final source", "short": "torch.compile", "c": "--s2", "y": mean(runs["tc_final2"])},
              {"name": "torch.vm, final source", "short": "torch.vm", "c": "--s1", "y": mean(runs["vm_final2"])}]
    top = (int(max(max(s["y"]) for s in series) / 5) + 1) * 5
    body = C.line_chart(series, list(range(1, passes + 1)), xlabels=[str(i) for i in range(1, passes + 1)], ymax=top, ymin=0, yticks=list(range(0, top + 1, 5)),
                        xtick_idx=[i for i in range(passes) if (i + 1) % 5 == 0 or i == 0], xlab="pass over the 128 images (8 steps each)", right=150,
                        yfmt=lambda v: f"{v:.0f}", vfmt=lambda v: f"{v:.3f}", tv_header="pass")
    return C.figure("Training with the final sources: loss per pass, deterministic kernels, float32",
                    "As on the first page: yolov3-spp from a seeded random start, the model's own detection loss and SGD settings, 300 steps of 16 images.", body)


if __name__ == "__main__":
    b3 = {s: final_best(s, half=1) for s in points}
    b6 = {s: final_best(s) for s in points}
    p3 = {s: final_best(s, True, half=1) for s in points}
    p6 = {s: final_best(s, True) for s in points}
    total = {s: M.dollars(M.tokens[s]["total"]) for s in points}
    print({"best 3h": b3, "best 6h": b6, "passing 3h": p3, "passing 6h": p6, "cost": total, "cost at 3h": COST_SPLIT})
    numbers = dict(b3=b3, b6=b6, p3=p3, p6=p6, total=total)
    import text6
    body = text6.body(sys.modules[__name__], numbers)
    title = "Three more hours: does the gap between torch.compile and torch.vm widen?"
    description = ("The same two AI agents get three more hours each on TorchBench yolov3 inference, one improving torch.compile and one torch.vm. The best recorded throughput barely moves "
                   "(3,973 and 4,614 images/s after six hours); what moves is how much of torch.compile's speed survives the checks, after its agent made freezing notice weight updates and found "
                   "a bug in one of its own earlier changes. With predictions, probes and training curves checked against eager. Nov 20, 2026.")
    footer = ('<footer>Written with an AI assistant (Claude Code), which also set up the experiment, ran the checks and made the charts; the two agents were instances of the same model. '
              'One model, one run per side, one GPU: an example, not a benchmark. The harness, the rules, both agents\' journals and final changes, the results and the generator of this page: '
              '<a href="https://github.com/bobrenjc93/explanations/tree/main/two-compilers-six-hours">two-compilers-six-hours</a>.</footer>')
    head = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{esc(title)}</title>\n'
            f'<meta name="description" content="{esc(description)}">\n<link rel="icon" href="../favicon.svg" type="image/svg+xml">\n<link rel="icon" href="../favicon.ico" sizes="any">\n'
            f'<link rel="apple-touch-icon" href="../apple-touch-icon.png">\n<style>{C.CSS}{C.EXTRA_CSS}{A1.CSS}</style>\n</head>\n<body>\n')
    page = f"{head}{body}{footer}\n<script>{C.JS}</script>\n</body>\n</html>\n"
    open(f"{H}/report/six.html", "w").write(page)
    print("wrote", f"{H}/report/six.html", len(page))
    if "--site" in sys.argv:
        dest = sys.argv[sys.argv.index("--site") + 1]
        os.makedirs(dest, exist_ok=True)
        open(os.path.join(dest, "index.html"), "w").write(page)
        print("wrote", os.path.join(dest, "index.html"))
