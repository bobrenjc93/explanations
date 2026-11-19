"""make.py: the write-up of the two-sided experiment as one HTML page (report/index.html), from the sides' logs,
the agents' transcripts (tokens) and the coordinator's validation runs. Everything that is not there yet is left out."""
import glob, json, os, statistics, subprocess, sys

sys.path.insert(0, "/home/dev/local/cfgloop/page")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appendix as A
import charts as C

H = "/home/dev/local/hill"
OUT = f"{H}/acc/out"
PRICE = {"input": 8.0, "cache_write": 10.0, "cache_read": 0.2, "output": 40.0}  # $ per million tokens, list price
NAME = {"tc": "torch.compile", "vm": "torch.vm"}
COLOR = {"tc": "--s2", "vm": "--s1"}
esc = C.esc


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def dollars(t):
    return sum(t[k] * PRICE[k] for k in PRICE) / 1e6


subprocess.run(["python3", "-I", f"{H}/tools/tokens.py", "--json", f"{OUT}/tokens.json"], check=True, stdout=subprocess.DEVNULL)
tokens = load(f"{OUT}/tokens.json")

# What each tc checkpoint's configuration switches on that stops following eager (read from its snapshot).
GIVES_UP = (("freezing = True", "freezing"), ('"guard_filter_fn":', "guard filter"), ('"triton.cudagraph_trees": False', "outputs reused"))


def gives_up(side, sha):
    repo = f"{H}/tc/tree" if side == "tc" else f"{H}/vm/snap"
    text = subprocess.run(["git", "-C", repo, "show", f"{sha}:hill_config.py"], capture_output=True, text=True).stdout
    text = text[text.find("def build"):]  # the options that are set, not the ones the docstring describes
    return [name for needle, name in GIVES_UP if needle in text]


SHORT = {
    "tc": ["start", "CUDA graphs", "freezing (weights built in), with a fix in Dynamo", "max pool one dimension at a time, in Inductor", "guard filter (unsafe)",
           "pool stages as unrolled kernels", "concatenation and in-place settings", "a fourth input channel for the first convolution", "a smaller inlining threshold",
           "(hardening)", "autotune: Inductor's own kernels for the 1x1 convolutions", "(reference: no guard filter)", "(repeat)", "CUDA graphs without the tree bookkeeping",
           "(reference: no guard filter)", "(repeat)"],
    "vm": ["start", "stores into module buffers allowed: one graph", "CUDA graphs, built as a new option", "fast path kept after the region's own stores",
           "weights check moved behind the kernel launch", "pass: max pools of 5, 9, 13 as a cascade of 5s", "(hardening)", "(hardening)", "(docstrings)",
           "pass: convolution + batch norm + LeakyReLU as one cuDNN kernel", "first layer and detection heads fused too", "fastest of cuDNN's first five kernel variants per layer",
           "pass: final concatenation as a kernel of its own", "pass: each pool as two one-dimensional pools", "input channels padded to a multiple of 4",
           "fastest of all ten kernel variants per layer", "input read in place instead of copied", "(steadier timing of the variants)"],
}
# The few steps named on the chart itself.
CHART = {("tc", 1): "CUDA graphs", ("tc", 2): "freezing", ("tc", 4): "guard filter (unsafe)", ("tc", 10): "autotune",
         ("vm", 1): "buffer stores: one graph", ("vm", 2): "CUDA graphs, as a new option", ("vm", 9): "pass: fused cuDNN conv kernel"}
points = {}
for side in ("tc", "vm"):
    rows, best = [], 0.0
    for i, p in enumerate(tokens[side]["points"]):
        probe, infer = load(f"{OUT}/probe_{side}_{i:02d}.json"), load(f"{OUT}/infer_{side}_{i:02d}.json")
        lost = gives_up(side, p["sha"])
        follows = None
        if probe:
            pr = probe["probes"]
            follows = all(v.get("followed", v.get("right")) is True for k, v in pr.items() if k != "model.train()" or True)
        new_best = p["eligible"] and p["qps"] > best
        best = max(best, p["qps"]) if p["eligible"] else best
        rows.append(dict(p, i=i, cost=dollars(p["tokens"]), lost=lost, probe=probe, infer=infer, follows=follows, new_best=new_best, best=best,
                         short=SHORT[side][i] if i < len(SHORT[side]) else p["note"][:40]))
    points[side] = rows
total = {side: dollars(tokens[side]["total"]) for side in points}
sound = [load(p) for p in sorted(glob.glob(f"{OUT}/bench_tc_finalsrc_cudagraphs_*.json"))]
sound_qps = statistics.median(r["qps"] for r in sound) if len(sound) == 3 else None  # final source, CUDA graphs only: passes every probe
nofreeze = [load(p) for p in sorted(glob.glob(f"{OUT}/bench_tc_sound_a_*.json"))]
nofreeze_qps = statistics.median(r["qps"] for r in nofreeze) if len(nofreeze) == 3 else None  # final source and settings, no freezing or guard filter
EAGER = statistics.median(r["eager_qps"] for side in ("tc", "vm") for r in [json.loads(l) for l in open(f"{H}/{side}/log.jsonl")] if r.get("eager_qps"))


def step_chart(xkey, xlab, xmax, xticks, xfmt, w=760, h=380, ymax=5000, labels=True):
    L, R, T, B = 46, 250, 30, 38
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
    # eager, for reference
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(EAGER):.1f}" y2="{Y(EAGER):.1f}" stroke="var(--base)" stroke-width="2" stroke-dasharray="5 4"/>')
    ends = [((L + pw, Y(EAGER)), "eager", f"{EAGER:,.0f}", "--base")]
    for side in ("tc", "vm"):
        c, pts = COLOR[side], points[side]
        end = max(xmax * 0.0, max(p[xkey] for p in pts))
        path, level = [], None
        for p in pts:
            if not p["eligible"]:
                continue
            if level is None:
                path.append(f"M{X(p[xkey]):.1f},{Y(p['best']):.1f}")
            elif p["best"] != level:
                path.append(f"L{X(p[xkey]):.1f},{Y(level):.1f} L{X(p[xkey]):.1f},{Y(p['best']):.1f}")
            level = p["best"]
        last_x = X(xmax_of[side][xkey])
        path.append(f"L{last_x:.1f},{Y(level):.1f}")
        o.append(f'<path class="ln" d="{" ".join(path)}" stroke="var({c})"/>')
        for p in pts:
            cx, cy = X(p[xkey]), Y(p["qps"])
            hollow = bool(p["lost"])
            tip = {"t": f"{NAME[side]}: {p['short']}", "r": [
                {"c": c, "n": "images/s", "v": f"{p['qps']:,.0f}"}, {"n": "agent time", "v": f"{p['minutes']:.0f} min"},
                {"n": "agent tokens, list price", "v": f"${p['cost']:.2f}"},
                {"n": "gives up", "v": ", ".join(p["lost"]) if hollow else "nothing"}]}
            o.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="11" tabindex="0" data-tip="{esc(json.dumps(tip))}"/>')
            if hollow:
                o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="var(--surface)" stroke="var({c})" stroke-width="2" pointer-events="none"/>')
            else:
                o.append(f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="var({c})" pointer-events="none"/>')
            rows.append([NAME[side], p["short"], f"{p['minutes']:.0f}", f"{p['cost']:.2f}", f"{p['qps']:,.0f}", ", ".join(p["lost"]) if hollow else "nothing"])
        ends.append(((last_x, Y(level)), NAME[side], f"{level:,.0f}", c))
    if sound_qps:
        x = X(xmax_of["tc"][xkey])
        tip = {"t": "torch.compile, final source with CUDA graphs: the fastest state that passes every probe (measured afterwards)", "r": [{"c": "--s2", "n": "images/s", "v": f"{sound_qps:,.0f}"}]}
        o.append(f'<circle class="hit" cx="{x:.1f}" cy="{Y(sound_qps):.1f}" r="11" tabindex="0" data-tip="{esc(json.dumps(tip))}"/>')
        o.append(f'<rect class="dot" x="{x-4.5:.1f}" y="{Y(sound_qps)-4.5:.1f}" width="9" height="9" rx="2" fill="var(--s2)" pointer-events="none"/>')
        ends.append(((x, Y(sound_qps)), "torch.compile, nothing given up", f"{sound_qps:,.0f}", "--s2"))
        rows.append(["torch.compile", "final source with CUDA graphs only (measured afterwards)", "", "", f"{sound_qps:,.0f}", "nothing"])
    ys = C._spread([e[0][1] for e in ends], gap=15)
    for (pt, name, value, c), y in zip(ends, ys):
        lx = L + pw + 8
        if abs(pt[0] - (L + pw)) > 2 or abs(y - pt[1]) > 3:
            o.append(f'<line class="axis" x1="{pt[0]+6:.1f}" y1="{pt[1]:.1f}" x2="{lx-2:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="lab" x="{lx:.1f}" y="{y+4:.1f}">{esc(name)} <tspan class="val">{value}</tspan></text>')
    if labels:
        for side, i, dx, dy, anchor, leader in LABELS:
            p = points[side][i]
            if leader:
                o.append(f'<line class="axis" x1="{X(p[xkey]):.1f}" y1="{Y(p["qps"])-7:.1f}" x2="{X(p[xkey]):.1f}" y2="{Y(p["qps"])+dy+5:.1f}"/>')
            o.append(f'<text class="lab" x="{X(p[xkey])+dx:.1f}" y="{Y(p["qps"])+dy:.1f}" text-anchor="{anchor}">{esc(CHART[(side, i)])}</text>')
    svg = f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(o) + "</svg>"
    return svg + C.table_view(["side", "step", "agent minutes", "agent tokens, $", "images/s", "options that give up what eager does"], rows)


xmax_of = {side: {"cost": total[side], "minutes": tokens[side]["minutes"]} for side in points}
# (side, index of the step, dx, dy, anchor): the few steps named on the chart; every step is in the tooltip and the table.
LABELS = [("vm", 1, 8, 14, "start", False), ("vm", 2, 8, 14, "start", False), ("vm", 9, 8, 16, "start", False), ("tc", 1, -14, -44, "start", True),
          ("tc", 2, 8, 17, "start", False), ("tc", 4, 8, -40, "end", True), ("tc", 10, -8, -10, "end", False)]
LEGEND = ('<div class="legend">'
          '<span><i style="border-top-color:var(--s1)"></i>torch.vm</span><span><i style="border-top-color:var(--s2)"></i>torch.compile</span>'
          '<span><i class="dash" style="border-top-color:var(--base)"></i>eager</span>'
          '<span><svg width="12" height="12" style="width:12px;display:inline"><circle cx="6" cy="6" r="4" fill="var(--ink2)"/></svg>measurement with no option that gives up what eager does</span>'
          '<span><svg width="12" height="12" style="width:12px;display:inline"><circle cx="6" cy="6" r="3.5" fill="none" stroke="var(--ink2)" stroke-width="2"/></svg>measurement with such an option (freezing, guard filter)</span>'
          '</div>')


PROBE = {"bias of the last convolution += 1": "last bias", "bias of a batch norm in the middle += 0.5": "batch-norm bias",
         "weight of a convolution in the middle *= 0.5": "convolution weight", "running mean of a batch norm in the middle += 0.5": "running mean",
         "a batch of 3 images": "batch size", "images of another size (320x416)": "image size", "the timed shape again": "timed shape again",
         "last bias replaced by a new Parameter": "replaced parameter", "model.train()": "model.train()"}
FOLDED = {"batch-norm bias", "convolution weight", "running mean"}


def probed(probe):
    """What the stricter probes found for one checkpoint, in a few words."""
    if not probe:
        return ""
    lost, raised = [], []
    for k, v in probe["probes"].items():
        if "raised" in v:
            raised.append(PROBE[k])
        elif not v.get("followed", v.get("right")):
            lost.append(PROBE[k])
    if FOLDED <= set(lost):
        lost = ["weight updates"] + [x for x in lost if x not in FOLDED]
    parts = (["not followed: " + ", ".join(lost)] if lost else []) + (["error on: " + ", ".join(raised)] if raised else [])
    return "; ".join(parts) or "all followed"


def steps_table(side):
    rows = []
    every = all(p["probe"] for sd in points for p in points[sd])
    for p in points[side]:
        t = p["tokens"]
        acc = f"{p['infer']['compiled_bfloat16']['mAP@0.5']:.3f}" if p["infer"] else ""
        rows.append([p["utc"][11:16], f"{p['minutes']:.0f}", f"{p['cost']:.2f}", f"{t['output']/1e3:.0f}k", f"{p['qps']:,.0f}", f'<a href="#{A.ANCHOR[side][p["i"]]}">{esc(p["short"])}</a>',
                     ", ".join(p["lost"]) if p["lost"] else "nothing", acc] + ([probed(p["probe"])] if every else []))
    return C.table(["UTC", "min", "$", "output tokens", "images/s", "step", "options that give up what eager does", "mAP@0.5, real weights"]
                   + (["probes, real weights"] if every else []), rows, num_from=9)


def loss_figure(prefix, title, cap):
    runs = {n: load(f"{OUT}/train_{prefix}{n}.json") for n in ("eager_a", "eager_b", "tc_start", "tc_final", "vm_start", "vm_final")}
    if not all(runs.values()):
        return "", None
    nudged = load(f"{OUT}/train_{prefix}eager_nudge_1e-6.json")
    n = min(r["steps"] for r in runs.values())
    per = 8
    passes = n // per
    mean = lambda r: [sum(r["loss"][i * per:(i + 1) * per]) / per for i in range(passes)]
    second = ("eager, one layer's weights changed in the 6th digit", nudged) if nudged else ("eager, run again", runs["eager_b"])
    series = [{"name": "eager", "c": "--ink2", "y": mean(runs["eager_a"])}, {"name": second[0], "short": "eager, nudged" if nudged else "eager again", "c": "--base", "y": mean(second[1]), "dash": True},
              {"name": "torch.compile, final source", "short": "torch.compile", "c": "--s2", "y": mean(runs["tc_final"])},
              {"name": "torch.vm, final source", "short": "torch.vm", "c": "--s1", "y": mean(runs["vm_final"])}]
    hi = max(max(s["y"]) for s in series)
    top = (int(hi / 5) + 1) * 5
    body = C.line_chart(series, list(range(1, passes + 1)), xlabels=[str(i) for i in range(1, passes + 1)], ymax=top, ymin=0, yticks=list(range(0, top + 1, 5)),
                        xtick_idx=[i for i in range(passes) if (i + 1) % 5 == 0 or i == 0], xlab="pass over the 128 images (8 steps each)", right=150,
                        yfmt=lambda v: f"{v:.0f}", vfmt=lambda v: f"{v:.3f}", tv_header="pass")
    ref = runs["eager_a"]["loss"]
    rel = lambda r: [abs(a - b) / abs(b) for a, b in zip(r["loss"][:n], ref[:n])]
    first = lambda r: next((i for i, (a, b) in enumerate(zip(r["loss"], ref)) if a != b), None)
    rows = []
    if nudged:
        runs["nudged"] = nudged
    for key, name in (("eager_b", "eager, run again"), ("nudged", "eager, one layer's weights x (1 + 1e-6)"), ("tc_start", "torch.compile, start source"),
                      ("tc_final", "torch.compile, final source"), ("vm_start", "torch.vm, start source"), ("vm_final", "torch.vm, final source")):
        if key not in runs:
            continue
        d = rel(runs[key])
        f = first(runs[key])
        rows.append([name, f"{runs[key]['loss'][0]:.5f}", f"{runs[key]['loss'][n-1]:.3f}", f"{100*statistics.mean(d):.2f}%", f"{100*max(d):.1f}%",
                     "never" if f is None else str(f + 1), f"{runs[key]['ms_per_step_last_50']:.0f}"])
    rows.insert(0, ["eager", f"{ref[0]:.5f}", f"{ref[n-1]:.3f}", "", "", "", f"{runs['eager_a']['ms_per_step_last_50']:.0f}"])
    table = C.table(["forward and backward through", "loss, step 1", f"loss, step {n}", "mean distance from eager's loss", "largest", "first step that differs", "ms per step"], rows)
    return C.figure(title, cap, body) + table, runs


def acc_table():
    rows = []
    for side, i, name in (("tc", 0, "torch.compile, start"), ("tc", len(points["tc"]) - 1, "torch.compile, final"), ("vm", 0, "torch.vm, start"), ("vm", len(points["vm"]) - 1, "torch.vm, final")):
        r = points[side][i]["infer"]
        if not r:
            continue
        if not rows:
            e32, e16 = r["eager_float32"], r["eager_bfloat16"]
            d = r["eager_bfloat16_against_float32"]
            rows.append(["eager, float32", "", f"{e32['mAP@0.5']:.3f}", "0", "all"])
            rows.append(["eager, bfloat16", f"{EAGER:,.0f}", f"{e16['mAP@0.5']:.3f}", f"{d['raw_output']['rmse']:.3f}", f"{d['detections_iou_0.5']['found']} of {d['detections_iou_0.5']['detections']}"])
        d = r["compiled_against_float32"]
        rows.append([name, f"{points[side][i]['qps']:,.0f}", f"{r['compiled_bfloat16']['mAP@0.5']:.3f}", f"{d['raw_output']['rmse']:.3f}", f"{d['detections_iou_0.5']['found']} of {d['detections_iou_0.5']['detections']}"])
    for key, name, qps in (("tc_sound_a", "torch.compile, final source and settings, no freezing or guard filter", nofreeze_qps),
                           ("tc_clean", "torch.compile, final source, CUDA graphs only", sound_qps)):
        r = load(f"{OUT}/infer_{key}.json")
        if r and qps:
            d = r["compiled_against_float32"]
            rows.append([name, f"{qps:,.0f}", f"{r['compiled_bfloat16']['mAP@0.5']:.3f}", f"{d['raw_output']['rmse']:.3f}", f"{d['detections_iou_0.5']['found']} of {d['detections_iou_0.5']['detections']}"])
    return C.table(["", "images/s", "mAP@0.5 on 128 COCO images", "error against float32 (RMSE of the raw output)", "float32's detections found"], rows)


def probe_table():
    cols = [("tc", len(points["tc"]) - 1, "torch.compile, final"), ("vm", len(points["vm"]) - 1, "torch.vm, final")]
    got = [(name, points[side][i]["probe"]) for side, i, name in cols]
    for at, key, name in ((1, "tc_sound_a", "torch.compile, final source and settings, no freezing or guard filter"), (2, "x_finalsrc_startsettings", "torch.compile, final source, CUDA graphs only")):
        extra = load(f"{OUT}/probe_{key}.json")
        if extra:
            got.insert(at, (name, extra))
    got = [(n, p) for n, p in got if p]
    if not got:
        return ""
    names = [k for k in got[0][1]["probes"] if k != "the timed shape again"]
    word = lambda v: "raises an error" if "raised" in v else ("followed" if v.get("followed", v.get("right")) else "not followed") if "followed" in v else ("right" if v.get("right") else "wrong")
    rows = [[k] + [word(p["probes"][k]) for _, p in got] for k in names]
    return C.table(["after the callable is warm"] + [n for n, _ in got], rows, num_from=9)


if __name__ == "__main__":
    final = {side: points[side][-1] for side in points}
    best = {side: max(p["qps"] for p in points[side] if p["eligible"]) for side in points}
    tiles = C.tiles([
        ("torch.compile, 3 hours of one agent", f"{points['tc'][0]['qps']:,.0f} → {best['tc']:,.0f}", f"images/s; ${total['tc']:.0f} of tokens at list price. The best state no longer follows weight updates or <code>model.train()</code>."),
        ("torch.vm, 3 hours of one agent", f"{points['vm'][0]['qps']:,.0f} → {best['vm']:,.0f}", f"images/s; ${total['vm']:.0f} of tokens. Every recorded state follows eager on every probe; from 105 minutes on, through a fused cuDNN kernel the agent wrote as a pass."),
        ("torch.compile with nothing given up", f"{sound_qps:,.0f}" if sound_qps else "…", f"images/s, measured afterwards: the agent's final source with CUDA graphs, the fastest state that passes every probe. With its other settings {nofreeze_qps:,.0f}, but then a recompilation raises an error."),
        ("Predictions, real weights and images", "0.752 / 0.752", "mAP@0.5 of the two final states on 128 COCO images; eager is 0.753 in the same precision. All 34 checkpoints: 0.751 to 0.754."),
    ])
    by_cost = C.figure("Throughput against what the agent had spent", "TorchBench yolov3 inference, batches of 8 images in bfloat16 on one H100. Every dot is an official measurement (median of three processes); the line is the best so far. x: the agent's tokens at list price.",
                       LEGEND + step_chart("cost", "agent tokens spent, $ at list price", 42, [0, 10, 20, 30, 40], lambda t: f"${t}"))
    by_time = C.figure("The same against the agent's time", "x: minutes since the agent started. Each side stopped once on its own before the deadline and was told to go on (torch.vm at 86 minutes, torch.compile at 152).",
                       LEGEND + step_chart("minutes", "minutes of agent time", 185, [0, 30, 60, 90, 120, 150, 180], lambda t: f"{t}", labels=False))
    loss32, _ = loss_figure("det_", "Training: loss per pass, deterministic kernels, float32",
                            "yolov3-spp from a seeded random start, the model's own detection loss and SGD settings, 300 steps of 16 images. Forward and backward through eager, torch.compile or torch.vm (Inductor graphs).")
    loss_default, _ = loss_figure("", "Training: the same with PyTorch's default kernels (TF32, non-deterministic)", "Here two eager runs already differ from each other; the compiled runs are inside that spread.")
    after = {side: load(f"{OUT}/after_{side}_{len(points[side]) - 1:02d}.json") for side in points}
    after_rows = []
    if all(after.values()):
        for key in ("at first", "after a batch of 3 images", "after images of another size", "after a weight was updated in place"):
            after_rows.append([key] + [f"{after[side][key]:,.0f}" for side in ("tc", "vm")])
    after_table = ""
    if after_rows:
        after_table = ("<p>And the speed afterwards, in the benchmark's setup: the callable is timed, then again after each event.</p>"
                       + C.table(["throughput at the timed shape, images/s", "torch.compile, final", "torch.vm, final"], after_rows)
                       + "<p>The first of two runs of this for <code>torch.compile</code> died when the new image size arrived: a cuDNN internal error while the new CUDA graph was being recorded. The second run is the one in the table.</p>")
    t = {side: tokens[side]["total"] for side in points}
    body = f"""<h1>One model, two compilers, the same agent for three hours each</h1>
<p class="sub">Where torch.compile and torch.vm start on one model, and how fast each climbs when the same AI agent is given three hours to improve it: throughput against the time and tokens spent, with the predictions and training curves checked against eager. Nov 19, 2026.</p>
<p><a href="../cfg-compile/"><code>torch.vm</code></a> is a prototype compiler for PyTorch: it runs a function's bytecode as a control-flow graph in a small VM and hands the tensor work to the same backend
<code>torch.compile</code> uses (AOTAutograd and Inductor). On the performance dashboard's suites <code>torch.vm</code> is a little ahead on average, and a few models are far behind. This page takes the one where <code>torch.vm</code> was furthest behind,
TorchBench's yolov3 in inference (1.00x of eager against 2.50x), and asks a different question: not where the two compilers are today, but what an hour of work buys in each.
Two instances of the same AI coding agent (Claude) got the same instructions and three hours, one for each compiler: its options and its source were theirs to change, the model and the measurement were not.
Every gain was recorded by a harness neither could touch. The chart is those records against what the agent had spent by then.</p>
<div class="tldr"><p><b>In short.</b> <code>torch.compile</code> started higher (2,013 images/s against 784; eager does {EAGER:,.0f}) and <code>torch.vm</code> climbed faster:
after three hours of one agent each, 3,973 against 4,507. By time, <code>torch.vm</code> is ahead from 105 minutes on. By tokens, <code>torch.compile</code> is ahead at every budget
up to the $30 its agent spent; the <code>torch.vm</code> agent spent $41 in the same three hours and passed it at $32. Two things qualify the end points.
The top of the <code>torch.compile</code> line is reached with freezing and an unsafe guard filter, after which it no longer follows a weight update or <code>model.train()</code>;
the fastest <code>torch.compile</code> state measured that passes every probe does {sound_qps:,.0f}. And the last 1,250 images/s of <code>torch.vm</code> come from a fused cuDNN kernel its agent wrote as a pass,
which is a kernel either compiler could call; without it the line ends at 3,261. Every recorded state of both sides predicts as well as eager on real weights and images.</p></div>
{tiles}
{by_cost}
{by_time}
<h2>What the chart shows</h2>
<ul>
<li><b>Start.</b> <code>torch.compile</code> gets 2.5x of eager out of the box. <code>torch.vm</code> got nothing: the model assigns two buffers on every call, a region refused a store into a module's buffer,
and so the model ran kernel by kernel. That one fix, 12 minutes in, is the largest step on the chart (784 to 1,983).</li>
<li><b>The first hour: the same ceiling from two sides.</b> Both agents found the same thing in their first profile: 2.5 ms of GPU kernels inside 3.9 ms of wall time. Both turned to CUDA graphs
(an option for <code>torch.compile</code>; the <code>torch.vm</code> agent built it, as a new option of <code>Graphs</code>). Both ended with the weights built into the graph: freezing for <code>torch.compile</code>,
after a fix in Dynamo that this model needed; <code>torch.vm</code> does it by default, and its agent moved the checks that guard it behind the kernel launch. Both rewrote the max pools. At 70 minutes they were at 3,270 and 3,261,
and <code>torch.vm</code>'s agent measured the GPU busy for 97% of a call. They hand the same graph to the same backend, so they met at its kernels.</li>
<li><b>After that it is about kernels.</b> <code>torch.compile</code>'s agent changed Inductor's lowering (separable max pool, a zero channel for the first convolution) and turned on autotuning and the older, cheaper CUDA-graph path: 3,973.
<code>torch.vm</code>'s agent stopped at 86 minutes, saying what was left was below the noise; told to go on, it replaced convolution + batch norm + LeakyReLU with one cuDNN kernel through cuDNN's
backend API, timed cuDNN's engine choices per layer, and reached 4,507. <code>torch.compile</code>'s agent named the same fusion as its next step and did not build it.</li>
<li><b>Per token.</b> The two agents read about the same amount (71 and 75 million cached input tokens); the <code>torch.vm</code> one wrote more
({t['vm']['output']/1e3:.0f}k output tokens against {t['tc']['output']/1e3:.0f}k, most of the difference being the 600-line pass). At equal dollars <code>torch.compile</code> stays ahead through its whole budget.</li>
<li><b>Hollow dots.</b> From 21 minutes on, every <code>torch.compile</code> measurement uses freezing, and from 59 minutes the guard filter
<code>skip_guard_on_all_nn_modules_unsafe</code>. Its agent said so in every note. The only <code>torch.compile</code> measurements that still follow a weight update are the first two (2,013 and 2,641)
and the ones taken afterwards with its final source (the square: {sound_qps:,.0f}). Every <code>torch.vm</code> measurement kept the weights watched; the final state is checked below.</li>
</ul>
<h2>Every step</h2>
<p>Time and dollars are the agent's own at the moment of the measurement; the dollars are its tokens at Anthropic's list price for the model ($8 per million input, $40 output, $10 cache write, $0.20 cache read; what a cloud provider charges may differ).
About 57 of each side's 183 minutes were spent waiting for the other side's GPU jobs.</p>
<p>Each step's name links to what it is, in the <a href="#appendix">appendix</a>.</p>
<h3>torch.compile: {points['tc'][0]['qps']:,.0f} to {best['tc']:,.0f}</h3>
<p>Three source files changed (166 lines added: Dynamo's attribute fallback for modules, Inductor's pooling and convolution lowering) and a 55-line configuration.</p>
{steps_table('tc')}
<h3>torch.vm: {points['vm'][0]['qps']:,.0f} to {best['vm']:,.0f}</h3>
<p>Three source files changed (485 lines added, nearly all in the regions) and a 628-line configuration that holds four graph passes and the fused kernel. No Rust was changed.</p>
{steps_table('vm')}
<h2>Are the predictions right?</h2>
<p>The benchmark runs the model with random weights, and with those the output hardly depends on the convolutions: halving a convolution's weight moves eager's output by exactly zero in bfloat16.
So the harness's check against eager says little. This one loads the original pretrained yolov3-spp weights into the same model and runs the 128 COCO images that ship with the benchmark,
letterboxed to the benchmark's 384x512 and in its batches of 8.</p>
{acc_table()}
<p>The compiled results are closer to float32 than eager's bfloat16 result is, as they should be where batch norms are folded in float32 before the cast.
All 34 recorded checkpoints were run through this: their mAP is between 0.751 and 0.754 (eager in bfloat16, in the same processes: 0.753 to 0.754), and it is in the tables above next to each step.</p>
<h2>What the final states still follow</h2>
<p>With the pretrained weights, after the callable is warm, a change is made to the callable's model and to an eager copy, and the two outputs are compared.
The table has the final states; the same probes were run on all 34 recorded checkpoints, and the last column of the tables above has the outcome for each.
Every <code>torch.vm</code> checkpoint follows all of them. Of <code>torch.compile</code>'s, the first two do.</p>
{probe_table()}
<p>Without freezing and the guard filter, <code>torch.compile</code> with its agent's final source and final settings ({nofreeze_qps:,.0f} images/s) follows weights and shapes, but where a recompilation is due
(a replaced parameter, <code>model.train()</code>) it raises an error from Inductor's CUDA-graph bookkeeping ("storage data ptrs are not allocated in pool"). The final source alone, or the settings alone on the
starting source, do not; it takes both. The fastest state measured that passes everything is the final source with CUDA graphs and no other setting: {sound_qps:,.0f}. The cause was not looked into.</p>
{after_table}
<h2>Tests</h2>
<p>On each side's final source:</p>
{C.table(["", "tests", "result"], [
    ["torch.vm: test/test_vm.py on the CPU", "408", "pass"], ["torch.vm: test/test_vm.py on the GPU", "425", "pass"],
    ["torch.vm: the agent's own tests of its changes", "16", "pass"],
    ["torch.compile: test/dynamo/test_modules.py", "211", "pass"],
    ["torch.compile: test/dynamo/test_misc.py", "1,013", "1 fails, the same way on the starting tree (its subprocess imports a checkout without libraries: the experiment's tree setup)"],
    ["torch.compile: test/inductor/test_torchinductor.py on the GPU, tests of convolutions, max pools and cat", "470", "pass"]], num_from=1)}
<h2>Training</h2>
<p>The experiment was inference. This checks that the compilers, as their agents left them, still train the model like eager does: yolov3-spp from a seeded random start, the model's own detection loss and
SGD settings, the 128 images in a fixed order, 300 steps of 16. The forward, and so the backward, goes through eager, <code>torch.compile</code> or <code>torch.vm</code> with Inductor graphs.</p>
{loss32}
<p>With deterministic kernels two eager runs are equal bit for bit, so the first row of differences is exact. <code>torch.compile</code>'s first loss equals eager's to seven digits; <code>torch.vm</code>'s first seven steps are
eager's bit for bit (it runs eager's kernels until its graphs are compiled). After that the curves separate, by as much as eager's own curve separates when one layer's weights are changed
in the sixth digit: this network, at a random start with batch norm in training mode, amplifies a last-digit difference to a percent within ten steps. The compiled curves are as far from eager as that nudged eager run is: 1.0% to 1.8% on average, against 1.9%.</p>
{loss_default}
<p><code>torch.vm</code> is not faster in training on this model: the fix that made inference one graph does not apply in training mode, where it still makes a graph per layer and runs at eager's speed.</p>
<h2>What this does and does not show</h2>
<ul>
<li><b>One model, one run per side.</b> It is an example of the two slopes, not a measurement of them. The model was picked because it is where <code>torch.vm</code> was furthest behind, which is where a steep climb is most likely.</li>
<li><b>Most of the climb is catching up.</b> Of <code>torch.vm</code>'s 3,723 images/s, 2,477 were reaching the ceiling <code>torch.compile</code> was already near. The rest is the fused kernel.</li>
<li><b>The agent knew one compiler and not the other.</b> It has read <code>torch.compile</code>'s source in training and had never seen <code>torch.vm</code>; it was given the documentation and the tree. The one-line diagnosis of this model
in <code>torch.vm</code>'s report was taken out of what it was given, and turned out to be wrong anyway (the report blamed the layer loop; the agent found the buffer stores).</li>
<li><b>What is not on the x axis:</b> the tokens that built <code>torch.vm</code> up to today, and those of the session that set the experiment up and checked it (the harness, the checks on this page).</li>
<li><b>The rules allowed what each side did.</b> Options that give up eager's behaviour were allowed if said; so was a kernel called from a pass. A stricter reading strikes both, and leaves {sound_qps:,.0f} against 3,261: 17% apart, which is weights built into the graph while they are still watched.</li>
</ul>
<h2>How it was run</h2>
<ul>
<li>Model and setup as the PyTorch performance dashboard has them: eval mode, batches of 8 images of 3x384x512, bfloat16, <code>no_grad</code>, one H100. Each side starts from the dashboard's configuration.</li>
<li>Each agent may change its compiler's options and Python source (and Rust, for <code>torch.vm</code>), in its own copy of the tree; not the model, not the harness, not eager, no C++ build. Neither may use the other's compiler.</li>
<li>A measurement: the state is snapshotted, three processes each warm the callable up (at least 300 calls and 20 s) and time 400 calls on four different inputs; the median counts if the outputs match eager's
by the dashboard's rule for bfloat16 and the probes pass (a weight updated in place, another batch size, the timed shape again).</li>
<li>Tokens are read from each agent's transcript, by the timestamp of each response.</li>
<li>Both agents stopped once before the deadline and were sent the same message to go on (<code>torch.vm</code> at 86 minutes, <code>torch.compile</code> at 152).</li>
<li>Found while the harness was built: the dashboard's runner compiles a function that calls the model, and that is what the <code>torch.compile</code> side starts from. <code>torch.compile(model)</code> on the module itself runs this model
at 7.6 ms per batch instead of 4.0, with the same graphs. The cause was not looked into.</li>
</ul>
{A.APPENDIX}
"""
    title = "One model, two compilers, the same agent for three hours each"
    description = ("TorchBench yolov3 inference under torch.compile and torch.vm: two instances of the same AI agent get three hours each to improve one compiler, and throughput is plotted "
                   "against the time and tokens spent. torch.compile starts higher (2,013 images/s against 784) and torch.vm climbs faster (3,973 against 4,507 at the end); what the end points "
                   "rest on, predictions checked with pretrained weights on COCO images, and training loss curves against eager. Nov 19, 2026.")
    footer = ('<footer>Written with an AI assistant (Claude Code), which also set up the experiment, ran the checks and made the charts; the two agents were instances of the same model. '
              'One model, one run per side, one GPU: an example, not a benchmark. The harness, the rules both agents were given, their journals and final changes, the results and the generator of this page: '
              '<a href="https://github.com/bobrenjc93/explanations/tree/main/two-compilers-one-model">two-compilers-one-model</a>.</footer>')
    head = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{esc(title)}</title>\n'
            f'<meta name="description" content="{esc(description)}">\n<link rel="icon" href="../favicon.svg" type="image/svg+xml">\n<link rel="icon" href="../favicon.ico" sizes="any">\n'
            f'<link rel="apple-touch-icon" href="../apple-touch-icon.png">\n<style>{C.CSS}{C.EXTRA_CSS}{A.CSS}</style>\n</head>\n<body>\n')
    page = f"{head}{body}{footer}\n<script>{C.JS}</script>\n</body>\n</html>\n"
    os.makedirs(f"{H}/report", exist_ok=True)
    open(f"{H}/report/index.html", "w").write(page)
    print("wrote", f"{H}/report/index.html", len(page))
    if "--site" in sys.argv:
        dest = sys.argv[sys.argv.index("--site") + 1]
        os.makedirs(dest, exist_ok=True)
        open(os.path.join(dest, "index.html"), "w").write(page)
        print("wrote", os.path.join(dest, "index.html"))
    for side in points:
        print(side, "start", round(points[side][0]["qps"]), "best", round(best[side]), "cost", round(total[side], 2), "minutes", round(tokens[side]["minutes"]))
    print("eager", round(EAGER), "sound tc", sound_qps and round(sound_qps))
