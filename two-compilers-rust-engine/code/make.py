"""make.py [--tokens] [--site DIR]: the write-up of the second run of the two-sided experiment as one HTML page
(report/index.html), from the sides' logs and snapshots, the token table and the validation runs made after the
deadline. Reads only files under /home/dev/local/hill2. `--tokens` runs tools/tokens.py again first (it reads the
agents' transcripts, which are not under that directory); without it the saved report/tokens.json is used.
`--site DIR` also writes the page to DIR/index.html. Checks that have not been run for a state are reported as such."""
import json, os, re, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import appendix as A
import charts as C

H = "/home/dev/local/hill2"
OUT = f"{H}/acc/out"
PRICE = {"input": 8.0, "cache_write": 10.0, "cache_read": 0.2, "output": 40.0}  # $ per million tokens, list price
NAME = {"tc": "torch.compile", "vm": "torch.vm"}
COLOR = {"tc": "--s2", "vm": "--s1"}
esc = C.esc


def load(path):
    return json.load(open(path)) if os.path.exists(path) and os.path.getsize(path) else None


def dollars(t):
    return sum(t[k] * PRICE[k] for k in PRICE) / 1e6


def git(side, *command):
    return subprocess.run(["git", "-C", f"{H}/{side}/tree", *command], capture_output=True, text=True).stdout


if "--tokens" in sys.argv or not os.path.exists(f"{HERE}/tokens.json"):
    subprocess.run(["python3", "-I", f"{H}/tools/tokens.py", "--json", f"{HERE}/tokens.json"], check=True, stdout=subprocess.DEVNULL)
tokens = load(f"{HERE}/tokens.json")

# The probes, in the order of the tables: (name in the check's output, short name, which script, announced to the agents).
PROBES = [
    ("bias of the last convolution += 1", "last bias", "probe", True),
    ("bias of a batch norm in the middle += 0.5", "batch-norm bias", "probe", True),
    ("weight of a convolution in the middle *= 0.5", "convolution weight", "probe", True),
    ("running mean of a batch norm in the middle += 0.5", "running mean", "probe", True),
    ("last bias replaced by a new Parameter", "replaced parameter", "probe", True),
    ("model.train()", "model.train()", "probe", True),
    ("a batch of 3 images", "batch size", "probe", True),
    ("images of another size (320x416)", "image size", "probe", True),
    ("the timed shape again", "timed shape again", "probe", True),
    ("negative slope of a LeakyReLU in the middle changed to 0.3", "LeakyReLU slope", "probe2", True),
    # The three that were not announced are probe4.py's. probe3.py was their first version: its two probes of kept
    # results held only the first tensor of the result, and its last call repeated the input of the kept call.
    ("a 3x3 convolution's weight halved through .data", "write through .data", "probe4", False),
    ("every tensor of a result kept across two calls with other inputs", "results kept as tensors", "probe4", False),
    ("every tensor of a result kept through DLPack across two calls with other inputs", "results kept through DLPack", "probe4", False),
]
LONG = {
    "bias of the last convolution += 1": "the last convolution's bias += 1, in place",
    "bias of a batch norm in the middle += 0.5": "a batch norm's bias += 0.5, in place",
    "weight of a convolution in the middle *= 0.5": "a convolution's weight *= 0.5, in place",
    "running mean of a batch norm in the middle += 0.5": "a batch norm's running mean += 0.5, in place",
    "last bias replaced by a new Parameter": "the last bias replaced by a new Parameter",
    "model.train()": "model.train(), then model.eval() again",
    "a batch of 3 images": "a batch of 3 images",
    "images of another size (320x416)": "images of another size (320x416)",
    "the timed shape again": "the timed shape again",
    "negative slope of a LeakyReLU in the middle changed to 0.3": "the slope of one LeakyReLU set to 0.3 (an attribute that is not a tensor)",
    "a 3x3 convolution's weight halved through .data": "a 3x3 convolution's weight halved through .data",
    "every tensor of a result kept across two calls with other inputs": "every tensor of a result kept across two calls with other inputs",
    "every tensor of a result kept through DLPack across two calls with other inputs": "the same kept through DLPack, by raw address",
}
FOLDED = {"batch-norm bias", "convolution weight", "running mean"}


def outcome(v):
    """ok / no / undo (followed, but wrong once it was undone) / error, for one probe's entry."""
    if v is None:
        return None
    if "raised" in v:
        return "error"
    if not v.get("followed", v.get("right")):
        return "no+" if v.get("right_after_undo") is False else "no"
    return "undo" if v.get("right_after_undo") is False else "ok"


def tests_log(path):
    """What a unittest log says: tests run and the names of those that failed or errored; None while it is not complete."""
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read()
    ran = re.findall(r"^Ran (\d+) tests? in", text, re.M)
    if not ran:
        return None
    return {"ran": int(ran[-1]), "bad": sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+)", text, re.M)))}


def kind_of(side, config):
    """Which of a side's two configurations a snapshot's config.py is, from the options it sets."""
    body = config[config.find("def build") if side == "vm" else config.find("import torch"):]
    if side == "tc":
        return "unchecked" if '"freezing": True' in body else "checked"
    return "fast" if 'weight_layout="keep"' in body else "safe"


KIND = {"checked": "checked line", "unchecked": "unchecked variants", "safe": "safe line", "fast": "fast configuration"}
KIND1 = {"checked": "checked configuration", "unchecked": "unchecked variant", "safe": "safe configuration", "fast": "fast configuration"}
points = {}
for side in ("tc", "vm"):
    rows = []
    for i, p in enumerate(tokens[side]["points"]):
        source = tuple(line.split()[2] for line in git(side, "ls-tree", p["sha"]).splitlines() if line.split("\t")[1] not in ("hill_config.py", ".gitignore"))
        files = {kind: load(f"{OUT}/{kind}_{side}_{i:02d}.json") for kind in ("infer", "probe", "probe2", "probe4")}
        raw = {name: (files[kind] or {}).get("probes", {}).get(name) for name, _, kind, _ in PROBES}
        checks = {name: outcome(raw[name]) if files[kind] else None for name, _, kind, _ in PROBES}
        rows.append(dict(p, i=i, cost=dollars(p["tokens"]), source=source, infer=files["infer"], checks=checks, raw=raw, own_tests=tests_log(f"{OUT}/ckpt_tests_{side}_{i:02d}.log"),
                         kind=kind_of(side, git(side, "show", f"{p['sha']}:hill_config.py")), short=A.SHORT[side][i], anchor=A.ANCHOR[side][i]))
    base = rows[0]["own_tests"]
    for p in rows:
        # The tests of a state: every run on a row with the same source. For some torch.compile rows only the one test
        # that tells the sources apart was run (acc/out/ckpt_tests_NOTE.txt); `tests` is the widest run of the source.
        same = [q for q in rows if q["source"] == p["source"] and q["own_tests"]]
        p["runs"] = [(q["i"], q["own_tests"]["ran"]) for q in same]
        widest = max(same, key=lambda q: q["own_tests"]["ran"], default=None)
        p["tests"], p["tests_from"] = (widest["own_tests"], widest["i"]) if widest else (None, None)
        bad = sorted({t for q in same for t in q["own_tests"]["bad"]})
        p["bad"] = bad
        p["broken"] = None if not same or (bad and base is None) else [t for t in bad if t not in base["bad"]]
        p["struck"] = bool(p["eligible"] and p["broken"])
        failed, missing = [], []
        if p["infer"] is None:
            missing.append("predictions")
        elif abs(p["infer"]["compiled_bfloat16"]["mAP@0.5"] - p["infer"]["eager_bfloat16"]["mAP@0.5"]) >= 0.01:
            failed.append("predictions")
        for name, short, kind, _ in PROBES:
            if p["checks"][name] is None:
                if kind not in missing:
                    missing.append(kind)
            elif p["checks"][name] != "ok":
                failed.append(short)
        if p["broken"] is None:
            missing.append("tests")
        p["failed"], p["missing"] = failed, missing
        p["passes"] = False if failed or p["struck"] or not p["eligible"] else (None if missing else True)
        p["counts"] = p["eligible"] and not p["struck"]
    points[side] = rows
total = {side: dollars(tokens[side]["total"]) for side in points}
minutes = {side: tokens[side]["minutes"] for side in points}
EAGER = statistics.median(r["eager_qps"] for side in ("tc", "vm") for r in [json.loads(l) for l in open(f"{H}/{side}/log.jsonl")] if r.get("eager_qps"))
INCOMPLETE = [(side, p["i"], p["missing"]) for side in points for p in points[side] if p["eligible"] and p["missing"]]
# Lines added per source file, last snapshot against the first (the rebuilt engine library is not source), and the
# length of the final configuration.
CHANGED, CONFIG_LINES = {}, {}
for side in points:
    first_sha = git(side, "rev-list", "--max-parents=0", "HEAD").split()[0]
    stat = [line.split("\t") for line in git(side, "diff", "--numstat", first_sha, points[side][-1]["sha"], "--", ".", ":!hill_config.py").splitlines()]
    CHANGED[side] = {path: int(added) for added, _, path in stat if added != "-"}
    CONFIG_LINES[side] = len(git(side, "show", f"{points[side][-1]['sha']}:hill_config.py").splitlines())
# How far the three processes of one official row are apart (median over the rows, the start row, the widest after it),
# and how many times each agent ran the announced checks on its own state during the run.
SPREAD, CHECK_RUNS = {}, {}
for side in points:
    official = [r for r in (json.loads(l) for l in open(f"{H}/{side}/log.jsonl")) if r["kind"] == "official" and r.get("qps_runs")]
    apart = [100 * (max(r["qps_runs"]) - min(r["qps_runs"])) / r["qps"] for r in official]
    SPREAD[side] = (statistics.median(apart), apart[0], max(apart[1:]))
    CHECK_RUNS[side] = sum(name.startswith("check_") and name.endswith("_probe.json") and os.path.getmtime(f"{H}/{side}/out/{name}") >= tokens[side]["start"]
                           for name in os.listdir(f"{H}/{side}/out"))


def holds(p, passing):
    """Whether a row counts for a line: every row that is neither refused nor struck for the best recorded, and for the
    other line those that pass every check (a state with a check still to run counts until the check says otherwise)."""
    return p["counts"] and (not passing or p["passes"] is not False)


def best_at(side, key, x, passing=False):
    best = 0.0
    for p in points[side]:
        if p[key] <= x and holds(p, passing):
            best = max(best, p["qps"])
    return best


def final_best(side, passing=False):
    return max(p["qps"] for p in points[side] if holds(p, passing))


def state_word(p):
    if not p["eligible"]:
        return "refused by the harness"
    if p["struck"]:
        return "struck: its source fails a test of its side"
    if p["passes"] is None:
        return "no failure found; not yet run: " + ", ".join(p["missing"])
    return "passes every check" if p["passes"] else "does not pass every check"


def probed(p):
    """What the probes found for one state, in a few words."""
    if all(v is None for v in p["checks"].values()):
        return "not run"
    lost = [short for name, short, _, _ in PROBES if p["checks"][name] in ("no", "no+", "undo")]
    raised = [short for name, short, _, _ in PROBES if p["checks"][name] == "error"]
    if FOLDED <= set(lost):
        lost = ["weight updates in place"] + [x for x in lost if x not in FOLDED]
    not_run = sorted({kind for name, _, kind, _ in PROBES if p["checks"][name] is None})
    done = sum(v is not None for v in p["checks"].values())
    parts = (["not followed: " + ", ".join(lost)] if lost else []) + (["error on: " + ", ".join(raised)] if raised else [])
    text = "; ".join(parts) or f"all {done} followed"
    return text + (f" ({', '.join(not_run)} not run)" if not_run else "")


def rows_text(rows):
    """Row numbers in words: 3 / 3 and 4 / 3, 4 and 7 / 8 to 20."""
    rows = sorted(rows)
    if len(rows) > 2 and rows == list(range(rows[0], rows[-1] + 1)):
        return f"{rows[0]} to {rows[-1]}"
    return str(rows[0]) if len(rows) == 1 else ", ".join(map(str, rows[:-1])) + f" and {rows[-1]}"


FINAL_GPU = tests_log(f"{OUT}/tests_tc_inductor_gpu.log")  # the Inductor GPU tests of convolutions, max pools and cat on torch.compile's final source


def ran_on(side, p):
    """Which of its side's tests were run on the source of a row, in words."""
    counts = sorted({n for _, n in p["runs"]}, reverse=True)
    parts = []
    for n in counts:
        on = [i for i, m in p["runs"] if m == n]
        what = "the stride test alone" if n == 1 else f"{n} tests"
        parts.append(f"{what} on row{'s' if len(on) > 1 else ''} {rows_text(on)}")
    if side == "tc" and FINAL_GPU and p["source"] == points["tc"][-1]["source"]:
        parts.append(f"the {FINAL_GPU['ran']} tests of convolutions, max pools and cat on the final source, which is this one")
    return "; ".join(parts) or "not run"


def tests_result(side, p):
    if p["tests"] is None:
        return "not run"
    final_bad = FINAL_GPU["bad"] if side == "tc" and FINAL_GPU and p["source"] == points["tc"][-1]["source"] else []
    if p["broken"] or final_bad:
        return "<b>fails " + esc(", ".join(n.replace("_cuda", "") for n in sorted(set(p["broken"] or []) | set(final_bad)))) + "</b>"
    if p["bad"] and p["source"] == points[side][0]["source"]:
        return f"{len(p['bad'])} errors: {esc(', '.join(p['bad']))}"
    if p["bad"]:
        return f"the same {len(p['bad'])} errors as on the starting source"
    return "pass"


def step_chart(xkey, xlab, xmax, xticks, xfmt, w=760, h=470, ymin=1500, ymax=3750, labels=()):
    """Throughput against what the agent had spent: a mark per official measurement and two step lines per side.
    The y axis starts at `ymin` (the paths lie between 1,800 and 3,600; eager is below it and named in a note).
    labels: [(side, row, text)], named in a band above the plot, each with a leader down to its mark."""
    L, R, B = 46, 262, 38
    T = 34 + 14 * len(labels)
    pw, ph = w - L - R, h - T - B
    X = lambda v: L + pw * v / xmax
    Y = lambda v: T + ph * (1 - (v - ymin) / (ymax - ymin))
    o, rows = [], []
    for t in range(ymin, ymax + 1, 500):
        o.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text x="{L-6}" y="{Y(t)+4:.1f}" text-anchor="end">{t:,}</text>')
    for t in xticks:
        o.append(f'<text x="{X(t):.1f}" y="{T+ph+15}" text-anchor="middle">{xfmt(t)}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{L+pw}" y1="{T+ph}" y2="{T+ph}"/>')
    o.append(f'<text x="{L+pw/2:.1f}" y="{h-4}" text-anchor="middle" class="lab">{esc(xlab)}</text>')
    o.append(f'<text x="{L-40}" y="{14}" class="lab">images/s</text>')
    o.append(f'<text x="{L+pw-4}" y="{T+ph-8}" text-anchor="end">the axis starts at {ymin:,}; eager does {EAGER:,.0f} images/s</text>')
    ends = []
    end_x = {"minutes": minutes, "cost": total}[xkey]
    # The named steps first, so that the lines and marks are drawn over their leaders.
    named = sorted(labels, key=lambda l: points[l[0]][l[1]][xkey])
    for k, (side, i, text) in enumerate(named):
        p = points[side][i]
        y = 30 + 14 * k
        o.append(f'<line class="axis" x1="{X(p[xkey]):.1f}" y1="{Y(p["qps"])-7:.1f}" x2="{X(p[xkey]):.1f}" y2="{y-3:.1f}"/>')
        o.append(f'<text class="lab" x="{X(p[xkey])+5:.1f}" y="{y:.1f}">{esc(text)}</text>')

    def line(side, passing):
        pts = [p for p in points[side] if holds(p, passing)]
        path, level = [], None
        for p in pts:
            new = max(level or 0, p["qps"])
            if level is None:
                path.append(f"M{X(p[xkey]):.1f},{Y(new):.1f}")
            elif new != level:
                path.append(f"L{X(p[xkey]):.1f},{Y(level):.1f} L{X(p[xkey]):.1f},{Y(new):.1f}")
            level = new
        last_x = X(end_x[side])
        path.append(f"L{last_x:.1f},{Y(level):.1f}")
        dash = ' stroke-dasharray="6 4"' if passing else ""
        o.append(f'<path class="ln" d="{" ".join(path)}" stroke="var({COLOR[side]})"{dash}/>')
        return (last_x, Y(level)), level

    for side in ("tc", "vm"):
        c = COLOR[side]
        end, level = line(side, False)
        ends.append((end, NAME[side], f"{level:,.0f}", c))
        end, level = line(side, True)
        ends.append((end, f"{NAME[side]}, passing every check", f"{level:,.0f}", c))
    for side in ("tc", "vm"):
        c = COLOR[side]
        for p in points[side]:
            cx, cy = X(p[xkey]), Y(p["qps"])
            state = state_word(p)
            tip = {"t": f"{NAME[side]}: {p['short']}", "r": [
                {"c": c, "n": "images/s", "v": f"{p['qps']:,.0f}"}, {"n": "agent time", "v": f"{p['minutes']:.0f} min"},
                {"n": "agent tokens, list price", "v": f"${p['cost']:.2f}"}, {"n": "checks", "v": state}]}
            o.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="9" tabindex="0" data-tip="{esc(json.dumps(tip))}"/>')
            if not p["eligible"]:
                o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="var(--surface)" stroke="var(--muted)" stroke-width="1.5" pointer-events="none"/>')
            elif p["struck"]:
                cross = f"M{cx-4:.1f},{cy-4:.1f} L{cx+4:.1f},{cy+4:.1f} M{cx-4:.1f},{cy+4:.1f} L{cx+4:.1f},{cy-4:.1f}"
                o.append(f'<path d="{cross}" stroke="var(--surface)" stroke-width="5" stroke-linecap="round" fill="none" pointer-events="none"/>')
                o.append(f'<path d="{cross}" stroke="var({c})" stroke-width="2" stroke-linecap="round" fill="none" pointer-events="none"/>')
            elif p["passes"] is False:
                o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="var(--surface)" stroke="var({c})" stroke-width="2" pointer-events="none"/>')
            else:
                o.append(f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="var({c})" pointer-events="none"/>')
            rows.append([NAME[side], p["short"], f"{p['minutes']:.0f}", f"{p['cost']:.2f}", f"{p['qps']:,.0f}", state])
    ys = C._spread([e[0][1] for e in ends], gap=15)
    for (pt, name, value, c), y in zip(ends, ys):
        lx = L + pw + 8
        if abs(pt[0] - (L + pw)) > 2 or abs(y - pt[1]) > 3:
            o.append(f'<line class="axis" x1="{pt[0]+6:.1f}" y1="{pt[1]:.1f}" x2="{lx-2:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="lab" x="{lx:.1f}" y="{y+4:.1f}">{esc(name)} <tspan class="val">{value}</tspan></text>')
    svg = f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(o) + "</svg>"
    return svg + C.table_view(["side", "step", "agent minutes", "agent tokens, $", "images/s", "checks"], rows)


def mark(kind, c="var(--ink2)"):
    inner = {"dot": f'<circle cx="6" cy="6" r="4" fill="{c}"/>', "ring": f'<circle cx="6" cy="6" r="3.5" fill="none" stroke="{c}" stroke-width="2"/>',
             "cross": f'<path d="M2,2 L10,10 M2,10 L10,2" stroke="{c}" stroke-width="2" stroke-linecap="round"/>',
             "small": '<circle cx="6" cy="6" r="3" fill="none" stroke="var(--muted)" stroke-width="1.5"/>'}[kind]
    return f'<svg width="12" height="12" style="width:12px;display:inline">{inner}</svg>'


def swatch(c, dashed=False):
    dash = ' stroke-dasharray="5 3"' if dashed else ""
    return f'<svg width="24" height="8" style="width:24px;display:inline"><line x1="1" x2="23" y1="4" y2="4" stroke="var({c})" stroke-width="2" stroke-linecap="round"{dash}/></svg>'


LEGEND = ('<div class="legend">'
          f'<span>{swatch("--s2")}torch.compile, best recorded so far</span>'
          f'<span>{swatch("--s2", True)}torch.compile, best so far that passes every check</span>'
          f'<span>{swatch("--s1")}torch.vm, best recorded so far</span>'
          f'<span>{swatch("--s1", True)}torch.vm, best so far that passes every check</span>'
          f'<span>{mark("dot")}measurement of a state that passes every check</span>'
          f'<span>{mark("ring")}of one that does not</span>'
          f'<span>{mark("cross")}struck: the source fails one of its side\'s tests</span>'
          f'<span>{mark("small")}refused by the harness</span>'
          '</div>')


def table(header, rows, num=(), cls=(), wide=False):
    """rows of cells (HTML as it is); num: indices of right-aligned columns; cls: a class per row."""
    h = "".join(f"<th{' class=num' if i in num else ''}>{c}</th>" for i, c in enumerate(header))
    b = "".join(f'<tr{" class=" + chr(34) + cls[k] + chr(34) if cls and cls[k] else ""}>' + "".join(f"<td{' class=num' if i in num else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
                for k, r in enumerate(rows))
    return f'<div class="scroll"><table{" class=wide" if wide else ""}><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def found(side, p):
    """What the checks found for one row, in a few words, for the tables of rows."""
    if not p["eligible"]:
        return "refused by the harness: a batch of another size raised an error"
    lost = ""
    if p["failed"]:
        twin = next((q for q in points[side] if q["i"] < p["i"] and q["eligible"] and q["failed"] == p["failed"] and len(p["failed"]) > 2), None)
        lost = f"not followed: the same as row {twin['i']}" if twin else probed(p)
    if p["struck"]:
        names = ", ".join(n.replace("_cuda", "") for n in p["broken"])
        return f"<b>struck</b>: its source fails {esc(names)}" + (f"; {esc(lost)}" if lost else "; the probes are all followed")
    if p["failed"]:
        return f"<b>does not pass</b>; {esc(lost)}"
    if p["missing"]:
        return "no failure found; not yet run: " + esc(", ".join(p["missing"]))
    return "passes every check"


def steps_table(side):
    rows, cls = [], []
    for p in points[side]:
        acc = f"{p['infer']['compiled_bfloat16']['mAP@0.5']:.3f} / {p['infer']['eager_bfloat16']['mAP@0.5']:.3f}" if p["infer"] else ("" if not p["eligible"] else "not run")
        qps = f"<s>{p['qps']:,.0f}</s>" if p["struck"] or not p["eligible"] else f"{p['qps']:,.0f}"
        rows.append([p["i"], p["utc"][11:16], f"{p['minutes']:.0f}", f"{p['cost']:.2f}", qps, f'<a href="#{p["anchor"]}">{esc(p["short"])}</a>', acc, found(side, p)])
        cls.append("off" if p["struck"] or not p["eligible"] else "")
    return table(["row", "UTC", "min", "$", "images/s", "step", "mAP@0.5, compiled / eager", "what the checks found"], rows, num=(0, 2, 3, 4), cls=cls, wide=True)


def groups(side):
    """The states of a side that count, grouped by configuration and by what the probes found: [(label, states, outcomes)]."""
    out = []
    for kind in ("checked", "unchecked") if side == "tc" else ("safe", "fast"):
        seen = {}
        for p in points[side]:
            if p["eligible"] and p["kind"] == kind:
                seen.setdefault(tuple(p["checks"][name] for name, _, _, _ in PROBES), []).append(p)
        for key, states in seen.items():
            # Where the states of one configuration did not come out alike, each group is named by its rows.
            rows = f", row{'s' if len(states) > 1 else ''} {rows_text([p['i'] for p in states])}" if len(seen) > 1 else ""
            out.append((KIND[kind] + rows, states, key))
    return out


WORD = {"ok": "followed", "no": "not followed", "no+": "not followed, and wrong after it is undone", "undo": "followed, but wrong after it is undone", "error": "raises an error", None: "not run"}
RIGHT = {"ok": "right", "no": "wrong", "error": "raises an error", None: "not run"}
KEPT = {"ok": "unchanged", "no": "changed by the later calls", "error": "raises an error", None: "not run"}


def kept_word(states, name, value):
    """For the probes of kept results: how many of the result's tensors the later calls changed, where the states agree."""
    if value != "no":
        return KEPT[value]
    counts = {(p["raw"][name].get("tensors"), p["raw"][name].get("unchanged")) for p in states}
    if len(counts) == 1 and None not in next(iter(counts)):
        n, same = next(iter(counts))
        return f"all {n} tensors of the result changed by the later calls" if same == 0 else f"{n - same} of the result's {n} tensors changed by the later calls"
    return KEPT[value]


def probe_table():
    cols = [(side, label, states, key) for side in ("tc", "vm") for label, states, key in groups(side)]
    head = ["after the callable is warm", "told to the agents"]
    for side, label, states, _ in cols:
        qs = sorted(p["qps"] for p in states)
        span = f"{qs[0]:,.0f} to {qs[-1]:,.0f}" if len(qs) > 1 else f"{qs[0]:,.0f}"
        head.append(f"{NAME[side]}, {label}: {len(states)} state{'s' if len(states) != 1 else ''}, {span}")
    rows = []
    for k, (name, short, kind, told) in enumerate(PROBES):
        words = RIGHT if short in ("batch size", "image size", "timed shape again") else WORD
        cells = []
        for _, _, states, key in cols:
            word = kept_word(states, name, key[k]) if short.startswith("results kept") else words.get(key[k], WORD[key[k]])
            cells.append(word if key[k] in ("ok", None) else f"<b>{word}</b>")
        rows.append([esc(LONG[name]), "yes" if told else "no"] + cells)
    return table(head, rows)


def struck_table():
    rows = []
    for side in ("tc", "vm"):
        for p in points[side]:
            if p["struck"]:
                rows.append([NAME[side], p["i"], f"<s>{p['qps']:,.0f}</s>", f'<a href="#{p["anchor"]}">{esc(p["short"])}</a>', KIND1[p["kind"]], tests_result(side, p)])
    return table(["side", "row", "images/s", "step", "configuration", "its side's tests on this row's source"], rows, num=(1, 2)) if rows else "<p>No row is struck.</p>"


def acc_table():
    rows = []
    picks = []
    for side in ("tc", "vm"):
        counted = [p for p in points[side] if p["counts"]]
        passing = [p for p in counted if holds(p, True)]
        picks += [(side, points[side][0], "start"), (side, max(passing, key=lambda p: p["qps"]), "best that passes every check"), (side, max(counted, key=lambda p: p["qps"]), "best recorded"),
                  (side, points[side][-1], "last row")]
    merged = []  # one line per row, named for everything it is
    for side, p, what in picks:
        same = next((m for m in merged if m[0] == side and m[1] is p), None)
        if same:
            same[2] += " and " + what
        else:
            merged.append([side, p, what])
    picks = merged
    first = next((p["infer"] for _, p, _ in picks if p["infer"]), None)
    if first:
        d = first["eager_bfloat16_against_float32"]
        rows.append(["eager, float32", "", f"{first['eager_float32']['mAP@0.5']:.3f}", "0", "all"])
        rows.append(["eager, bfloat16", f"{EAGER:,.0f}", f"{first['eager_bfloat16']['mAP@0.5']:.3f}", f"{d['raw_output']['rmse']:.3f}", f"{d['detections_iou_0.5']['found']} of {d['detections_iou_0.5']['detections']}"])
    for side, p, what in picks:
        r = p["infer"]
        if not r:
            rows.append([f"{NAME[side]}, {what}", f"{p['qps']:,.0f}", "not run", "", ""])
            continue
        d = r["compiled_against_float32"]
        rows.append([f"{NAME[side]}, {what}", f"{p['qps']:,.0f}", f"{r['compiled_bfloat16']['mAP@0.5']:.3f}", f"{d['raw_output']['rmse']:.3f}", f"{d['detections_iou_0.5']['found']} of {d['detections_iou_0.5']['detections']}"])
    return table(["", "images/s", "mAP@0.5 on 128 COCO images", "error against float32 (RMSE of the raw output)", "float32's detections found"], rows, num=(1, 2, 3, 4))


def four_states():
    """The states the cross-cache and speed-after-events checks were run on: each side's highest row and its last."""
    out = []
    for side in ("tc", "vm"):
        for p in points[side]:
            if load(f"{OUT}/after_{side}_{p['i']:02d}.json") or load(f"{OUT}/xcache_{side}_{p['i']:02d}_pretrained.json"):
                out.append((side, p))
    return out


def state_name(side, p):
    return f"{NAME[side]}, {KIND1[p['kind']]}, {'<s>' if p['struck'] else ''}{p['qps']:,.0f}{'</s> (struck)' if p['struck'] else ''}"


def after_table():
    data = [(state_name(side, p), load(f"{OUT}/after_{side}_{p['i']:02d}.json")) for side, p in four_states()]
    data = [(n, d) for n, d in data if d]
    keys = ("at first", "after a batch of 3 images", "after images of another size", "after a weight was updated in place")
    return table(["throughput at the timed shape, images/s"] + [n for n, _ in data], [[k] + [f"{d[k]:,.0f}" for _, d in data] for k in keys], num=range(1, 9)) if data else ""


def cache_table():
    rows = []
    for side, p in four_states():
        r = load(f"{OUT}/xcache_{side}_{p['i']:02d}_pretrained.json")
        if not r:
            continue
        d = r["compiled_against_float32"]
        rows.append([state_name(side, p), f"{r['compiled_bfloat16']['mAP@0.5']:.3f}", f"{r['eager_bfloat16']['mAP@0.5']:.3f}", f"{d['raw_output']['rmse']:.3f}",
                     f"{d['detections_iou_0.5']['found']} of {d['detections_iou_0.5']['detections']}"])
    return table(["state, compiled first for the random-weight model, then for the pretrained one", "mAP@0.5", "eager's", "error against float32", "float32's detections found"], rows, num=(1, 2, 3)) if rows else ""


def tests_table():
    rows = []
    for side in ("tc", "vm"):
        seen = []
        for p in points[side]:
            if p["source"] in seen:
                continue
            seen.append(p["source"])
            same = [q["i"] for q in points[side] if q["source"] == p["source"]]
            rows.append([NAME[side], rows_text(same), ran_on(side, p), tests_result(side, p)])
    return table(["side", "rows with this source", "its side's tests that were run on it", "result"], rows)


FINAL_TESTS = [("tests_vm_cpu.log", "torch.vm: test/test_vm.py on the CPU"), ("tests_vm_gpu.log", "torch.vm: test/test_vm.py on the GPU"),
               ("tests_tc_modules.log", "torch.compile: test/dynamo/test_modules.py"), ("tests_tc_misc.log", "torch.compile: test/dynamo/test_misc.py"),
               ("tests_tc_guard_manager.log", "torch.compile: test/dynamo/test_guard_manager.py"),
               ("tests_tc_inductor_gpu.log", "torch.compile: test/inductor/test_torchinductor.py on the GPU, tests of convolutions, max pools and cat")]


def final_tests_table():
    rows = []
    for name, what in FINAL_TESTS:
        t = tests_log(f"{OUT}/{name}")
        if t:
            rows.append([what, f"{t['ran']:,}", "pass" if not t["bad"] else f"{len(t['bad'])} not passing: {', '.join(t['bad'])}"])
    return table(["on each side's final source", "tests", "result"], rows, num=(1,)) if rows else ""


def final_tests(name):
    return tests_log(f"{OUT}/{name}")


# Whether the failure in test_misc.py on torch.compile's final source is the tree setup: the test's subprocess
# imports the shared checkout, which has no libraries, and never reaches the side's tree.
_misc = f"{OUT}/tests_tc_misc.log"
MISC_IS_SETUP = os.path.exists(_misc) and "torch is not fully installed" in open(_misc, errors="replace").read()


def ahead_table():
    rows = []
    for x in (30, 60, 90, 120, 180, 240, 300, 360):
        rows.append([f"{x} minutes"] + [f"{best_at(s, 'minutes', x, passing):,.0f}" for passing in (True, False) for s in ("tc", "vm")])
    for x in (10, 20, 30, 40, 50):
        rows.append([f"${x} of tokens"] + [f"{best_at(s, 'cost', x, passing):,.0f}" for passing in (True, False) for s in ("tc", "vm")])
    return table(["best so far, images/s, when the agent had spent", "torch.compile, passing every check", "torch.vm, passing every check", "torch.compile, best recorded", "torch.vm, best recorded"], rows, num=(1, 2, 3, 4))


TRAIN = (("det_", "deterministic kernels, float32"), ("", "default kernels, float32"), ("amp_", "default kernels, autocast to bfloat16"))
RUNS = (("eager_a", "eager"), ("eager_b", "eager, run again"), ("eager_nudge_1e-6", "eager, one layer's weights x (1 + 1e-6)"), ("tc_start", "torch.compile, starting source"),
        ("tc_final", "torch.compile, final source"), ("vm_start", "torch.vm, starting source"), ("vm_final", "torch.vm, final source"))


def train_runs(prefix):
    return {key: load(f"{OUT}/train_{prefix}{key}.json") for key, _ in RUNS}


def loss_figure():
    runs = train_runs("det_")
    if not all(runs[k] for k in ("eager_a", "eager_nudge_1e-6", "tc_final", "vm_final")):
        return ""
    per = 8
    passes = min(r["steps"] for r in runs.values() if r) // per
    mean = lambda r: [sum(r["loss"][i * per:(i + 1) * per]) / per for i in range(passes)]
    series = [{"name": "eager", "c": "--ink2", "y": mean(runs["eager_a"])},
              {"name": "eager, one layer's weights changed in the 6th digit", "short": "eager, nudged", "c": "--base", "y": mean(runs["eager_nudge_1e-6"]), "dash": True},
              {"name": "torch.compile, final source", "short": "torch.compile", "c": "--s2", "y": mean(runs["tc_final"])},
              {"name": "torch.vm, final source", "short": "torch.vm", "c": "--s1", "y": mean(runs["vm_final"])}]
    top = (int(max(max(s["y"]) for s in series) / 5) + 1) * 5
    body = C.line_chart(series, list(range(1, passes + 1)), xlabels=[str(i) for i in range(1, passes + 1)], ymax=top, ymin=0, yticks=list(range(0, top + 1, 5)),
                        xtick_idx=[i for i in range(passes) if (i + 1) % 5 == 0 or i == 0], xlab="pass over the 128 images (8 steps each)", right=150,
                        yfmt=lambda v: f"{v:.0f}", vfmt=lambda v: f"{v:.3f}", tv_header="pass")
    return C.figure("Training with the final sources: loss per pass, deterministic kernels, float32",
                    "yolov3-spp from a seeded random start, the model's own detection loss and SGD settings, 300 steps of 16 images. Forward and backward through eager, torch.compile or torch.vm with Inductor graphs, each in its default configuration.", body)


def train_table():
    rows = []
    for prefix, label in TRAIN:
        runs = train_runs(prefix)
        ref = runs["eager_a"]
        if not ref:
            continue
        for key, name in RUNS:
            r = runs[key]
            if not r:
                continue
            n = min(r["steps"], ref["steps"])
            d = [abs(a - b) / abs(b) for a, b in zip(r["loss"][:n], ref["loss"][:n])]
            first = next((i for i, (a, b) in enumerate(zip(r["loss"], ref["loss"])) if a != b), None)
            comp = r.get("compiled") or {}
            graphs = comp.get("unique_graphs", comp.get("graphs", ""))
            if "operations" in comp:
                graphs = f"{graphs}, of {comp['operations']} operations"
            rows.append([label, name, f"{r['loss'][n-1]:.3f}", "" if key == "eager_a" else f"{100*statistics.mean(d):.2f}%", "" if key == "eager_a" else ("never" if first is None else str(first + 1)),
                         f"{r['ms_per_step_last_50']:.0f}", "" if key.startswith("eager") else f"{r['ms_per_step_last_50'] / ref['ms_per_step_last_50']:.2f}", graphs])
    return table(["settings", "forward and backward through", "loss, step 300", "mean distance from eager's loss", "first step that differs", "ms per step", "of eager's time", "graphs compiled"],
                 rows, num=(2, 3, 4, 5, 6)) if rows else ""


def train_ms(prefix, key):
    r = load(f"{OUT}/train_{prefix}{key}.json")
    return r["ms_per_step_last_50"] if r else None


if __name__ == "__main__":
    import text

    numbers = {
        "start": {s: points[s][0]["qps"] for s in points}, "best": {s: final_best(s) for s in points}, "passing": {s: final_best(s, True) for s in points},
        "total": total, "minutes": minutes,
    }
    body = text.body(sys.modules[__name__], numbers)
    page = f"{text.head(C, A)}{body}{text.FOOTER}\n<script>{C.JS}</script>\n</body>\n</html>\n"
    open(f"{HERE}/index.html", "w").write(page)
    print("wrote", f"{HERE}/index.html", len(page))
    if "--site" in sys.argv:
        dest = sys.argv[sys.argv.index("--site") + 1]
        os.makedirs(dest, exist_ok=True)
        open(os.path.join(dest, "index.html"), "w").write(page)
        print("wrote", os.path.join(dest, "index.html"))
    for side in points:
        struck = [round(p["qps"]) for p in points[side] if p["struck"]]
        print(side, "start", round(numbers["start"][side]), "best recorded", round(numbers["best"][side]), "best passing", round(numbers["passing"][side]), "struck", struck,
              "cost", round(total[side], 2), "minutes", round(minutes[side]), "does not pass", [round(p["qps"]) for p in points[side] if p["counts"] and p["passes"] is False])
    print("eager", round(EAGER))
    for side, i, missing in INCOMPLETE:
        print(f"INCOMPLETE {side}_{i:02d}: not run: {', '.join(missing)}")
