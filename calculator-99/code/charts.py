"""Tiny static-SVG chart kit for the write-up (line, horizontal bar, x/y families, heat table).
Colors are CSS custom properties so light/dark swap in one place; every chart ships a legend
(when it has >= 2 series), a hover layer and a table view."""
import html
import json

CSS = """
:root { color-scheme: light dark;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --border:rgba(11,11,11,.10); --code:#f0efec; --link:#1c5cab;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#6250d6; --s8:#e34948;
  --base:#898781; --seq-lo:#cde2fb; --seq-hi:#184f95; }
@media (prefers-color-scheme: dark) { :root {
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --border:rgba(255,255,255,.10); --code:#232321; --link:#86b6ef;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#0ca30c; --s7:#9085e9; --s8:#e66767;
  --base:#898781; --seq-lo:#0d366b; --seq-hi:#9ec5f4; } }
body { font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:var(--page);
  max-width:860px; margin:40px auto; padding:0 20px; }
h1 { font-size:1.75em; line-height:1.2; margin-bottom:.2em; }
h2 { margin-top:2.4em; padding-bottom:.25em; border-bottom:1px solid var(--grid); }
h3 { margin-top:1.8em; }
a { color:var(--link); }
.sub { color:var(--ink2); margin-top:0; }
code { font:.9em ui-monospace,SFMono-Regular,Menlo,monospace; background:var(--code); padding:.1em .35em; border-radius:4px; }
pre { background:var(--code); border:1px solid var(--border); border-radius:6px; padding:12px 14px; overflow-x:auto; line-height:1.45; }
pre code { background:none; padding:0; }
.tldr { border-left:4px solid var(--s1); background:var(--surface); padding:10px 18px; border-radius:0 8px 8px 0; }
.note { border-left:4px solid var(--muted); background:var(--surface); padding:8px 18px; border-radius:0 8px 8px 0; color:var(--ink2); }
figure { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 18px 10px; margin:1.6em 0; }
figure h4 { margin:0 0 2px; font-size:1em; }
figure .cap { color:var(--ink2); font-size:.9em; margin:0 0 10px; }
figure svg { width:100%; height:auto; display:block; overflow:visible; }
.legend { display:flex; flex-wrap:wrap; gap:4px 16px; font-size:.85em; color:var(--ink2); margin:2px 0 8px; }
.legend span { display:inline-flex; align-items:center; gap:6px; }
.legend i { display:inline-block; width:16px; height:0; border-top:2px solid; border-radius:2px; }
.legend i.box { width:10px; height:10px; border:0; border-radius:2px; }
.legend i.dash { border-top-style:dashed; }
svg text { font:12px system-ui,-apple-system,"Segoe UI",sans-serif; fill:var(--muted); font-variant-numeric:tabular-nums; }
svg text.lab { fill:var(--ink2); }
svg text.val { fill:var(--ink); font-weight:600; }
svg .grid { stroke:var(--grid); stroke-width:1; }
svg .axis { stroke:var(--axis); stroke-width:1; }
svg .ln { fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }
svg .dot { stroke:var(--surface); stroke-width:2; }
svg .hit { fill:transparent; stroke:none; cursor:default; }
svg .hit:hover + .dot, svg .hit:focus + .dot { stroke:var(--ink); }
svg .bar:hover { filter:brightness(1.12); }
svg .xh { stroke:var(--muted); stroke-width:1; visibility:hidden; }
details.tv { font-size:.85em; color:var(--ink2); margin-top:6px; }
details.tv summary { cursor:pointer; color:var(--muted); }
table { border-collapse:collapse; width:100%; margin:.6em 0; font-size:.92em; }
th, td { border-bottom:1px solid var(--grid); padding:5px 8px; text-align:left; vertical-align:top; }
th { color:var(--ink2); font-weight:600; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.heat td.cell { text-align:center; font-variant-numeric:tabular-nums; border:2px solid var(--surface); padding:4px 2px; border-radius:4px; }
.scroll { overflow-x:auto; }
#tip { position:fixed; pointer-events:none; z-index:10; background:var(--surface); color:var(--ink); border:1px solid var(--border);
  border-radius:8px; padding:8px 10px; font-size:.82em; box-shadow:0 4px 16px rgba(0,0,0,.18); display:none; max-width:320px; }
#tip .t { color:var(--ink2); margin-bottom:4px; }
#tip .r { display:flex; align-items:center; gap:8px; white-space:nowrap; }
#tip .r i { width:14px; border-top:2px solid; display:inline-block; }
#tip .r b { margin-left:auto; padding-left:14px; font-variant-numeric:tabular-nums; }
.seq { font:13px ui-monospace,SFMono-Regular,Menlo,monospace; display:flex; flex-wrap:wrap; gap:3px; align-items:center; margin:4px 0 12px; }
.seq b { font-weight:400; border:1px solid var(--axis); border-radius:4px; padding:1px 6px; background:var(--surface); }
.seq b.th { border-color:var(--s1); box-shadow:inset 0 -3px 0 var(--s1); }
.seq b.um { border-color:var(--s2); box-shadow:inset 0 -3px 0 var(--s2); }
.seq b.dt { border-color:var(--s3); box-shadow:inset 0 -3px 0 var(--s3); }
.seq b.sc { border-color:var(--s6); box-shadow:inset 0 -3px 0 var(--s6); }
.seq b.an { border-color:var(--ink2); font-weight:600; }
.seq span { color:var(--muted); font-family:system-ui,sans-serif; font-size:12px; margin:0 4px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:0 18px; }
@media (max-width:700px) { .grid2 { grid-template-columns:1fr; } }
footer { margin-top:3em; color:var(--ink2); font-size:.9em; border-top:1px solid var(--grid); padding-top:1em; }
"""

JS = """
(function(){
  const tip = document.createElement('div'); tip.id = 'tip'; document.body.appendChild(tip);
  function place(ev){ const pad=14, w=tip.offsetWidth, h=tip.offsetHeight;
    let x=ev.clientX+pad, y=ev.clientY+pad;
    if (x+w>innerWidth-8) x=ev.clientX-w-pad; if (y+h>innerHeight-8) y=ev.clientY-h-pad;
    tip.style.left=Math.max(4,x)+'px'; tip.style.top=Math.max(4,y)+'px'; }
  function show(title, rows, ev){ tip.textContent='';
    const t=document.createElement('div'); t.className='t'; t.textContent=title; tip.appendChild(t);
    for (const r of rows){ const d=document.createElement('div'); d.className='r';
      if (r.c){ const i=document.createElement('i'); i.style.borderTopColor='var('+r.c+')'; d.appendChild(i); }
      const n=document.createElement('span'); n.textContent=r.n; d.appendChild(n);
      const b=document.createElement('b'); b.textContent=r.v; d.appendChild(b); tip.appendChild(d); }
    tip.style.display='block'; place(ev); }
  function hide(){ tip.style.display='none'; }
  // per-mark tooltips
  document.querySelectorAll('[data-tip]').forEach(el=>{
    const d=JSON.parse(el.getAttribute('data-tip'));
    const on=ev=>show(d.t, d.r, ev.clientX!==undefined?ev:{clientX:el.getBoundingClientRect().right,clientY:el.getBoundingClientRect().top});
    el.addEventListener('pointermove', on); el.addEventListener('focus', on);
    el.addEventListener('pointerleave', hide); el.addEventListener('blur', hide); });
  // crosshair for shared-x line charts
  document.querySelectorAll('svg[data-xh]').forEach(svg=>{
    const d=JSON.parse(svg.getAttribute('data-xh')); const xh=svg.querySelector('.xh');
    svg.addEventListener('pointermove', ev=>{
      const r=svg.getBoundingClientRect(); const x=(ev.clientX-r.left)/r.width*d.w;
      let bi=0, bd=1e9; d.px.forEach((p,i)=>{ const dd=Math.abs(p-x); if(dd<bd){bd=dd;bi=i;} });
      xh.setAttribute('x1', d.px[bi]); xh.setAttribute('x2', d.px[bi]); xh.style.visibility='visible';
      show(d.xl[bi], d.s.filter(s=>s.v[bi]!==null).map(s=>({c:s.c,n:s.n,v:s.v[bi]})), ev); });
    svg.addEventListener('pointerleave', ()=>{ xh.style.visibility='hidden'; hide(); }); });
})();
"""


def esc(s):
    return html.escape(str(s), quote=True)


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def table(header, rows, num_from=1, cls=""):
    h = "".join(f"<th{' class=num' if i >= num_from else ''}>{esc(c)}</th>" for i, c in enumerate(header))
    b = "".join("<tr>" + "".join(f"<td{' class=num' if i >= num_from else ''}>{c if str(c).startswith('<') else esc(c)}</td>"
                                for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def table_view(header, rows):
    return f'<details class="tv"><summary>Table view</summary>{table(header, rows)}</details>'


def legend(items):
    """items: [(name, css var, kind)] with kind in line|dash|box."""
    if len(items) < 2:
        return ""
    out = []
    for name, c, kind in items:
        style = f"background:var({c})" if kind == "box" else f"border-top-color:var({c})"
        out.append(f'<span><i class="{"box" if kind == "box" else ("dash" if kind == "dash" else "")}" style="{style}"></i>{esc(name)}</span>')
    return '<div class="legend">' + "".join(out) + "</div>"


def figure(title, cap, body):
    return f'<figure><h4>{esc(title)}</h4><p class="cap">{cap}</p>{body}</figure>'


def _spread(ys, gap=13):
    """Nudge label y positions apart (pixels) while keeping order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def line_chart(series, xs, xlabels=None, w=760, h=300, ymax=1.0, ymin=0.0, yticks=(0, .25, .5, .75, 1.0), xtick_idx=None,
               xlab="", end_labels=True, vlines=(), right=118, yfmt=lambda v: f"{100*v:.0f}%", vfmt=pct, tv_header=None,
               show_legend=True, tv=True, ylog=False):
    """Shared-x multi-series line chart with a crosshair tooltip.
    series: [{name, c (css var), y: [..] (None = missing), dash: bool}]"""
    L, R, T, B = 40, right if end_labels else 14, 10, 34 if xlab else 24
    pw, ph = w - L - R, h - T - B
    n = len(xs)
    px = [L + (pw * i / (n - 1) if n > 1 else pw / 2) for i in range(n)]
    if ylog:  # logarithmic y axis (ymin > 0)
        import math
        Y = lambda v: T + ph * (1 - (math.log10(max(v, ymin)) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)))
    else:
        Y = lambda v: T + ph * (1 - (v - ymin) / (ymax - ymin))
    xlabels = xlabels or [str(x) for x in xs]
    o = []
    for t in yticks:
        o.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text x="{L-6}" y="{Y(t)+4:.1f}" text-anchor="end">{yfmt(t)}</text>')
    for xi, lab in vlines:  # xi may be a fractional index
        vx = L + pw * xi / max(n - 1, 1)
        o.append(f'<line class="axis" x1="{vx:.1f}" x2="{vx:.1f}" y1="{T}" y2="{T+ph}"/>')
        o.append(f'<text x="{vx+4:.1f}" y="{T+10}">{esc(lab)}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{L+pw}" y1="{Y(ymin):.1f}" y2="{Y(ymin):.1f}"/>')
    for i in (xtick_idx if xtick_idx is not None else range(n)):
        o.append(f'<text x="{px[i]:.1f}" y="{T+ph+15}" text-anchor="middle">{esc(xlabels[i])}</text>')
    if xlab:
        o.append(f'<text x="{L+pw/2:.1f}" y="{h-3}" text-anchor="middle" class="lab">{esc(xlab)}</text>')
    ends = []
    for s in series:
        pts = [(px[i], Y(v)) for i, v in enumerate(s["y"]) if v is not None]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if s.get("dash") else ""
        o.append(f'<path class="ln" d="{d}" stroke="var({s["c"]})"{dash}/>')
        o.append(f'<circle class="dot" cx="{pts[-1][0]:.1f}" cy="{pts[-1][1]:.1f}" r="4" fill="var({s["c"]})"/>')
        ends.append((pts[-1], s))
    if end_labels:
        ys = _spread([e[0][1] for e in ends])
        for (pt, s), y in zip(ends, ys):
            if abs(y - pt[1]) > 3:
                o.append(f'<line class="axis" x1="{pt[0]+5:.1f}" y1="{pt[1]:.1f}" x2="{pt[0]+11:.1f}" y2="{y:.1f}"/>')
            last = [v for v in s["y"] if v is not None][-1]
            o.append(f'<text class="lab" x="{pt[0]+13:.1f}" y="{y+4:.1f}">{esc(s.get("short", s["name"]))} <tspan class="val">{vfmt(last)}</tspan></text>')
    o.append(f'<line class="xh" x1="0" x2="0" y1="{T}" y2="{T+ph}"/>')
    data = {"w": w, "px": [round(p, 1) for p in px], "xl": [f"{xlab + ': ' if xlab else ''}{l}" for l in xlabels],
            "s": [{"n": s["name"], "c": s["c"], "v": [None if v is None else vfmt(v) for v in s["y"]]} for s in series]}
    svg = f'<svg viewBox="0 0 {w} {h}" role="img" data-xh="{esc(json.dumps(data))}">' + "".join(o) + "</svg>"
    leg = legend([(s["name"], s["c"], "dash" if s.get("dash") else "line") for s in series]) if show_legend else ""
    rows = [[xlabels[i]] + ["" if s["y"][i] is None else vfmt(s["y"][i]) for s in series] for i in range(n)]
    return leg + svg + (table_view([tv_header or xlab or "x"] + [s["name"] for s in series], rows) if tv else "")


def hbar_chart(items, w=760, xmax=1.0, xticks=(0, .25, .5, .75, 1.0), label_w=210, vfmt=pct, bar=16, gap=12, groups=None):
    """items: [{name, v, c, dots: [..], note}] one bar per row, value at the tip; optional per-seed dots."""
    L, R, T, B = label_w, 64, 6, 22
    pw = w - L - R
    h = T + B + len(items) * (bar + gap)
    X = lambda v: L + pw * v / xmax
    o = []
    for t in xticks:
        o.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{h-B}"/>')
        o.append(f'<text x="{X(t):.1f}" y="{h-6}" text-anchor="middle">{100*t:.0f}%</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{L}" y1="{T}" y2="{h-B}"/>')
    for i, it in enumerate(items):
        y = T + gap / 2 + i * (bar + gap)
        x1 = X(it["v"])
        r = 4
        d = f"M{L},{y} H{max(L, x1-r):.1f} Q{x1:.1f},{y} {x1:.1f},{y+r} V{y+bar-r} Q{x1:.1f},{y+bar} {max(L, x1-r):.1f},{y+bar} H{L} Z"
        tip = {"t": it["name"], "r": [{"n": "exact match", "v": vfmt(it["v"])}] + [{"n": k, "v": v} for k, v in it.get("tip", [])]}
        o.append(f'<path class="bar" d="{d}" fill="var({it["c"]})" tabindex="0" data-tip="{esc(json.dumps(tip))}"/>')
        o.append(f'<text class="lab" x="{L-8}" y="{y+bar/2+4:.1f}" text-anchor="end">{esc(it["name"])}</text>')
        for dv in it.get("dots", []):
            o.append(f'<circle class="dot" cx="{X(dv):.1f}" cy="{y+bar/2:.1f}" r="3.5" fill="var(--ink2)"/>')
        o.append(f'<text class="val" x="{max(x1, max([X(d) for d in it.get("dots", [])] or [0]))+8:.1f}" y="{y+bar/2+4:.1f}">{vfmt(it["v"])}</text>')
    svg = f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(o) + "</svg>"
    rows = [[it["name"], vfmt(it["v"])] + [vfmt(d) for d in it.get("dots", [])] for it in items]
    nd = max([len(it.get("dots", [])) for it in items] or [0])
    return svg + table_view(["variant", "exact match"] + [f"seed {i}" for i in range(nd)], rows)


def xy_chart(families, points=(), w=760, h=340, xlab="", ylab="", xmin=None, xmax=None, ymin=0.0, ymax=1.0, yticks=None, xticks=None,
             vfmt=pct, xfmt=lambda v: f"{v:.0f}", right=130):
    """families: [{name, c, pts: [(x, y, label)]}] drawn as connected lines; points: [{name, c, x, y}] labelled singles."""
    allx = [p[0] for f in families for p in f["pts"]] + [p["x"] for p in points]
    xmin = min(allx) if xmin is None else xmin
    xmax = max(allx) if xmax is None else xmax
    L, R, T, B = 44, right, 10, 36
    pw, ph = w - L - R, h - T - B
    X = lambda v: L + pw * (v - xmin) / (xmax - xmin)
    Y = lambda v: T + ph * (1 - (v - ymin) / (ymax - ymin))
    o = []
    for t in yticks or []:
        o.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text x="{L-6}" y="{Y(t)+4:.1f}" text-anchor="end">{100*t:.0f}%</text>')
    for t in xticks or []:
        o.append(f'<text x="{X(t):.1f}" y="{T+ph+15}" text-anchor="middle">{xfmt(t)}</text>')
    o.append(f'<line class="axis" x1="{L}" x2="{L+pw}" y1="{T+ph}" y2="{T+ph}"/>')
    o.append(f'<text x="{L+pw/2:.1f}" y="{h-3}" text-anchor="middle" class="lab">{esc(xlab)}</text>')
    ends, rows = [], []
    for f in families:
        pts = [(X(x), Y(y)) for x, y, _ in f["pts"]]
        o.append(f'<path class="ln" d="M{" L".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" stroke="var({f["c"]})"/>')
        for (x, y, lab), (cx, cy) in zip(f["pts"], pts):
            tip = {"t": f"{f['name']} · {lab}", "r": [{"c": f["c"], "n": ylab, "v": vfmt(y)}, {"n": xlab, "v": xfmt(x)}]}
            o.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12" tabindex="0" data-tip="{esc(json.dumps(tip))}"/>')
            o.append(f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="var({f["c"]})" pointer-events="none"/>')
            rows.append([f["name"], lab, xfmt(x), vfmt(y)])
        ends.append((pts[-1], f["name"], f["pts"][-1][1]))
    for p in points:
        cx, cy = X(p["x"]), Y(p["y"])
        tip = {"t": p["name"], "r": [{"c": p["c"], "n": ylab, "v": vfmt(p["y"])}, {"n": xlab, "v": xfmt(p["x"])}]}
        o.append(f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12" tabindex="0" data-tip="{esc(json.dumps(tip))}"/>')
        o.append(f'<rect class="dot" x="{cx-4.5:.1f}" y="{cy-4.5:.1f}" width="9" height="9" rx="2" fill="var({p["c"]})" pointer-events="none"/>')
        dx, anchor = (10, "start") if p.get("side", "r") == "r" else (-10, "end")
        o.append(f'<text class="lab" x="{cx+dx:.1f}" y="{cy+4+p.get("dy", 0):.1f}" text-anchor="{anchor}">{esc(p.get("short", p["name"]))}</text>')
        rows.append([p["name"], "", xfmt(p["x"]), vfmt(p["y"])])
    ys = _spread([e[0][1] for e in ends])
    for (pt, name, v), y in zip(ends, ys):
        if abs(y - pt[1]) > 3:
            o.append(f'<line class="axis" x1="{pt[0]+5:.1f}" y1="{pt[1]:.1f}" x2="{pt[0]+11:.1f}" y2="{y:.1f}"/>')
        o.append(f'<text class="lab" x="{pt[0]+13:.1f}" y="{y+4:.1f}">{esc(name)}</text>')
    svg = f'<svg viewBox="0 0 {w} {h}" role="img">' + "".join(o) + "</svg>"
    leg = legend([(f["name"], f["c"], "line") for f in families] + [(p["name"], p["c"], "box") for p in points])
    return leg + svg + table_view(["series", "setting", xlab, ylab], rows)


def heat_table(row_labels, col_labels, values, group_labels=None, vfmt=lambda v: f"{100*v:.0f}", lo=0.0, hi=1.0, first_col="", notes=None):
    """Sequential one-hue heat table; values[r][c] in [lo, hi] or None. Cell text is always printed."""
    h = f"<th>{esc(first_col)}</th>" + ("<th class=num>n</th>" if notes else "") + "".join(f'<th class="num" style="text-align:center">{esc(c)}</th>' for c in col_labels)
    body = []
    for r, lab in enumerate(row_labels):
        if group_labels and group_labels[r]:
            body.append(f'<tr><td colspan="{len(col_labels)+1+(1 if notes else 0)}" style="padding-top:12px;font-weight:600;border:0">{esc(group_labels[r])}</td></tr>')
        cells = []
        for v in values[r]:
            if v is None:
                cells.append('<td class="cell"></td>')
                continue
            t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
            ink = "var(--ink)" if t < 0.55 else "var(--surface)"
            cells.append(f'<td class="cell" style="background:color-mix(in oklab, var(--seq-hi) {100*t:.0f}%, var(--seq-lo));color:{ink}">{vfmt(v)}</td>')
        n = f"<td class=num>{notes[r]}</td>" if notes else ""
        body.append(f"<tr><td>{esc(lab)}</td>{n}{''.join(cells)}</tr>")
    return f'<div class="scroll"><table class="heat"><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def seq(*parts):
    """Token-strip diagram: parts are (text, cls) or plain strings for annotations."""
    o = []
    for p in parts:
        if isinstance(p, str):
            o.append(f"<span>{esc(p)}</span>")
        else:
            o.append(f'<b class="{p[1]}">{esc(p[0])}</b>')
    return '<div class="seq">' + "".join(o) + "</div>"
