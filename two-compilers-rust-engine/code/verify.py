"""verify.py: prints, for every recorded row, what make.py read from the checks, so that the page's statements can
be compared with the files: the probes not followed (by script), the tests of the row's source, struck or not,
passing or not; then the groups of the probe table and the headline numbers with the file each comes from."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
import make as M

for side in ("tc", "vm"):
    print(f"== {M.NAME[side]}")
    for p in M.points[side]:
        by = {kind: [short for name, short, k, _ in M.PROBES if k == kind and p["checks"][name] not in ("ok", None)] for kind in ("probe", "probe2", "probe4")}
        kept = [f"{p['raw'][name].get('unchanged')}/{p['raw'][name].get('tensors')}" for name, short, _, _ in M.PROBES if short.startswith("results kept") and p["raw"][name]]
        not_run = sorted({k for name, _, k, _ in M.PROBES if p["checks"][name] is None})
        acc = f"{p['infer']['compiled_bfloat16']['mAP@0.5']:.4f}/{p['infer']['eager_bfloat16']['mAP@0.5']:.4f}" if p["infer"] else "no infer"
        print(f"  {p['i']:2d} {p['qps']:6.0f} {p['kind']:9s} counted={p['eligible']!s:5s} mAP {acc}  missed: probe {by['probe']} probe2 {by['probe2']} probe4 {by['probe4']} (kept tensors unchanged, as tensors and through DLPack: {kept})"
              f"{' NOT RUN ' + str(not_run) if not_run else ''}  tests runs {p['runs']} bad {p['bad']} broken {p['broken']}  struck={p['struck']} passes={p['passes']}")
    for label, states, key in M.groups(side):
        missed = [short for (name, short, _, _), v in zip(M.PROBES, key) if v not in ("ok", None)]
        not_run = [short for (name, short, _, _), v in zip(M.PROBES, key) if v is None]
        print(f"  group {label}: rows {[p['i'] for p in states]} missed {len(missed)} of {len(M.PROBES)}: {missed}{' NOT RUN: ' + str(not_run) if not_run else ''}")
    best, passing = M.final_best(side), M.final_best(side, True)
    top = max((p for p in M.points[side] if M.holds(p, False)), key=lambda p: p["qps"])
    print(f"  the best recorded state (row {top['i']}) misses {len(top['failed'])} of {len(M.PROBES)}: {top['failed']}")
    print(f"  start {M.points[side][0]['qps']:.0f}  best recorded {best:.0f} (rows {[p['i'] for p in M.points[side] if M.holds(p, False) and p['qps'] == best]})"
          f"  best passing every check {passing:.0f} (rows {[p['i'] for p in M.points[side] if M.holds(p, True) and p['qps'] == passing]})"
          f"  struck {[round(p['qps']) for p in M.points[side] if p['struck']]}  cost ${M.total[side]:.2f}  minutes {M.minutes[side]:.1f}")
print("incomplete:", M.INCOMPLETE or "none")
