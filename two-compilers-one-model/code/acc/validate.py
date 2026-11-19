"""validate.py [--only SIDE] [--what infer,probe] [--last]: for every official checkpoint of both sides that counted
as a result: build its standalone tree, then check its predictions with the pretrained weights on real images
(infer.py) and what it still follows (probe.py). Final checkpoints first. Skips what is already done."""
import json, os, subprocess, sys

H = "/home/dev/local/hill"
what = sys.argv[sys.argv.index("--what") + 1].split(",") if "--what" in sys.argv else ["infer", "probe"]
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
jobs = []
for side in ("tc", "vm"):
    if only and side != only:
        continue
    rows = [json.loads(l) for l in open(f"{H}/{side}/log.jsonl")]
    official = [r for r in rows if r["kind"] == "official"]
    for i, r in enumerate(official):
        if r["eligible"]:
            jobs.append((side, i, r))
last = {side: max(i for s, i, _ in jobs if s == side) for side in {s for s, _, _ in jobs}}
jobs.sort(key=lambda j: (j[1] != last[j[0]], -j[1]))
if "--last" in sys.argv:
    jobs = [j for j in jobs if j[1] == last[j[0]]]
for side, i, r in jobs:
    tree = f"{H}/acc/trees/{side}_{i:02d}"
    if not os.path.exists(tree):
        subprocess.run(["python3", "-I", f"{H}/tools/mktree.py", side, r["sha"], tree], check=True, stdout=subprocess.DEVNULL)
    label = f"{side} #{i}: {r['qps']:.0f} img/s: {r['note']}"
    for kind in what:
        out = f"{H}/acc/out/{kind}_{side}_{i:02d}.json"
        if os.path.exists(out):
            continue
        env = dict(os.environ, HILL_CACHE=f"{H}/acc/cache_{side}", HILL_TIMEOUT="1500")
        with open(out.replace(".json", ".log"), "w") as log:
            code = subprocess.run([f"{H}/acc/gpu", "python", "-I", f"{H}/acc/{kind}.py", "--tree", tree, "--config", f"{tree}/hill_config.py",
                                   "--out", out, "--label", label], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        print(f"{kind} {side} #{i:02d} ({r['qps']:.0f} img/s): {'done' if os.path.exists(out) else f'FAILED rc {code}'}", flush=True)
print("VALIDATION DONE", flush=True)
