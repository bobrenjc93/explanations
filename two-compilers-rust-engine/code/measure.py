#!/usr/bin/env python3
"""measure.py SIDE [--official NOTE] [--probes]: the throughput of SIDE's (tc or vm) current state.

Without --official: one process of bench.py, nothing recorded as a result (a quick look; logged as "quick").
With --official NOTE: the state is snapshotted, bench.py is run in three processes (the last with the probes),
and a row goes into SIDE/log.jsonl: the median of the three, whether the results were right, which files
differ from the start. The official rows are the result of the experiment. Holds the GPU lock while it runs.
Not to be changed by either side."""
import argparse, fcntl, json, os, statistics, subprocess, sys, time

H = "/home/dev/local/hill2"
ENV = "/home/dev/local/g/pytorch-env/bin"
parser = argparse.ArgumentParser()
parser.add_argument("side", choices=("tc", "vm"))
parser.add_argument("--official", metavar="NOTE")
parser.add_argument("--probes", action="store_true")
args = parser.parse_args()
side, D = args.side, os.path.join(H, args.side)
now = time.time()
deadline = None
if os.path.exists(os.path.join(D, "deadline")):
    deadline = float(open(os.path.join(D, "deadline")).read())
if args.official and deadline and now > deadline:
    sys.exit("measure: the deadline has passed; no more official measurements.")

env = dict(os.environ)
env["PATH"] = f"{ENV}:{os.path.expanduser('~/.cargo/bin')}:{env['PATH']}"
env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(D, "cache")
env.pop("PYTHONPATH", None)
env.pop("CUDA_VISIBLE_DEVICES", None)
env["PYTHONPATH"] = os.path.join(H, side, "boot")


def git(repo, *command):
    return subprocess.run(["git", "-C", repo, "-c", "user.name=hill", "-c", "user.email=hill@localhost", *command], capture_output=True, text=True)


def snapshot(note):
    """The state that is measured, as a commit: the side's tree (the Python of torch; for vm also the engine's
    Rust crate in tree/rust and the built library in tree/torch), with its config.py."""
    repo = os.path.join(H, side, "tree")
    subprocess.run(["cp", os.path.join(D, "config.py"), os.path.join(repo, "hill_config.py")], check=True)
    if not os.path.isdir(os.path.join(repo, ".git")):
        git(repo, "init", "-q")
        with open(os.path.join(repo, ".gitignore"), "a") as f:
            f.write("__pycache__/\n*.pyc\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", note)
    sha = git(repo, "rev-parse", "HEAD").stdout.strip()
    first = git(repo, "rev-list", "--max-parents=0", "HEAD").stdout.split()[0]
    changed = [f for f in git(repo, "diff", "--name-only", first, "HEAD").stdout.split("\n") if f]
    return sha, changed


lock = open(os.path.join(H, "gpu.lock"), "w")
t0 = time.time()
fcntl.flock(lock, fcntl.LOCK_EX)
waited = time.time() - t0
sha, changed = snapshot(args.official) if args.official else (None, None)
stamp = time.strftime("%m%d_%H%M%S", time.gmtime())
runs, n = [], 3 if args.official else 1
t1 = time.time()
for i in range(n):
    out = os.path.join(D, "out", f"{stamp}_{i}.json")
    command = ["taskset", "-c", "0-11", "python3", "-I", "/home/dev/local/cfgloop/guard/safe_run.py", "64",
               f"{ENV}/python", os.path.join(H, "bench.py"), "--config", os.path.join(D, "config.py"), "--out", out]
    if args.probes or (args.official and i == n - 1):
        command.append("--probes")
    with open(out.replace(".json", ".log"), "w") as log:
        try:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1500, cwd=D).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    if os.path.exists(out):
        runs.append(json.load(open(out)))
    else:
        runs.append({"failed": code, "log": out.replace(".json", ".log")})
held = time.time() - t1
fcntl.flock(lock, fcntl.LOCK_UN)

good = [r for r in runs if "qps" in r]
row = {
    "time": time.time(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "side": side,
    "kind": "official" if args.official else "quick", "note": args.official or "",
    "qps": statistics.median(r["qps"] for r in good) if len(good) == n else None,
    "qps_runs": [r.get("qps") for r in runs],
    "eager_qps": statistics.median(r["eager_qps"] for r in good) if good else None,
    "accuracy_ok": len(good) == n and all(r["accuracy_ok"] for r in good),
    "error_over_eager_error": max((r["error_over_eager_error"] for r in good), default=None),
    "probes": next((r["probes"] for r in good if "probes" in r), None),
    "warmup_s": [r.get("warmup_s") for r in runs], "first_call_s": [r.get("first_call_s") for r in runs],
    "sha": sha, "changed": changed, "lock_wait_s": waited, "held_s": held,
    "failed": [r for r in runs if "qps" not in r],
}
probes = row["probes"] or {}
row["eligible"] = bool(
    args.official and row["accuracy_ok"] and row["qps"]
    and probes.get("right_for_another_batch_size") is True and probes.get("right_afterwards") is True
    and probes.get("right_after_weight_restored") is True
)
with open(os.path.join(D, "log.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")

print(f"{row['kind']} measurement of {side}: " + (f"{row['qps']:.0f} images/s" if row["qps"] else "FAILED"),
      f"(runs: {', '.join(f'{q:.0f}' if q else 'failed' for q in row['qps_runs'])}; eager {row['eager_qps'] or 0:.0f})")
print(f"  results right: {row['accuracy_ok']} (error against float64 over eager's: {row['error_over_eager_error']})")
if row["probes"] is not None:
    print(f"  probes: {json.dumps(row['probes'])}")
for r in row["failed"]:
    print(f"  a run failed ({r['failed']}): see {r['log']}")
if args.official:
    print(f"  counts as a result: {row['eligible']}; files changed since the start: {len(changed)}; snapshot {sha[:10]}")
    rows = [json.loads(line) for line in open(os.path.join(D, "log.jsonl"))]
    best = max((r for r in rows if r.get("eligible")), key=lambda r: r["qps"], default=None)
    if best:
        print(f"  best so far: {best['qps']:.0f} images/s ({best['note']})")
if deadline:
    print(f"  time left: {max(0, deadline - time.time()) / 60:.0f} min")
