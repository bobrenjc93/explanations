"""Run the jobs in a file (one per line: name + train.py args), at most N at a time, skipping finished ones."""
import os, subprocess, sys, time

jobs_file, out, n_par = sys.argv[1], sys.argv[2], int(sys.argv[3])
extra = sys.argv[4:]
os.makedirs(out, exist_ok=True)
os.makedirs("logs", exist_ok=True)
running = {}
started = set()
while True:
    jobs = [l.split() for l in open(jobs_file) if l.strip() and not l.startswith("#")]
    for name in [n for n, p in running.items() if p.poll() is not None]:
        print(time.strftime("%H:%M:%S"), "done", name, "rc", running.pop(name).returncode, flush=True)
    todo = [j for j in jobs if j[0] not in started and not os.path.exists(f"{out}/{j[0]}.json")]
    while todo and len(running) < n_par:
        j = todo.pop(0)
        cmd = [".venv/bin/python", os.environ.get("SCHED_SCRIPT", "train.py"), "--name", j[0], "--out", out] + extra + j[1:]
        running[j[0]] = subprocess.Popen(cmd, stdout=open(f"logs/{j[0]}.log", "w"), stderr=subprocess.STDOUT)
        started.add(j[0])
        print(time.strftime("%H:%M:%S"), "start", " ".join(j), flush=True)
    if not running and not todo:
        break
    time.sleep(5)
print("ALL DONE", flush=True)
