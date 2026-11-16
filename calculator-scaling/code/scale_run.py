"""Run the scaling study: two queues side by side on one GPU, every job resumable. Re-running this script after an interruption
continues where it stopped (finished jobs are skipped, a half-trained run resumes from its last checkpoint).
Usage: scale_run.py"""
import os, subprocess, threading, time

os.chdir(os.path.expanduser("~/local/mathllm"))
OUT = "runs/scale"
BASE = ["--data", "hard/full.npz", "--out", OUT, "--ckpt_every", "500", "--gen_every", "5000", "--eval_every", "5000"]
N19 = ["--d", "512", "--L", "6", "--heads", "8", "--lo", "2", "--hi", "4", "--accum", "2"]
N08, N18, N02 = ["--d", "128", "--heads", "4"], ["--d", "192", "--heads", "4"], ["--d", "64", "--heads", "2"]
QUEUES = [
    [("S_n19", N19), ("S_n08_30k", N08 + ["--steps", "30000"])],  # the long ones
    [("S_n08", N08), ("S_n18", N18), ("S_n02", N02), ("S_t5k", ["--steps", "5000"]), ("S_t2k5", ["--steps", "2500"]),
     ("S_d560k", ["--data_frac", "0.02"]), ("S_d5m6", ["--data_frac", "0.2"]), ("S_d056k", ["--data_frac", "0.002"])],
]
ENV = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=os.path.abspath(".inductor_cache"), CUDA_MPS_PIPE_DIRECTORY=os.path.abspath("mps/pipe"),
           CUDA_MPS_LOG_DIRECTORY=os.path.abspath("mps/log"))


def sh(cmd, log):
    with open(log, "a") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV).returncode


def worker(q, jobs):
    for name, extra in jobs:
        if not os.path.exists(f"{OUT}/{name}.json"):
            print(time.strftime("%H:%M:%S"), f"queue {q}: train {name}", flush=True)
            rc = sh([".venv/bin/python", "train3.py", "--name", name] + BASE + extra, f"logs/{name}.log")
            print(time.strftime("%H:%M:%S"), f"queue {q}: {name} rc {rc}", flush=True)
            if rc:
                continue
        if not os.path.exists(f"{OUT}/{name}.stretch100.json"):
            sh([".venv/bin/python", "eval_stretch.py", f"{OUT}/{name}.pt", "hard/stretch.json", "stretch100", "100"], f"logs/stretch100_{name}.log")
            print(time.strftime("%H:%M:%S"), f"queue {q}: {name} stretch done", flush=True)


if not any(c.strip() == "nvidia-cuda-mps" for c in os.popen("ps -eo comm").read().split("\n")):
    os.makedirs("mps/pipe", exist_ok=True); os.makedirs("mps/log", exist_ok=True)
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=ENV)
    time.sleep(2)
threads = [threading.Thread(target=worker, args=(i, jobs)) for i, jobs in enumerate(QUEUES)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(time.strftime("%H:%M:%S"), "ALL DONE", flush=True)
