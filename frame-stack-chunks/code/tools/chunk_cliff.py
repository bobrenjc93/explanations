"""chunk_cliff.py: plain Python, no PyTorch. CPython (3.11 and later) keeps the frames of a thread in chunks of 16 KB
and gives a chunk back to the system when the frame at its start returns. A call whose frame is the first that does
not fit in the current chunk therefore maps a chunk and unmaps it, every time it is made. This calls a small function
2000 times from each of 200 depths of the stack and reports, per depth, the page faults of the process and the time of
a call: at the depths where the callee's frame is the one that does not fit, each call is a page fault."""
import json
import resource
import statistics
import sys
import time


def callee(a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0):
    return a


def caller(n):
    for _ in range(n):
        callee()


def deeper(depth, n):
    return caller(n) if depth == 0 else deeper(depth - 1, n)


def measure(depth, n=2000):
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    start = time.perf_counter()
    deeper(depth, n)
    took = time.perf_counter() - start
    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults, took / n * 1e9


for depth in range(200):  # once to warm the interpreter's caches of the three functions up
    measure(depth, 200)
rows = [measure(depth) for depth in range(200)]
slow = [i for i, (faults, _) in enumerate(rows) if faults >= 1000]
usual = statistics.median(ns for faults, ns in rows if faults < 1000)
print(json.dumps({"python": sys.version.split()[0], "depths": len(rows), "depths_with_a_fault_per_call": slow,
                  "ns_a_call_usually": round(usual, 1), "ns_a_call_there": round(statistics.median(rows[i][1] for i in slow), 1) if slow else None,
                  "faults": [f for f, _ in rows], "ns": [round(ns, 1) for _, ns in rows]}))
