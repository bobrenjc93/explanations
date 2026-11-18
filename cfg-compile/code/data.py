"""Loads the TorchBench results of the torch.rewrite stack and derives what the page shows."""
import collections
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Next to this file while the page is being written; one level up in the published layout (code/ beside results/).
RESULTS = next(d for d in (os.path.join(HERE, "results"), os.path.join(HERE, "..", "results")) if os.path.isdir(d))
TOL = 1e-2


def load(*paths):
    """Rows of one or more result files; a later file replaces a model for which it has all three compilers."""
    by = {"eval": collections.defaultdict(dict), "train": collections.defaultdict(dict)}
    for k, path in enumerate(paths):
        again = {"eval": collections.defaultdict(dict), "train": collections.defaultdict(dict)}
        for line in open(os.path.join(RESULTS, path)):
            r = json.loads(line)
            again[r["test"]][r["name"]][r["mode"]] = r
        for test, models in again.items():
            for name, modes in models.items():
                if k == 0 or set(modes) == {"eager", "compile", "cfg"}:
                    by[test][name] = modes
    return by


def geo(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def ok3(d):
    return all(m in d and d[m]["ok"] for m in ("eager", "compile", "cfg"))


def turns(d):
    c = d.get("cfg")
    if c and c.get("ok") and "paired_cfg_s" in c:
        return c["paired_cfg_s"] / c["paired_compile_s"]
    return None


def summary(models):
    names = sorted(n for n, d in models.items() if ok3(d))
    eager_ran = sum(1 for d in models.values() if d.get("eager", {}).get("ok"))
    alone = {n: models[n]["cfg"]["call_s"] / models[n]["compile"]["call_s"] for n in names}
    paired = {n: turns(models[n]) for n in names if turns(models[n]) is not None}
    speed = {m: geo(models[n]["eager"]["call_s"] / models[n][m]["call_s"] for n in names) for m in ("compile", "cfg")}
    out = {
        "names": names,
        "eager_ran": eager_ran,
        "alone": alone,
        "paired": paired,
        "alone_geo": geo(alone.values()),
        "paired_geo": geo(paired.values()),
        "speed": speed,
        "compile_geo": {m: geo(max(models[n][m].get("compile_s", 0.0), 0.05) for n in names) for m in ("compile", "cfg")},
        "compile_sum": {m: sum(models[n][m].get("compile_s", 0.0) for n in names) for m in ("compile", "cfg")},
        "warm_geo": {m: geo(max(models[n][m]["warmup_s"], 0.05) for n in names) for m in ("compile", "cfg")},
        "one_graph": {m: sum(models[n][m].get("graphs") == 1 for n in names) for m in ("compile", "cfg")},
        "slower_than_eager": {m: sorted(n for n in names if models[n]["eager"]["call_s"] / models[n][m]["call_s"] < 0.95) for m in ("compile", "cfg")},
        "within": {m: sum(models[n][m].get("difference", 0.0) <= TOL for n in names) for m in ("compile", "cfg")},
        "within5": {m: sum(models[n][m].get("difference", 0.0) <= 1e-5 for n in names) for m in ("compile", "cfg")},
        "exact": {m: sum(models[n][m].get("difference", 0.0) == 0.0 for n in names) for m in ("compile", "cfg")},
        "over": {m: sorted(((models[n][m].get("difference", 0.0), n) for n in names if models[n][m].get("difference", 0.0) > TOL), reverse=True) for m in ("compile", "cfg")},
        "worst": {m: max((models[n][m].get("difference", 0.0), n) for n in names) for m in ("compile", "cfg")},
        "ahead": sorted(n for n in names if alone[n] < 0.97),
        "behind": sorted(n for n in names if alone[n] > 1.03),
        "level": sorted(n for n in names if 0.97 <= alone[n] <= 1.03),
        "failed": sorted((n, m, d[m].get("error", "")) for n, d in models.items() for m in d if not d[m]["ok"]),
    }
    return out


def held(models):
    """Peak memory of the torch.rewrite process before torch.compile joins it, over torch.compile's process."""
    out = []
    for n, d in models.items():
        if ok3(d) and "process_peak_rss_mb" in d["cfg"] and d["compile"].get("peak_rss_mb") and d["eager"].get("peak_rss_mb"):
            out.append((d["cfg"]["peak_rss_mb"] / d["compile"]["peak_rss_mb"], n, d["cfg"]["peak_rss_mb"] / 1024, d["compile"]["peak_rss_mb"] / 1024,
                        d["eager"].get("peak_rss_mb", 0) / 1024))
    return sorted(out)


def ms(seconds):
    v = seconds * 1e3
    if v >= 10000:
        return f"{v / 1e3:.1f} s"
    if v >= 1000:
        return f"{v / 1e3:.2f} s"
    if v >= 100:
        return f"{v:.0f} ms"
    if v >= 10:
        return f"{v:.1f} ms"
    return f"{v:.2f} ms"


FINAL = load("final_r22.jsonl")
V4 = load("final_r21f.jsonl")
VERSIONS = [
    ("v1", "Nov 14", load("final_r18.jsonl")),
    ("v2", "Nov 14", load("final_r19.jsonl", "rerun_r19.jsonl", "rerun2_r19.jsonl")),
    ("v3", "Nov 15", load("final_r20.jsonl")),
    ("v4", "Nov 16", V4),
    ("v5", "Nov 17", FINAL),
]


def gpu(path):
    """A GPU run: per test, the models where eager, torch.compile and torch.rewrite all ran."""
    by = {}
    for line in open(os.path.join(RESULTS, path)):
        r = json.loads(line)
        by.setdefault((r["test"], r["name"]), {})[r["mode"]] = r
    out = {"eval": [], "train": []}
    for (test, name), d in sorted(by.items()):
        if all(m in d and d[m].get("ok") for m in ("eager", "compile", "cfg")):
            e, c, g = (d[m]["call_s"] * 1e3 for m in ("eager", "compile", "cfg"))
            held = [d[m].get("peak_cuda_mb") or 0 for m in ("eager", "compile", "cfg")]
            out[test].append({"name": name, "eager": e, "compile": c, "rewrite": g, "ratio": g / c, "held": held})
    return out


GPU = gpu("gpu_r23.jsonl")
# The same harness on the seventh version (inference; its training rows are the sixth version's). Like every run of
# that harness on a GPU, on a build of PyTorch without cuDNN.
GPU7 = gpu("gpu_r24.jsonl")


def suites(folder):
    """The perf dashboard's runner (benchmarks/dynamo) on one GPU: (suite, configuration) -> model -> row of its CSV.
    A row has the speedup over eager, the compiled latency, the compile time and the peak GPU memory."""
    import csv
    import glob

    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, folder, "*.csv"))):
        config, suite = os.path.basename(path)[:-4].split("_", 1)
        with open(path, newline="") as f:
            out[(suite, config)] = {r["name"]: r for r in csv.DictReader(f) if float(r["speedup"]) > 0}
    return out


# Model by model, the compilers one right after the other (the machine is shared), on the build that keeps the freed
# chunks of the interpreter's frame stack and with the runner warmed up as its timed loop calls.
SUITES = suites("suites_r25e")
SUITES_PAIRED = SUITES
SUITES_TRAIN = suites("suites_r25e_train")


def chunks(folder, configs, least=100):
    """The models whose timed loop asked for at least `least` chunks of the frame stack a repetition under one of
    `configs`, with that number: from the chunks.log the runs wrote (the count after the loop's first repetition
    where it was taken apart, else the count from the loop's start over its 30 repetitions)."""
    import re

    found = {}
    with open(os.path.join(RESULTS, folder, "chunks.log")) as f:
        for line in f:
            if "keep 1" in line.split("|")[0]:
                continue
            config = "rewrite" if "--rewrite" in line else "freezing" if "--freezing" in line else "inductor"
            steady = re.search(r"(-?\d+) after its first repetition", line)
            per = int(steady.group(1)) / 29 if steady else int(line.split()[0]) / 30
            name = re.search(r"--only (\S+)", line).group(1)
            if config in configs and per >= least:
                found[name] = max(found.get(name, 0), per)
    return ", ".join(f"{name} ({per:.0f})" for name, per in sorted(found.items(), key=lambda kv: -kv[1])) or "none"


def kept(table, base):
    """The time of torch.rewrite over torch.compile's (geometric mean, and over how many models), with the models
    that were run a second time with the chunks kept for torch.compile's process too taken from that second pair."""
    ratios, again = [], 0
    for (suite, config), rows in table.items():
        if config != base:
            continue
        rewrite, plain2, rewrite2 = table.get((suite, "rewrite"), {}), table.get((suite, base + "-kept"), {}), table.get((suite, "rewrite-kept"), {})
        for name, row in rows.items():
            if name in rewrite:
                if name in plain2 and name in rewrite2:
                    ratios.append(float(plain2[name]["speedup"]) / float(rewrite2[name]["speedup"]))
                    again += 1
                else:
                    ratios.append(float(row["speedup"]) / float(rewrite[name]["speedup"]))
    return geo(ratios), len(ratios), again
GPU_FIRST = gpu("gpu_first.jsonl")
