"""Loads the TorchBench results of the cfg_compile stack and derives what the page shows."""
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
    """Peak memory of the cfg_compile process before torch.compile joins it, over torch.compile's process."""
    out = []
    for n, d in models.items():
        if ok3(d) and "process_peak_rss_mb" in d["cfg"] and "peak_rss_mb" in d["compile"]:
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


FINAL = load("final_r21f.jsonl")
VERSIONS = [
    ("v1", "Nov 14", load("final_r18.jsonl")),
    ("v2", "Nov 14", load("final_r19.jsonl", "rerun_r19.jsonl", "rerun2_r19.jsonl")),
    ("v3", "Nov 15", load("final_r20.jsonl")),
    ("v4", "Nov 16", FINAL),
]
