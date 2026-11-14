"""Load every finished run into plain Python structures for the page generator."""
import collections, glob, json, os, re
import numpy as np
import mathgen as mg
from subtypes import subtype

RUNS = os.environ.get("MATHLLM_RUNS", "runs/main")
R, LOG, OK = {}, {}, {}
for f in sorted(glob.glob(f"{RUNS}/*.json")):
    name = os.path.basename(f)[:-5]
    if name == "probes":
        continue
    R[name] = json.load(open(f))
    LOG[name] = [json.loads(l) for l in open(f"{RUNS}/{name}.jsonl")]
    # answer-level correctness: scratchpad runs get it from probes.py (their teacher-forced file also grades the scratchpad)
    for suffix in (".ansok.npy", ".ok.npy"):
        if os.path.exists(f"{RUNS}/{name}{suffix}"):
            OK[name] = np.load(f"{RUNS}/{name}{suffix}")
            break
PROBES = json.load(open(f"{RUNS}/probes.json")) if os.path.exists(f"{RUNS}/probes.json") else {}

D = np.load("data.npz")
TEST_LV = D["test_level"]
EXPRS = [mg.decode(r)[:-1] for r in D["test_P"]]
ANSWERS = [mg.decode(r[: list(r).index(mg.EOS)]) for r in D["test_A"]]
KEYS = [(int(l), subtype(int(l), e)) for l, e in zip(TEST_LV, EXPRS)]
SUBKEYS = sorted(set(KEYS))
SUB_N = collections.Counter(KEYS)
SUB_IDX = {k: np.array([kk == k for kk in KEYS]) for k in SUBKEYS}


def seeds(prefix):
    """Run names for a variant, e.g. 'cont8' -> ['cont8_s0', 'cont8_s1']."""
    return sorted(n for n in R if re.fullmatch(re.escape(prefix) + r"_s\d", n))


def has(prefix):
    return bool(seeds(prefix))


def em(prefix):
    """(mean over seeds, per-seed list) of the level-averaged exact match."""
    v = [float(np.mean(R[n]["final"]["em_gen"])) for n in seeds(prefix)]
    return float(np.mean(v)), v


def levels(prefix):
    return np.mean([R[n]["final"]["em_gen"] for n in seeds(prefix)], 0).tolist()


def close(prefix):
    return float(np.mean([np.mean(R[n]["final"]["close"]) for n in seeds(prefix)]))


def tle(prefix):
    return float(np.mean([R[n]["final"]["tle"] for n in seeds(prefix)]))


def sub(prefix):
    """Per-sub-skill exact match, averaged over seeds."""
    return np.mean([[OK[n][SUB_IDX[k]].mean() for k in SUBKEYS] for n in seeds(prefix) if n in OK], 0).tolist()


def sub_one(prefix, level, name):
    return sub(prefix)[SUBKEYS.index((level, name))]


def curve(name):
    return [r["step"] for r in LOG[name]], [r["em"] for r in LOG[name]]
