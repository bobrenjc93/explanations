"""Training pool and test set for the 14-level benchmark (the 7 calculator levels + 7 harder tiers).

Usage: build_data3.py <rows per level> <out.npz> <version> [weights] [regime]
  version   full    '[' worked solution ']' answer   (mathgen2 v2.3 procedures, mathgen3 for the new tiers)
            values  the same solutions without the digit-level work (one 'op=value' per operation)
            none    the answer alone
  regime    train   the original size ranges (default)
            wide    the original ranges plus one size up (mathgen3.REGIME = "wide": a few sizes are left out as gaps), sin/cos of
                    arguments out to +-100 with argument reduction, solutions up to 4,096 tokens; the stretch-set problems are excluded

Rows are bucketed by prompt class (fits 32 tokens or not) and target length, so shapes are static and padding small.
The test set is the old 14,000 problems plus 2,000 per new tier, the same problems for every version; <out>.test.json
lists them (text, true answer, sub-type, length of the full worked solution).
"""
import json, random, sys, time
import numpy as np
from multiprocessing import Pool
import mathgen as mg, mathgen2 as m2, mathgen3 as m3

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
OUT = sys.argv[2] if len(sys.argv) > 2 else "data3.npz"
VERSION = sys.argv[3] if len(sys.argv) > 3 else "full"
WEIGHTS = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1.0] * m3.NLEVELS
CHUNK = 25_000
WIDTHS = [20, 32, 48, 72, 112, 176, 272, 400, 576, 832, 1216, 1728, m3.T_MAX]
PW = [32, m3.P_MAX]  # prompt classes
N_TEST = 2000
EOS = bytes([mg.EOS])
m2.configure("values6" if VERSION == "values" else "v2.3")
REGIME = sys.argv[5] if len(sys.argv) > 5 else "train"
EXCLUDE = set()
if REGIME == "wide":
    m3.REGIME, m2.OPT["reduce"], mg.SIN_RANGE = "wide", True, 100
    WIDTHS, PW = WIDTHS + [3072, 4096], [32, 128]
    EXCLUDE = set(json.load(open("hard/stretch.json"))["text"])


def enc(text):
    return text.encode("ascii").translate(m3.TABLE)


def sample(lv, rng):
    """-> (text, problem or None for an old level)"""
    if lv < m3.OLD:
        return mg.sample(lv, rng)[0], None
    p = m3.sample(lv, rng)
    return p["text"], p


def target(text, p, rng=None):
    """Token bytes of what follows the '=' for this version (raises Reject when the solution cannot be written)."""
    if VERSION == "none":
        return enc(p["truth"] if p else mg.fmt(mg.ev(mg.parse(text)))) + EOS
    sc, ans = m3.solve(p, rng) if p else m2.solve(text, rng)
    return enc("[" + sc + "]" + ans) + EOS


def work(args):
    lv, seed, n = args
    rng = random.Random(seed)
    P, T, nrej = [], [], 0
    while len(P) < n:
        text, p = sample(lv, rng)
        if (m3.held_out(p) if p else mg.is_test(text)) or text in EXCLUDE or len(text) + 2 > PW[-1]:
            continue
        try:
            t = target(text, p, rng)
        except mg.Reject:
            nrej += 1
            continue
        if len(t) > WIDTHS[-1]:
            nrej += 1
            continue
        P.append(bytes([mg.BOS]) + enc(text + "="))
        T.append(t)
    return (lv, np.array([len(x) for x in P], np.uint16), np.frombuffer(b"".join(P), np.uint8),
            np.array([len(x) for x in T], np.uint16), np.frombuffer(b"".join(T), np.uint8), nrej)


def test_tier(lv):
    """2,000 held-out problems of a new tier whose full worked solution fits; identical for every version."""
    rng = random.Random(9000 + lv)
    out, seen = [], set()
    saved, regime, sin_range = dict(m2.OPT), m3.REGIME, mg.SIN_RANGE
    m2.configure("v2.3")
    m3.REGIME, mg.SIN_RANGE = "train", 0  # the test set is the same 2,000 problems per tier whatever the pool's regime
    while len(out) < N_TEST:
        p = m3.sample(lv, rng)
        if not m3.held_out(p) or p["text"] in seen:
            continue
        seen.add(p["text"])
        try:
            sc, ans = m3.solve(p)
        except mg.Reject:
            continue
        n = len(sc) + 2 + len(ans) + 1
        if n > m3.T_MAX:
            continue
        if lv == m3.NLEVELS - 1:
            p["kind"] += f", {sc.count(';x=')} iterations"
        p["full_len"] = n
        out.append(p)
    m2.OPT.update(saved)
    m3.REGIME, mg.SIN_RANGE = regime, sin_range
    return out


def pad_fill(lens, flat, rows, width, left=False):
    """Rows of a flat token stream as a padded [len(rows), width] array (left: right-aligned, for prompts)."""
    off = np.concatenate([[0], np.cumsum(lens.astype(np.int64))])
    out = np.zeros((len(rows), width), np.uint8)
    L = lens[rows].astype(np.int64)
    col = np.arange(width)[None, :]
    if left:
        mask = col >= width - L[:, None]
        src = off[rows][:, None] + col - (width - L[:, None])
    else:
        mask = col < L[:, None]
        src = off[rows][:, None] + col
    out[mask] = flat[src[mask]]
    return out


def choose_buckets(pc, wi):
    """Populated (prompt class, width) cells; sparse cells are folded into the cheapest bucket that still fits them.
    -> (list of buckets, cell -> bucket index table)"""
    counts = np.zeros((2, len(WIDTHS)), np.int64)
    np.add.at(counts, (pc, wi), 1)
    keep = {(c, i) for c in range(2) for i in range(len(WIDTHS)) if counts[c, i]}
    cost = lambda b: PW[b[0]] + WIDTHS[b[1]]
    cover = lambda cell, pool: min((b for b in pool if b[0] >= cell[0] and b[1] >= cell[1]), key=cost, default=None)
    load = {b: counts[b] for b in keep}
    min_rows = int(0.0015 * counts.sum())
    while True:
        small = [b for b in keep if load[b] < min_rows and cover(b, keep - {b})]
        if not small:
            break
        b = min(small, key=lambda b: load[b])
        keep.discard(b)
        load[cover(b, keep)] += load.pop(b)
    buckets = sorted(keep, key=lambda b: (cost(b), b))
    table = -np.ones((2, len(WIDTHS)), np.int64)
    for c in range(2):
        for i in range(len(WIDTHS)):
            b = cover((c, i), keep)
            if b is not None:
                table[c, i] = buckets.index(b)
    return buckets, table


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(lv, 100_000 * lv + i, CHUNK) for lv in range(m3.NLEVELS) for i in range(int(N_TRAIN * WEIGHTS[lv]) // CHUNK)]
    parts = []
    with Pool(22) as pool:
        for r in pool.imap_unordered(work, jobs, chunksize=1):
            parts.append(r)
        new_test = pool.map(test_tier, range(m3.OLD, m3.NLEVELS))
    print("generated in %.0fs, rejected %d" % (time.time() - t0, sum(p[5] for p in parts)), flush=True)
    plens = np.concatenate([p[1] for p in parts]); pflat = np.concatenate([p[2] for p in parts])
    tlens = np.concatenate([p[3] for p in parts]); tflat = np.concatenate([p[4] for p in parts])
    lv = np.concatenate([np.full(len(p[1]), p[0], np.uint8) for p in parts])
    del parts

    # ---- test set: the old 14k problems, then 2,000 per new tier
    Dz = np.load("data.npz")
    texts = [mg.decode(r)[:-1] for r in Dz["test_P"]]
    truth = [mg.decode(r[: list(r).index(mg.EOS)]) for r in Dz["test_A"]]
    t_level = [int(x) for x in Dz["test_level"]]
    kinds, full_len, probs = [], [], [None] * len(texts)
    try:
        from subtypes import subtype
    except ImportError:
        subtype = lambda l, e: ""
    saved = dict(m2.OPT)
    m2.configure("v2.3")
    for l, e in zip(t_level, texts):
        kinds.append(subtype(l, e))
        try:
            sc, ans = m2.solve(e)
            full_len.append(len(sc) + 2 + len(ans) + 1)
        except mg.Reject:
            full_len.append(3)
    m2.OPT.update(saved)
    for tier in new_test:
        for p in tier:
            texts.append(p["text"]); truth.append(p["truth"]); t_level.append(p["level"]); kinds.append(p["kind"])
            full_len.append(p["full_len"]); probs.append(p)
    tP, tT = [], []
    for e, p in zip(texts, probs):
        try:
            t = target(e, p)
        except mg.Reject:
            t = enc("[]") + EOS
        tP.append(bytes([mg.BOS]) + enc(e + "=")); tT.append(t[: WIDTHS[-1]])
    tplens = np.array([len(x) for x in tP], np.uint16); tpflat = np.frombuffer(b"".join(tP), np.uint8)
    ttlens = np.array([len(x) for x in tT], np.uint16); ttflat = np.frombuffer(b"".join(tT), np.uint8)

    # ---- buckets
    cell = lambda pl, tl: ((pl > PW[0]).astype(np.int64), np.searchsorted(WIDTHS, tl))
    buckets, table = choose_buckets(*cell(plens, tlens))
    bidx = table[cell(plens, tlens)]
    tb = table[cell(tplens, ttlens)]
    for i in np.where(tb < 0)[0]:  # a test row with no fitting bucket of its own class: the cheapest bucket that holds it
        fit = [k for k, (c, w) in enumerate(buckets) if PW[c] >= tplens[i] and WIDTHS[w] >= ttlens[i]]
        tb[i] = fit[0]
    assert (bidx >= 0).all() and (tb >= 0).all()
    save = {"widths": np.array([WIDTHS[w] for _, w in buckets])}
    pw = []
    for k, (c, w) in enumerate(buckets):
        rows, trows = np.where(bidx == k)[0], np.where(tb == k)[0]
        longest = max(int(plens[rows].max()) if len(rows) else 0, int(tplens[trows].max()) if len(trows) else 0)
        width = PW[0] if c == 0 else min(PW[1], -(-longest // 8) * 8)
        pw.append(width)
        save[f"P{k}"], save[f"T{k}"], save[f"L{k}"] = pad_fill(plens, pflat, rows, width, True), pad_fill(tlens, tflat, rows, WIDTHS[w]), lv[rows]
        save[f"tP{k}"], save[f"tT{k}"], save[f"tI{k}"] = pad_fill(tplens, tpflat, trows, width, True), pad_fill(ttlens, ttflat, trows, WIDTHS[w]), trows
        print(f"bucket {k:2d} prompt {width:3d} target {WIDTHS[w]:4d}: {len(rows):9d} rows  mean len {tlens[rows].mean() if len(rows) else 0:7.1f}  "
              f"test {len(trows):5d}  levels {np.bincount(lv[rows], minlength=m3.NLEVELS).tolist()}", flush=True)
    save["pw"] = np.array(pw)
    A = np.zeros((len(truth), m3.A_MAX), np.uint8)
    for i, a in enumerate(truth):
        t = enc(a) + EOS
        A[i, : len(t)] = np.frombuffer(t, np.uint8)
    save["test_A"], save["test_level"] = A, np.array(t_level, np.uint8)
    per_level = [float(tlens[lv == l].mean()) for l in range(m3.NLEVELS)]
    print("mean target tokens per level:", " ".join(f"{x:.0f}" for x in per_level), " all: %.1f" % tlens.mean(), flush=True)
    np.savez(OUT, **save)
    json.dump({"text": texts, "truth": truth, "kind": kinds, "level": t_level, "full_len": full_len, "version": VERSION,
               "train_tokens_per_level": per_level, "train_rows": int(len(tlens)), "train_tokens_mean": float(tlens.mean())},
              open(OUT.replace(".npz", ".test.json"), "w"))
    print("wrote", OUT, "%.0fs" % (time.time() - t0))
