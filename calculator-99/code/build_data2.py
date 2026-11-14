"""Training pool and test set with v2 scratchpads, bucketed by target length (static shapes, little padding)."""
import random, sys, time
import numpy as np
from multiprocessing import Pool
import mathgen as mg, mathgen2 as m2

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
CHUNK = 50_000
WIDTHS = [20, 32, 48, 72, 112, 176, 272, 400, 576]
OUT = sys.argv[2] if len(sys.argv) > 2 else "data2.npz"
# rows per level, in units of N_TRAIN: the hard levels have more to memorise and get more of the pool
WEIGHTS = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1.0] * mg.NLEVELS


def prompt_row(e):
    p = [mg.BOS] + mg.encode(e) + [mg.STOI["="]]
    return [mg.PAD] * (mg.P_MAX - len(p)) + p


def target(e, rng=None):
    sc, ans = m2.solve(e, rng)
    return m2.encode("[" + sc + "]" + ans) + [mg.EOS]


def work(args):
    lv, seed, n = args
    rng = random.Random(seed)
    P, T, nrej = [], [], 0
    while len(P) < n:
        e, _, _ = mg.sample(lv, rng)
        if mg.is_test(e):
            continue
        try:
            t = target(e, rng)
        except mg.Reject:
            nrej += 1
            continue
        if len(t) > WIDTHS[-1]:
            nrej += 1
            continue
        P.append(prompt_row(e))
        T.append(t)
    lens = np.array([len(t) for t in T], np.uint16)
    return lv, np.array(P, np.uint8), lens, np.fromiter((x for t in T for x in t), np.uint8, int(lens.sum())), nrej


def bucketize(P, lens, flat, extra):
    """-> per-bucket (P, T padded, extra) arrays"""
    off = np.concatenate([[0], np.cumsum(lens.astype(np.int64))])
    b = np.searchsorted(WIDTHS, lens)
    out = []
    for i, w in enumerate(WIDTHS):
        idx = np.where(b == i)[0]
        T = np.zeros((len(idx), w), np.uint8)
        # vectorised fill: position j of row r comes from flat[off[idx[r]] + j] when j < len
        L = lens[idx].astype(np.int64)
        col = np.arange(w)[None, :]
        src = off[idx][:, None] + col
        mask = col < L[:, None]
        T[mask] = flat[src[mask]]
        out.append((P[idx], T, extra[idx]))
    return out


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(lv, 5000 * lv + i, CHUNK) for lv in range(mg.NLEVELS) for i in range(int(N_TRAIN * WEIGHTS[lv]) // CHUNK)]
    parts = []
    with Pool(26) as pool:
        for r in pool.imap_unordered(work, jobs):
            parts.append(r)
    print("generated in %.0fs, rejected %d" % (time.time() - t0, sum(p[4] for p in parts)), flush=True)
    P = np.concatenate([p[1] for p in parts]); lens = np.concatenate([p[2] for p in parts]); flat = np.concatenate([p[3] for p in parts])
    lv = np.concatenate([np.full(len(p[1]), p[0], np.uint8) for p in parts])
    save = {"widths": np.array(WIDTHS)}
    for i, (Pb, Tb, Lb) in enumerate(bucketize(P, lens, flat, lv)):
        save[f"P{i}"], save[f"T{i}"], save[f"L{i}"] = Pb, Tb, Lb
        print(f"train bucket {i} width {WIDTHS[i]:3d}: {len(Pb):9d} rows  mean len {(Tb != 0).sum(1).mean():.1f}  levels {np.bincount(Lb, minlength=7)}", flush=True)
    del P, lens, flat, parts
    # test set: same 14k problems and true answers as v1
    Dz = np.load("data.npz")
    tP = Dz["test_P"]
    T, drop = [], 0
    for r in tP:
        e = mg.decode(r)[:-1]
        try:
            t = target(e)
        except mg.Reject:
            t = m2.encode("[]") + [mg.EOS]; drop += 1
        T.append(t[: WIDTHS[-1]])
    lens = np.array([len(t) for t in T], np.uint16)
    flat = np.fromiter((x for t in T for x in t), np.uint8, int(lens.sum()))
    for i, (Pb, Tb, Ib) in enumerate(bucketize(tP, lens, flat, np.arange(len(tP)))):
        save[f"tP{i}"], save[f"tT{i}"], save[f"tI{i}"] = Pb, Tb, Ib
    save["test_A"], save["test_level"] = Dz["test_A"], Dz["test_level"]
    print("test rows without a trace:", drop, " test lens max", lens.max(), flush=True)
    np.savez(OUT, **save)
    print("wrote", OUT, "%.0fs" % (time.time() - t0))
