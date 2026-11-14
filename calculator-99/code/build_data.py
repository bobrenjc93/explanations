"""Pre-generate the train pool and held-out test set (disjoint by crc32 of the expression)."""
import sys, random, time
import numpy as np
from multiprocessing import Pool
import mathgen as mg

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000
N_TEST = 2000
CHUNK = 50_000


def enc_row(e, a, sc):
    p = [mg.BOS] + mg.encode(e) + [mg.STOI["="]]
    p = [mg.PAD] * (mg.P_MAX - len(p)) + p
    ans = mg.encode(a) + [mg.EOS]
    cot = mg.encode("[" + ",".join(sc) + "]" + a) + [mg.EOS]
    return p, ans + [mg.PAD] * (mg.A_MAX - len(ans)), cot + [mg.PAD] * (mg.C_MAX - len(cot))


def work(args):
    lv, seed, n, want_test = args
    rng = random.Random(seed)
    P, A, C, seen = [], [], [], set()
    while len(P) < n:
        e, a, sc = mg.sample(lv, rng)
        if mg.is_test(e) != want_test or (want_test and e in seen):
            continue
        seen.add(e)
        p, a_, c = enc_row(e, a, sc)
        P.append(p); A.append(a_); C.append(c)
    return lv, want_test, np.array(P, np.uint8), np.array(A, np.uint8), np.array(C, np.uint8)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(lv, 1000 * lv + i, CHUNK, False) for lv in range(mg.NLEVELS) for i in range(N_TRAIN // CHUNK)]
    jobs += [(lv, 999_000 + lv, N_TEST, True) for lv in range(mg.NLEVELS)]
    out = {}
    with Pool(28) as pool:
        for lv, tst, P, A, C in pool.imap_unordered(work, jobs):
            out.setdefault((lv, tst), []).append((P, A, C))
    save = {}
    for tst, name in ((False, "train"), (True, "test")):
        for j, key in enumerate("PAC"):
            save[f"{name}_{key}"] = np.concatenate([np.concatenate([x[j] for x in out[(lv, tst)]]) for lv in range(mg.NLEVELS)])
        save[f"{name}_level"] = np.concatenate([np.full(sum(len(x[0]) for x in out[(lv, tst)]), lv, np.uint8) for lv in range(mg.NLEVELS)])
    np.savez("data.npz", **save)
    print({k: v.shape for k, v in save.items()}, "%.0fs" % (time.time() - t0))
    for lv in range(mg.NLEVELS):
        i = lv * N_TEST
        print(mg.LEVELS[lv], "|", mg.decode(save["test_P"][i]), mg.decode(save["test_A"][i]), "|", mg.decode(save["test_C"][i]))
    P = save["train_P"]
    for lv in range(mg.NLEVELS):
        rows = P[lv * N_TRAIN : (lv + 1) * N_TRAIN]
        u = len(np.unique(rows[:200_000], axis=0))
        print(mg.LEVELS[lv], "unique among first 200k train rows:", u)
