"""Post-hoc probes on saved checkpoints: thinking-time sweeps, memory knockouts, thought interventions,
logit lens over the thoughts, confidence-based halting, and per-sub-skill accuracy."""
import collections, json, os, sys
import numpy as np
import torch
import mathgen as mg
from analyze import load, P, A, lv, exprs, keys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs/main"
dev = "cuda"
out = {}


def have(name):
    return os.path.exists(f"{RUNS}/{name}.pt")


@torch.no_grad()
def exact(m, **kw):
    ok = []
    for i in range(0, len(P), 3500):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ok.append(m.exact(P[i : i + 3500], A[i : i + 3500], **kw))
    return torch.cat(ok).cpu().numpy()


def summarize(ok):
    return {"mean": float(np.mean([ok[lv == l].mean() for l in range(mg.NLEVELS)])),
            "levels": [float(ok[lv == l].mean()) for l in range(mg.NLEVELS)]}


# 1. train at one K, test at another
out["ksweep"] = {}
for name in ["cont8_s0", "cont16_s0", "cont4_s0", "um8_s0", "um16_s0", "direct8_s0", "mid16_s0", "win1_s0", "disc8_s0", "soft8_s0"]:
    if have(name):
        m = load(f"{RUNS}/{name}.pt")
        out["ksweep"][name] = {K: summarize(exact(m, K=K)) for K in (0, 1, 2, 4, 6, 8, 12, 16, 24)}
        print(name, {K: round(v["mean"], 3) for K, v in out["ksweep"][name].items()}, flush=True)

# 2. is the scratchpad memory or just compute?  (a) hide old thoughts, (b) corrupt what the thoughts say
out["knockout"] = {}
for name in ["cont8_s0", "cont8_s1", "cont16_s0", "direct8_s0", "soft8_s0", "disc8_s0", "mid16_s0", "coco_cont8_s0"]:
    if have(name) and "coco" not in name:
        m = load(f"{RUNS}/{name}.pt")
        r = {"full": summarize(exact(m))}
        for w in (1, 2, 4):
            r[f"window{w}"] = summarize(exact(m, window=w))
        for iv in ("zero", "shuffle", "noise"):
            r[iv] = summarize(exact(m, intervene=iv))
        out["knockout"][name] = r
        print(name, {k: round(v["mean"], 3) for k, v in r.items()}, flush=True)

# 3. logit lens: what would each thought say if it were decoded as a token?
if have("cont8_s0"):
    m = load(f"{RUNS}/cont8_s0.pt")
    lens_tok = []
    with torch.no_grad():
        for i in range(0, len(P), 3500):
            states = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m.think(P[i : i + 3500], collect=states)
                lens_tok.append(torch.cat([m.head(m.ln_f(h)).argmax(-1) for h in states], 1))
    lens_tok = torch.cat(lens_tok).cpu().numpy()  # [N, K]
    Anp = A.cpu().numpy()
    K = lens_tok.shape[1]
    out["lens"] = {
        "match_char": [[float((lens_tok[:, k] == Anp[:, j]).mean()) for k in range(K)] for j in range(4)],
        "examples": [],
    }
    rng = np.random.RandomState(0)
    for l in range(mg.NLEVELS):
        for i in rng.choice(np.where(lv == l)[0], 3, replace=False):
            row = Anp[i].tolist()
            out["lens"]["examples"].append({"level": l, "expr": exprs[i], "answer": mg.decode(row[: row.index(mg.EOS)]),
                                            "thoughts": [mg.ITOS.get(int(t), "<eos>" if t == mg.EOS else "?") for t in lens_tok[i]]})
    # geometry: how similar are consecutive thoughts?
    with torch.no_grad():
        states = []
        m.think(P[:3500], collect=states)
        S = torch.nn.functional.normalize(torch.cat(states, 1).float(), dim=-1)  # [n, K, d]
        out["lens"]["cos"] = (S @ S.transpose(1, 2)).mean(0).cpu().tolist()
    print("lens first-char match by thought:", [round(x, 3) for x in out["lens"]["match_char"][0]], flush=True)

# 4. halting by confidence on the variable-K model
for name in ["krand_cont_s0", "krand_um_s0"]:
    if have(name):
        m = load(f"{RUNS}/{name}.pt")
        Ks = [0, 1, 2, 4, 8, 16]
        conf, corr = [], []
        truth = A.cpu().numpy()
        with torch.no_grad():
            for K in Ks:
                c, g = [], []
                for i in range(0, len(P), 3500):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        toks, logp = m.generate(P[i : i + 3500], mg.A_MAX, conf=True, K=K)
                    c.append(logp.cpu().numpy())
                    t = toks.cpu().numpy()
                    g.append(np.pad(t, ((0, 0), (0, mg.A_MAX - t.shape[1]))))
                g = np.concatenate(g)
                ok = np.array([_g[: list(_t).index(mg.EOS) + 1].tolist() == _t[: list(_t).index(mg.EOS) + 1].tolist() for _g, _t in zip(g, truth)])
                conf.append(np.concatenate(c))
                corr.append(ok)
        conf, corr = np.array(conf), np.array(corr)  # [nK, N]
        fixed = [{"K": K, "acc": float(np.mean([corr[i][lv == l].mean() for l in range(mg.NLEVELS)]))} for i, K in enumerate(Ks)]
        curve = []
        for tau in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.1]:
            stop = np.full(conf.shape[1], len(Ks) - 1)
            for i in reversed(range(len(Ks))):
                stop = np.where(np.exp(conf[i]) >= tau, i, stop)
            okk = corr[stop, np.arange(conf.shape[1])]
            used = np.array(Ks)[stop]
            # thoughts actually spent: the policy has to think up to K and try decoding at each rung on the way
            curve.append({"tau": tau, "acc": float(np.mean([okk[lv == l].mean() for l in range(mg.NLEVELS)])),
                          "mean_K": float(np.mean([used[lv == l].mean() for l in range(mg.NLEVELS)])),
                          "K_by_level": [float(used[lv == l].mean()) for l in range(mg.NLEVELS)]})
        oracle = corr.any(0)
        first = np.where(corr.any(0), corr.argmax(0), len(Ks) - 1)
        out.setdefault("halting", {})[name] = {
            "fixed": fixed, "curve": curve,
            "oracle_acc": float(np.mean([oracle[lv == l].mean() for l in range(mg.NLEVELS)])),
            "oracle_mean_K": float(np.mean([np.array(Ks)[first][lv == l].mean() for l in range(mg.NLEVELS)])),
            "acc_by_level_K": [[float(corr[i][lv == l].mean()) for i in range(len(Ks))] for l in range(mg.NLEVELS)],
        }
        print(name, "fixed", [(f["K"], round(f["acc"], 3)) for f in fixed], "oracle", round(out["halting"][name]["oracle_acc"], 3), flush=True)

# 4b. how wrong are the wrong answers?  relative error of the printed number
def rel_errors(m, **kw):
    preds = []
    with torch.no_grad():
        for i in range(0, len(P), 3500):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = m.generate(P[i : i + 3500], mg.A_MAX, **kw).cpu().tolist()
            for row in toks:
                row = row[: row.index(mg.EOS)] if mg.EOS in row else row
                preds.append(mg.decode(row))
    truth = [mg.decode(r[: r.index(mg.EOS)]) for r in A.cpu().tolist()]
    err = []
    for a, b in zip(preds, truth):
        try:
            err.append(abs(float(a) - float(b)) / max(abs(float(b)), 1e-9))
        except ValueError:
            err.append(float("inf"))
    return np.array(err)


out["relerr"] = {}
for name in ["none_s0", "none_s1", "none_s2", "um8_s0", "disc8_s0", "disc8_s1", "soft8_s0", "cont8_s0", "cont8_s1", "cont8_s2", "cont16_s0", "direct8_s0"]:
    if have(name):
        err = rel_errors(load(f"{RUNS}/{name}.pt"))
        out["relerr"][name] = {str(t): [float((err[lv == l] <= t).mean()) for l in range(mg.NLEVELS)] for t in (0.0, 0.001, 0.01, 0.1)}
        print(name, "within 10%:", round(float(np.mean(out["relerr"][name]["0.1"])), 3), "within 1%:", round(float(np.mean(out["relerr"][name]["0.01"])), 3), flush=True)

# 4c. scratchpad runs: grade the final answer only (the teacher-forced file also grades the scratchpad)
truth_str = [mg.decode(r[: r.index(mg.EOS)]) for r in A.cpu().tolist()]
for f in sorted(os.listdir(RUNS)):
    if f.endswith(".json") and f != "probes.json":
        args = json.load(open(f"{RUNS}/{f}"))["args"]
        name = f[:-5]
        if (args["mode"] == "cot" or args.get("coco") or args.get("scratch")) and have(name):
            m = load(f"{RUNS}/{name}.pt")
            preds = []
            with torch.no_grad():
                for i in range(0, len(P), 3500):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        toks = m.generate(P[i : i + 3500], mg.C_MAX).cpu().tolist()
                    for row in toks:
                        row = row[: row.index(mg.EOS)] if mg.EOS in row else row
                        preds.append(mg.decode(row).split("]")[-1])
            ok = np.array([a == b for a, b in zip(preds, truth_str)])
            np.save(f"{RUNS}/{name}.ansok.npy", ok)
            print(name, "answer-level exact:", round(float(np.mean([ok[lv == l].mean() for l in range(mg.NLEVELS)])), 4), flush=True)

# 5. per-sub-skill accuracy from the saved per-example results
out["subskills"] = {"keys": sorted(set(keys)), "n": collections.Counter(keys), "runs": {}}
out["subskills"]["n"] = [out["subskills"]["n"][k] for k in out["subskills"]["keys"]]
kidx = {k: np.array([kk == k for kk in keys]) for k in out["subskills"]["keys"]}
for f in sorted(os.listdir(RUNS)):
    if f.endswith(".ok.npy"):
        ok = np.load(f"{RUNS}/{f}")
        if os.path.exists(f"{RUNS}/{f[:-7]}.ansok.npy"):
            ok = np.load(f"{RUNS}/{f[:-7]}.ansok.npy")
        out["subskills"]["runs"][f[:-7]] = [float(ok[kidx[k]].mean()) for k in out["subskills"]["keys"]]

json.dump(out, open(f"{RUNS}/probes.json", "w"))
print("wrote", f"{RUNS}/probes.json")
