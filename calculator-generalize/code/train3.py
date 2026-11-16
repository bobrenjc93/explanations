"""Train on the 14-level benchmark (build_data3.py). Same recipe as train2.py: one optimiser step accumulates gradients
over a mini-batch from every bucket (together a uniform draw from the pool); buckets differ in prompt and target width."""
import argparse, gc, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as Fn
import mathgen as mg, mathgen3 as m3
import model as M
from model import Net, Cfg

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--out", default="runs/hard")
p.add_argument("--data", default="hard/full.npz")
p.add_argument("--d", type=int, default=256)
p.add_argument("--L", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--ut", type=int, default=2)
p.add_argument("--lo", type=int, default=1)
p.add_argument("--hi", type=int, default=3)
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--bs", type=int, default=2048)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--decay_from", type=float, default=0.7, help="fraction of training after which the lr decays (cosine to 10%)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_every", type=int, default=1000)
p.add_argument("--compile", default="reduce-overhead")
p.add_argument("--init", default="", help="checkpoint to start from")
p.add_argument("--think", default="none", help="thinking tokens in front of the output: none | cont | um")
p.add_argument("--K", type=int, default=0)
p.add_argument("--gen_every", type=int, default=1000, help="every N steps, generate solutions for a test subset and log answer accuracy")
p.add_argument("--gen_n", type=int, default=200, help="test problems per level in that subset")
p.add_argument("--log_every", type=int, default=250)
p.add_argument("--accum", type=int, default=1, help="split every optimiser step into this many passes over the buckets (same batch, less memory)")
p.add_argument("--data_frac", type=float, default=1.0, help="train on a fixed random fraction of the pool's rows (the data-scaling runs)")
p.add_argument("--save", type=int, default=1)
p.add_argument("--ckpt_every", type=int, default=0, help="save a resumable checkpoint every N steps; a run restarted with the same name continues from it")
p.add_argument("--time_steps", type=int, default=0, help="print per-bucket wall time for this many steps after the warm-up (diagnostic)")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
M.FUSED = True
dev = "cuda"
os.makedirs(args.out, exist_ok=True)
Z = np.load(args.data)
W = [int(w) for w in Z["widths"]]
PW = [int(w) for w in Z["pw"]]
NB = len(W)
trP = [torch.from_numpy(Z[f"P{b}"]).to(dev) for b in range(NB)]
trT = [torch.from_numpy(Z[f"T{b}"]).to(dev) for b in range(NB)]
teP = [torch.from_numpy(Z[f"tP{b}"]).to(dev).long() for b in range(NB)]
teT = [torch.from_numpy(Z[f"tT{b}"]).to(dev).long() for b in range(NB)]
teI = [torch.from_numpy(Z[f"tI{b}"]).long() for b in range(NB)]
te_lv = Z["test_level"]
truth = [m3.decode(r[: list(r).index(mg.EOS)]) for r in Z["test_A"]]
NL = m3.NLEVELS
N = sum(len(x) for x in trP)
nb = [max(2, round(args.bs // args.accum * len(x) / N)) if len(x) else 0 for x in trP]  # rows per bucket in one pass
nb[int(np.argmax(nb))] += args.bs // args.accum - sum(nb)
# a fixed random subset of every bucket when --data_frac < 1: the same rows are revisited for the whole run
gsub = torch.Generator().manual_seed(1234)
keep = [None if args.data_frac >= 1 or not len(x) else torch.randperm(len(x), generator=gsub)[: max(n, round(len(x) * args.data_frac))].to(dev)
        for x, n in zip(trP, nb)]
unique_rows = int(sum(len(x) if k is None else len(k) for x, k in zip(trP, keep)))
# expected number of supervised tokens per step: normalises the summed loss so buckets add up to a token average
tok_norm = args.accum * float(sum(n * (t[:: max(1, len(t) // 200_000)] != 0).sum(1).float().mean().item() for n, t in zip(nb, trT) if n))

mode = args.think if args.think != "none" and args.K > 0 else ("ut" if args.ut > 1 else "none")
cfg = Cfg(vocab=m3.VOCAB, d=args.d, L=args.L, heads=args.heads, mode=mode, K=args.K if mode in ("cont", "um") else 0, ut=args.ut, lo=args.lo,
          hi=args.hi, max_pos=max(PW) + max(W) + 64)
model = Net(cfg).to(dev)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=dev)["state"])
decay = [q for q in model.parameters() if q.ndim >= 2]
nodecay = [q for q in model.parameters() if q.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": nodecay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
print(json.dumps(model.describe() | {"batch_per_bucket": nb, "widths": W, "prompt_widths": PW, "tok_norm": tok_norm}), flush=True)


def lr_at(step):
    w = min(1.0, (step + 1) / args.warmup)
    t = step / args.steps
    d = 1.0 if t < args.decay_from else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - args.decay_from) / (1 - args.decay_from)))
    return args.lr * w * d


def lossfn(P, T):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(P, T)
    return Fn.cross_entropy(logits.flatten(0, 1).float(), T.flatten(), ignore_index=0, reduction="sum") / tok_norm


fns = [torch.compile(lossfn, mode=args.compile, dynamic=False) if args.compile != "none" else lossfn for _ in range(NB)]
# decoding budget per bucket: the model's own solution can be much longer than the reference (a different starting estimate
# for a root turns a short division into a long one), and a budget tied to the reference length silently truncates those
BUDGET = [min(2 * W[b] + 96, W[-1] + 24) for b in range(NB)]
rows_at = lambda b: max(16, min(1024, int(3e5 / (PW[b] + W[b]))))  # teacher-forced evaluation batch: fewer rows for long sequences
# decoding batch: as many rows as keep the key/value buffers of one batch under about 6 GB (fewer batches = fewer sequential decodes)
rows_gen = lambda b: max(16, min(1024, int(6e9 / (args.d * BUDGET[b] * 4 * len(model.sched)))))


@torch.no_grad()
def evaluate():
    """Teacher-forced exact match on the whole target, per level."""
    model.eval()
    ok = np.zeros(len(truth), bool)
    for b in range(NB):
        n = rows_at(b)
        for i in range(0, len(teP[b]), n):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ok[teI[b][i : i + n].numpy()] = model.exact(teP[b][i : i + n], teT[b][i : i + n]).cpu().numpy()
    model.train()
    torch.cuda.empty_cache()  # give the evaluation's memory back: another run may share the GPU
    return [float(ok[te_lv == l].mean()) for l in range(NL)]


# a fixed subset of the test set (the first gen_n problems of every level) for answer accuracy during training
rank = np.zeros(len(truth), int)
for l in range(NL):
    idx = np.where(te_lv == l)[0]
    rank[idx] = np.arange(len(idx))
sub_rows = [torch.from_numpy(np.where(rank[teI[b].numpy()] < args.gen_n)[0]) for b in range(NB)]
sub_mask = rank < args.gen_n


@torch.no_grad()
def evaluate_gen(subset=False):
    """Greedy generation; the answer is what follows the closing bracket (the whole output when there is no block)."""
    model.eval()
    preds = [""] * len(truth)
    for b in range(NB):
        P, I = (teP[b][sub_rows[b].to(dev)], teI[b][sub_rows[b]]) if subset else (teP[b], teI[b])
        n = rows_gen(b)
        for i in range(0, len(P), n):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model.generate_long(P[i : i + n], BUDGET[b]).cpu().tolist()
            for j, row in zip(I[i : i + n].tolist(), toks):
                preds[j] = m3.decode(row[: row.index(mg.EOS)] if mg.EOS in row else row)
        torch.cuda.empty_cache()
    model.train()
    ans = [t.split("]")[-1] for t in preds]
    em = np.array([a == b for a, b in zip(ans, truth)])
    return em, ans, preds


def within(ans):
    """Every value of the answer within 1% of the true one (same count of values)."""
    out = []
    for a, b in zip(ans, truth):
        try:
            x, y = [float(v) for v in a.split(",")], [float(v) for v in b.split(",")]
            out.append(len(x) == len(y) and all(abs(u - v) <= 0.01 * abs(v) + 1e-9 for u, v in zip(x, y)))
        except ValueError:
            out.append(a == b)
    return np.array(out)


# gradients are accumulated over the buckets; the buffers must exist before any compiled backward records a CUDA graph
for q in model.parameters():
    q.grad = torch.zeros_like(q)
g = torch.Generator(device=dev).manual_seed(args.seed)
CKPT = f"{args.out}/{args.name}.ckpt.pt"
start = 0
if args.ckpt_every and os.path.exists(CKPT):  # an interrupted run: pick up where it stopped
    ck = torch.load(CKPT, map_location=dev)
    model.load_state_dict(ck["state"])
    opt.load_state_dict(ck["opt"])
    start = ck["step"]
    print(f"resuming from step {start}", flush=True)
log = open(f"{args.out}/{args.name}.jsonl", "a" if start else "w")
t0 = time.time() - (ck["elapsed"] if start else 0.0)
model.train()
run_loss, n_loss = 0.0, 0
log_loss, log_n = 0.0, 0
lvl = lambda x, m=None: [float(x[(te_lv == l) & (m if m is not None else True)].mean()) for l in range(NL)]
for step in range(start, args.steps):
    for grp in opt.param_groups:
        grp["lr"] = lr_at(step)
    fresh = step - start < 6  # CUDA graphs are being recorded: keep the cyclic GC from freeing kernels mid-capture
    if fresh:
        gc.collect()
        gc.disable()
    opt.zero_grad(set_to_none=False)
    tot = 0.0
    for b in [b for _ in range(args.accum) for b in range(NB)]:
        if not nb[b]:
            continue
        idx = torch.randint(len(trP[b]) if keep[b] is None else len(keep[b]), (nb[b],), device=dev, generator=g)
        if keep[b] is not None:
            idx = keep[b][idx]
        torch.compiler.cudagraph_mark_step_begin()
        tb = time.time()
        loss = fns[b](trP[b][idx].long(), trT[b][idx].long())
        loss.backward()
        if 6 <= step < 6 + args.time_steps:
            torch.cuda.synchronize()
            print(f"  step {step} bucket {b:2d} ({PW[b]:2d}+{W[b]:4d} x {nb[b]:3d}) {1000 * (time.time() - tb):7.1f} ms", flush=True)
        if step % 20 == 0:
            tot += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if fresh:
        gc.enable()
        print(f"step {step} {time.time() - t0:.0f}s", flush=True)
    if step % 20 == 0:
        run_loss += tot
        n_loss += 1
        log_loss += tot
        log_n += 1
    if (step + 1) % args.log_every == 0:
        log.write(json.dumps({"step": step + 1, "train_loss": log_loss / max(log_n, 1), "t": time.time() - t0}) + "\n")
        log.flush()
        log_loss, log_n = 0.0, 0
    if args.gen_every and ((step + 1) % args.gen_every == 0 or step + 1 == args.steps):
        tg = time.time()
        em, _, _ = evaluate_gen(subset=True)
        ans = lvl(em, sub_mask)
        # per-problem outcomes of the subset, in test order: the horizon over training is computed from these
        log.write(json.dumps({"step": step + 1, "ans": ans, "ok": "".join("1" if x else "0" for x in em[sub_mask]), "t": time.time() - t0}) + "\n")
        log.flush()
        print(f"{step+1:6d} answers " + " ".join(f"{x:.2f}" for x in ans) + f"  mean {np.mean(ans):.3f}  ({time.time() - tg:.0f}s)", flush=True)
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        em = evaluate()
        rec = {"step": step + 1, "loss": run_loss / max(n_loss, 1), "em": em, "t": time.time() - t0, "lr": lr_at(step)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"{step+1:6d} loss {rec['loss']:.4f}  tf-exact " + " ".join(f"{x:.2f}" for x in em) + f"  mean {np.mean(em):.3f}  {rec['t']:.0f}s", flush=True)
        run_loss, n_loss = 0.0, 0

    if args.ckpt_every and (step + 1) % args.ckpt_every == 0 and step + 1 < args.steps:
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "elapsed": time.time() - t0}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)

train_time = time.time() - t0
if args.save:
    torch.save({"cfg": model.describe(), "state": model.state_dict()}, f"{args.out}/{args.name}.pt")
tg = time.time()


@torch.no_grad()
def tf_stats(Ps, Ts, levels=None):
    """Teacher-forced cross-entropy per token and whole-target exact match over lists of (prompt, target) buckets."""
    model.eval()
    nll = tok = 0.0
    okc = n = 0
    per = np.zeros((NL, 2))
    for b in range(NB):
        P, T = Ps[b], Ts[b]
        for i in range(0, len(P), rows_at(b)):
            Pb, Tb = P[i : i + rows_at(b)].long(), T[i : i + rows_at(b)].long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(Pb, Tb)
            ce = Fn.cross_entropy(logits.flatten(0, 1).float(), Tb.flatten(), ignore_index=0, reduction="none").view(Tb.shape).sum(1)
            cnt = (Tb != 0).sum(1)
            nll += ce.sum().item()
            tok += cnt.sum().item()
            okc += ((logits.argmax(-1) == Tb) | (Tb == 0)).all(1).sum().item()
            n += len(Pb)
            if levels is not None:
                lv_b = te_lv[levels[b][i : i + rows_at(b)].numpy()]
                for l in range(NL):
                    m = torch.from_numpy(lv_b == l).to(dev)
                    per[l] += [ce[m].sum().item(), cnt[m].sum().item()]
    model.train()
    return nll / max(tok, 1), okc / max(n, 1), (per[:, 0] / np.maximum(per[:, 1], 1)).tolist()


test_loss, test_tf, test_loss_level = tf_stats(teP, teT, teI)
gtr = torch.Generator(device=dev).manual_seed(7)
tr_rows = [torch.randint(len(trP[b]) if keep[b] is None else len(keep[b]), (max(2, round(4000 * nb[b] * args.accum / args.bs)),), device=dev, generator=gtr)
           if nb[b] else torch.zeros(0, dtype=torch.long, device=dev) for b in range(NB)]
tr_rows = [r if keep[b] is None else keep[b][r] for b, r in enumerate(tr_rows)]
train_loss, train_tf, _ = tf_stats([trP[b][tr_rows[b]] for b in range(NB)], [trT[b][tr_rows[b]] for b in range(NB)])
print(f"test loss {test_loss:.4f} tf-exact {test_tf:.4f} | train (rows it trained on) loss {train_loss:.4f} tf-exact {train_tf:.4f} | unique rows {unique_rows:,}", flush=True)
em, ans, preds = evaluate_gen()
close = within(ans)
n_prompt = np.zeros(len(truth))
for b in range(NB):
    n_prompt[teI[b].numpy()] = (teP[b] != 0).sum(1).cpu().numpy()
n_gen = np.array([len(t) + 1 for t in preds])
apps = len(model.sched)
compute = {"target_tokens_per_step": tok_norm, "padded_positions_per_step": float(args.accum * sum(n * (pw + w + cfg.K) for n, pw, w in zip(nb, PW, W))),
           "layer_applications": apps, "batch": args.bs, "mean_target_tokens": tok_norm / args.bs}
compute["unique_rows"] = unique_rows
res = {"name": args.name, "args": vars(args), "model": model.describe(), "train_time": train_time, "gen_time": time.time() - tg, "compute": compute,
       "loss": {"test": test_loss, "test_tf_exact": test_tf, "test_level": test_loss_level, "train": train_loss, "train_tf_exact": train_tf},
       "final": {"em": lvl(em), "close": lvl(close), "mean_em": float(np.mean(lvl(em))), "tle": float(np.mean((n_prompt + n_gen + cfg.K) * apps)),
                 "tle_level": lvl((n_prompt + n_gen + cfg.K) * apps), "gen_tokens_level": lvl(n_gen)}}
json.dump(res, open(f"{args.out}/{args.name}.json", "w"))
np.save(f"{args.out}/{args.name}.ok.npy", em)
json.dump(preds, open(f"{args.out}/{args.name}.preds.json", "w"))
print("FINAL exact " + " ".join(f"{x:.3f}" for x in lvl(em)) + f"  mean {np.mean(lvl(em)):.4f}   within1% {np.mean(lvl(close)):.4f}  train {train_time/60:.1f} min", flush=True)
