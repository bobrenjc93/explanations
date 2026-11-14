"""Train on v2 scratchpads (data2.npz). Batches are drawn per length bucket so shapes stay static and padding small;
each optimiser step accumulates gradients over one mini-batch from every bucket (together a uniform draw from the pool)."""
import argparse, collections, gc, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as Fn
import mathgen as mg, mathgen2 as m2
from model import Net, Cfg

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--out", default="runs/v2")
p.add_argument("--data", default="data2.npz")
p.add_argument("--d", type=int, default=256)
p.add_argument("--L", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--ut", type=int, default=2)
p.add_argument("--lo", type=int, default=1)
p.add_argument("--hi", type=int, default=3)
p.add_argument("--pos", default="rope")
p.add_argument("--norm", default="ln")
p.add_argument("--wbits", type=int, default=0)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--bs", type=int, default=2048)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--decay_from", type=float, default=0.7, help="fraction of training after which the lr decays (cosine to 10%)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_every", type=int, default=2000)
p.add_argument("--compile", default="reduce-overhead")
p.add_argument("--init", default="", help="checkpoint to start from")
p.add_argument("--think", default="none", help="thinking tokens in front of the output: none | cont | um | disc")
p.add_argument("--force_open", default="", help="evaluate with the output forced to start with this character (thinking effort)")
p.add_argument("--K", type=int, default=0)
p.add_argument("--gen_every", type=int, default=0, help="every N steps, generate solutions for a test subset and log answer accuracy")
p.add_argument("--gen_n", type=int, default=200, help="test problems per level in that subset")
p.add_argument("--log_every", type=int, default=250)
p.add_argument("--save", type=int, default=1)
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
dev = "cuda"
os.makedirs(args.out, exist_ok=True)
Z = np.load(args.data)
W = [int(w) for w in Z["widths"]]
NB = len(W)
trP = [torch.from_numpy(Z[f"P{b}"]).to(dev) for b in range(NB)]
trT = [torch.from_numpy(Z[f"T{b}"]).to(dev) for b in range(NB)]
teP = [torch.from_numpy(Z[f"tP{b}"]).to(dev).long() for b in range(NB)]
teT = [torch.from_numpy(Z[f"tT{b}"]).to(dev).long() for b in range(NB)]
teI = [torch.from_numpy(Z[f"tI{b}"]).long() for b in range(NB)]
te_lv = Z["test_level"]
truth = [mg.decode(r[: list(r).index(mg.EOS)]) for r in Z["test_A"]]
NL = mg.NLEVELS
N = sum(len(x) for x in trP)
nb = [max(2, round(args.bs * len(x) / N)) if len(x) else 0 for x in trP]  # short formats leave the widest buckets empty
nb[int(np.argmax(nb))] += args.bs - sum(nb)
# expected number of supervised tokens per step: normalises the summed loss so buckets add up to a token average
# (rows are sampled with a stride: a bucket is ordered by level, so its head is not representative)
tok_norm = float(sum(n * (t[:: max(1, len(t) // 200_000)] != 0).sum(1).float().mean().item() for n, t in zip(nb, trT) if n))

mode = args.think if args.think != "none" and args.K > 0 else ("ut" if args.ut > 1 else "none")
cfg = Cfg(vocab=m2.VOCAB, d=args.d, L=args.L, heads=args.heads, mode=mode, K=args.K if mode in ("cont", "um", "disc") else 0, ut=args.ut, lo=args.lo,
          hi=args.hi, pos=args.pos, norm=args.norm, wbits=args.wbits, max_pos=mg.P_MAX + W[-1] + 40)
model = Net(cfg).to(dev)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=dev)["state"])
decay = [q for q in model.parameters() if q.ndim >= 2]
nodecay = [q for q in model.parameters() if q.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": nodecay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
print(json.dumps(model.describe() | {"batch_per_bucket": nb, "widths": W, "tok_norm": tok_norm}), flush=True)


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


@torch.no_grad()
def evaluate():
    """Teacher-forced exact match on the whole worked solution, per level."""
    model.eval()
    ok = np.zeros(len(truth), bool)
    for b in range(NB):
        for i in range(0, len(teP[b]), 2048):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ok[teI[b][i : i + 2048].numpy()] = model.exact(teP[b][i : i + 2048], teT[b][i : i + 2048]).cpu().numpy()
    model.train()
    return [float(ok[te_lv == l].mean()) for l in range(NL)]


FORCE = [m2.STOI[c] for c in args.force_open] or None
# a fixed subset of the test set (the first gen_n problems of every level) for answer accuracy during training
rank = np.zeros(len(truth), int)
for l in range(NL):
    idx = np.where(te_lv == l)[0]
    rank[idx] = np.arange(len(idx))
sub_rows = [torch.from_numpy(np.where(rank[teI[b].numpy()] < args.gen_n)[0]) for b in range(NB)]


@torch.no_grad()
def evaluate_gen(subset=False):
    """Greedy generation of the worked solution; the answer is what follows the closing bracket."""
    model.eval()
    preds = [""] * len(truth)
    for b in range(NB):
        P, I = (teP[b][sub_rows[b].to(dev)], teI[b][sub_rows[b]]) if subset else (teP[b], teI[b])
        for i in range(0, len(P), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model.generate(P[i : i + 1024], min(2 * W[b] + 96, W[-1] + 24), prefix=FORCE).cpu().tolist()
            for j, row in zip(I[i : i + 1024].tolist(), toks):
                row = row[: row.index(mg.EOS)] if mg.EOS in row else row
                preds[j] = m2.decode(row)
    model.train()
    if subset:
        ok = np.array([p.split("]")[-1] == t for p, t in zip(preds, truth)])
        return [float(ok[(te_lv == l) & (rank < args.gen_n)].mean()) for l in range(NL)]
    ans = [t.split("]")[-1] for t in preds]  # what follows the thinking block; the whole output when there is none
    em = np.array([a == b for a, b in zip(ans, truth)])
    close = []
    for a, b in zip(ans, truth):
        try:
            close.append(abs(float(a) - float(b)) <= 0.01 * abs(float(b)) + 1e-9)
        except ValueError:
            close.append(False)
    close = np.array(close)
    return em, close, preds


# gradients are accumulated over the buckets; the buffers must exist before any compiled backward records a CUDA graph
for q in model.parameters():
    q.grad = torch.zeros_like(q)
g = torch.Generator(device=dev).manual_seed(args.seed)
log = open(f"{args.out}/{args.name}.jsonl", "w")
t0 = time.time()
model.train()
run_loss, n_loss = 0.0, 0
log_loss, log_n = 0.0, 0
for step in range(args.steps):
    for grp in opt.param_groups:
        grp["lr"] = lr_at(step)
    fresh = step < 6  # CUDA graphs are being recorded: keep the cyclic GC from freeing kernels mid-capture
    if fresh:
        gc.collect()
        gc.disable()
    opt.zero_grad(set_to_none=False)
    tot = 0.0
    for b in range(NB):
        if not nb[b]:
            continue
        idx = torch.randint(len(trP[b]), (nb[b],), device=dev, generator=g)
        torch.compiler.cudagraph_mark_step_begin()
        loss = fns[b](trP[b][idx].long(), trT[b][idx].long())
        loss.backward()
        if step % 20 == 0:
            tot += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if fresh:
        gc.enable()
    if step % 20 == 0:
        run_loss += tot
        n_loss += 1
        log_loss += tot
        log_n += 1
    if (step + 1) % args.log_every == 0:
        log.write(json.dumps({"step": step + 1, "train_loss": log_loss / max(log_n, 1), "t": time.time() - t0}) + "\n")
        log_loss, log_n = 0.0, 0
    if args.gen_every and ((step + 1) % args.gen_every == 0 or step + 1 == args.steps):
        ans = evaluate_gen(subset=True)
        log.write(json.dumps({"step": step + 1, "ans": ans, "t": time.time() - t0}) + "\n")
        log.flush()
        print(f"{step+1:6d} answers " + " ".join(f"{x:.2f}" for x in ans) + f"  mean {np.mean(ans):.3f}", flush=True)
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        em = evaluate()
        rec = {"step": step + 1, "loss": run_loss / max(n_loss, 1), "em": em, "t": time.time() - t0, "lr": lr_at(step)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"{step+1:6d} loss {rec['loss']:.4f}  tf-exact " + " ".join(f"{x:.2f}" for x in em) + f"  mean {np.mean(em):.3f}  {rec['t']:.0f}s", flush=True)
        run_loss, n_loss = 0.0, 0

train_time = time.time() - t0
if args.save:
    torch.save({"cfg": model.describe(), "state": model.state_dict()}, f"{args.out}/{args.name}.pt")
em, close, preds = evaluate_gen()
lvl = lambda x: [float(x[te_lv == l].mean()) for l in range(NL)]
n_prompt = np.zeros(len(truth)); n_gen = np.zeros(len(truth))
for b in range(NB):
    n_prompt[teI[b].numpy()] = (teP[b] != 0).sum(1).cpu().numpy()
n_gen = np.array([len(t) for t in preds])
tle = float(np.mean((n_prompt + n_gen + cfg.K) * len(model.sched)))
compute = {"target_tokens_per_step": tok_norm, "padded_positions_per_step": float(sum(n * (mg.P_MAX + w + cfg.K) for n, w in zip(nb, W))),
           "real_positions_per_step": float(sum(n * ((p[:: max(1, len(p) // 100_000)] != 0).sum(1).float().mean().item() + cfg.K) for n, p in zip(nb, trP) if n) + tok_norm),
           "layer_applications": len(model.sched), "batch": args.bs, "mean_target_tokens": tok_norm / args.bs}
res = {"name": args.name, "args": vars(args), "model": model.describe(), "train_time": train_time, "compute": compute,
       "final": {"em_gen": lvl(em), "close": lvl(close), "mean_em": float(np.mean(lvl(em))), "tle": tle, "em": lvl(em)},
       "preds": [preds[i] for i in range(0, len(preds), len(preds) // 280)]}
json.dump(res, open(f"{args.out}/{args.name}.json", "w"))
np.save(f"{args.out}/{args.name}.ok.npy", em)
print("FINAL exact " + " ".join(f"{x:.3f}" for x in lvl(em)) + f"  mean {np.mean(lvl(em)):.4f}   within1% {np.mean(lvl(close)):.4f}  tle {tle:.0f}  train {train_time/60:.1f} min", flush=True)
