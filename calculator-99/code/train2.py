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
nb = [max(2, round(args.bs * len(x) / N)) for x in trP]
nb[int(np.argmax(nb))] += args.bs - sum(nb)
# expected number of supervised tokens per step: normalises the summed loss so buckets add up to a token average
tok_norm = float(sum(n * (t[: 200_000] != 0).sum(1).float().mean().item() for n, t in zip(nb, trT)))

cfg = Cfg(vocab=m2.VOCAB, d=args.d, L=args.L, heads=args.heads, mode="ut" if args.ut > 1 else "none", ut=args.ut, lo=args.lo, hi=args.hi,
          pos=args.pos, norm=args.norm, wbits=args.wbits, max_pos=mg.P_MAX + W[-1] + 40)
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


@torch.no_grad()
def evaluate_gen():
    """Greedy generation of the worked solution; the answer is what follows the closing bracket."""
    model.eval()
    preds = [""] * len(truth)
    for b in range(NB):
        for i in range(0, len(teP[b]), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model.generate(teP[b][i : i + 1024], min(2 * W[b] + 96, W[-1] + 24)).cpu().tolist()
            for j, row in zip(teI[b][i : i + 1024].tolist(), toks):
                row = row[: row.index(mg.EOS)] if mg.EOS in row else row
                preds[j] = m2.decode(row)
    model.train()
    ans = [t.split("]")[-1] if "]" in t else "" for t in preds]
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
tle = float(np.mean((n_prompt + n_gen) * len(model.sched)))
res = {"name": args.name, "args": vars(args), "model": model.describe(), "train_time": train_time,
       "final": {"em_gen": lvl(em), "close": lvl(close), "mean_em": float(np.mean(lvl(em))), "tle": tle, "em": lvl(em)},
       "preds": [preds[i] for i in range(0, len(preds), len(preds) // 280)]}
json.dump(res, open(f"{args.out}/{args.name}.json", "w"))
np.save(f"{args.out}/{args.name}.ok.npy", em)
print("FINAL exact " + " ".join(f"{x:.3f}" for x in lvl(em)) + f"  mean {np.mean(lvl(em)):.4f}   within1% {np.mean(lvl(close)):.4f}  tle {tle:.0f}  train {train_time/60:.1f} min", flush=True)
