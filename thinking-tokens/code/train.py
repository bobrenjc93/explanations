"""Train one variant on the calculator curriculum and write logs/results/checkpoint."""
import argparse, collections, gc, json, math, os, time
import numpy as np
import torch
import mathgen as mg
from model import Net, Cfg

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--mode", default="none")
p.add_argument("--K", type=int, default=0)
p.add_argument("--lo", type=int, default=0)
p.add_argument("--hi", type=int, default=-1)
p.add_argument("--window", type=int, default=0)
p.add_argument("--slots", type=int, default=0)
p.add_argument("--vt", type=int, default=1024)
p.add_argument("--ut", type=int, default=1)
p.add_argument("--tau", type=float, default=1.0)
p.add_argument("--d", type=int, default=256)
p.add_argument("--L", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--bs", type=int, default=2048)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--curriculum", type=int, default=1)
p.add_argument("--krand", type=int, default=0, help="train with K ~ U{0..K} per step")
p.add_argument("--evalK", default="", help="comma list of K to evaluate at the end")
p.add_argument("--eval_every", type=int, default=1000)
p.add_argument("--out", default="runs")
p.add_argument("--save", type=int, default=1)
p.add_argument("--compile", default="reduce-overhead")
p.add_argument("--coco", type=int, default=0, help="stages of CoT internalisation (0 = off)")
p.add_argument("--scratch", type=int, default=0, help="train on scratchpad targets with any thinking mode")
p.add_argument("--coco_frac", type=float, default=0.6, help="fraction of steps spent on plain CoT first")
p.add_argument("--kset", default="", help="with --krand: comma list of K values to draw from")
args = p.parse_args()

torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = "cuda"
os.makedirs(args.out, exist_ok=True)
D = np.load("data.npz")
cot = args.mode == "cot" or args.coco > 0 or args.scratch > 0
tr_P = torch.from_numpy(D["train_P"]).to(dev)
tr_A = torch.from_numpy(D["train_C" if cot else "train_A"]).to(dev)
te_P = torch.from_numpy(D["test_P"]).to(dev).long()
te_A = torch.from_numpy(D["test_A"]).to(dev).long()
te_C = torch.from_numpy(D["test_C"]).to(dev).long()
te_lv = torch.from_numpy(D["test_level"]).to(dev).long()
NL = mg.NLEVELS
N = len(tr_P) // NL

cfg = Cfg(vocab=mg.VOCAB, d=args.d, L=args.L, heads=args.heads, mode=args.mode, K=args.K, lo=args.lo, hi=args.hi,
          window=args.window, slots=args.slots, vt=args.vt, ut=args.ut, tau=args.tau)
model = Net(cfg).to(dev)
decay = [q for n, q in model.named_parameters() if q.ndim >= 2]
nodecay = [q for n, q in model.named_parameters() if q.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.98))
print(json.dumps(model.describe()), flush=True)

# curriculum: a new level every stage (half of each batch), earlier levels replayed, then a uniform phase
UNITS = np.array([1, 1, 2, 2, 2, 3, 3, 2], float)
BOUNDS = np.cumsum(UNITS) / UNITS.sum()


def level_probs(step):
    if not args.curriculum:
        return torch.full((NL,), 1 / NL)
    s = int(np.searchsorted(BOUNDS, (step + 0.5) / (args.steps * (args.coco_frac if args.coco else 1.0))))
    pr = torch.zeros(NL)
    if s >= NL:
        pr[:] = 1 / NL
    elif s == 0:
        pr[0] = 1
    else:
        pr[:s] = 0.5 / s
        pr[s] = 0.5
    return pr


def lr_at(step):
    w = min(1.0, (step + 1) / 500)
    t = step / args.steps
    d = 1.0 if t < 0.7 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - 0.7) / 0.3))
    return args.lr * w * d


COMMA, RB = mg.STOI[","], mg.STOI["]"]


def drop_items(C, n):
    """Remove the first n scratchpad items from '[a,b,c]ans' rows (n >= number of items: '[]ans')."""
    if n == 0:
        return C
    comma = C == COMMA
    cs = comma.cumsum(1)
    nth = ((cs == n) & comma).float().argmax(1)
    cut = torch.where(cs[:, -1] >= n, nth, (C == RB).float().argmax(1) - 1)
    col = torch.arange(C.shape[1], device=C.device)[None]
    src = col + cut[:, None]
    out = torch.gather(C, 1, src.clamp(max=C.shape[1] - 1))
    out = torch.where(src >= C.shape[1], torch.zeros_like(out), out)
    out[:, 0] = C[:, 0]
    return out


def coco_stage(step):
    """-> (items dropped, thoughts) for this step; stage 0 is plain CoT."""
    if not args.coco:
        return 0, args.K
    t = step / args.steps
    if t < args.coco_frac:
        return 0, 0
    st = min(args.coco, 1 + int((t - args.coco_frac) / (1 - args.coco_frac) * args.coco))
    return (99 if st == args.coco else st), round(args.K * st / args.coco)


@torch.no_grad()
def evaluate(n_per_level=None, drop=0, **kw):
    """Teacher-forced exact match per level (== greedy decoding exact match)."""
    model.eval()
    ok = torch.zeros(len(te_P), dtype=torch.bool, device=dev)
    idx = torch.arange(len(te_P), device=dev)
    if n_per_level:
        idx = idx.view(NL, -1)[:, :n_per_level].flatten()
    for i in range(0, len(idx), 3500):
        j = idx[i : i + 3500]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ok[j] = model.exact(te_P[j], drop_items(te_C[j], drop) if cot else te_A[j], **kw)
    model.train()
    return [ok[idx[te_lv[idx] == lv]].float().mean().item() for lv in range(NL)], ok.cpu().numpy()


def answers(toks):
    out = []
    for row in toks.tolist():
        row = row[: row.index(mg.EOS)] if mg.EOS in row else row
        s = mg.decode(row)
        out.append(s.split("]")[-1] if cot else s)
    return out


@torch.no_grad()
def evaluate_gen(**kw):
    """Greedy generation: exact match, 'within 1%' and final answers."""
    model.eval()
    preds = []
    for i in range(0, len(te_P), 3500):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds += answers(model.generate(te_P[i : i + 3500], mg.C_MAX if cot else mg.A_MAX, **kw))
    model.train()
    truth = answers(te_A)
    if cot:
        truth = [mg.decode(r[: r.index(mg.EOS)]) for r in te_A.tolist()]
    em = np.array([a == b for a, b in zip(preds, truth)])
    close = []
    for a, b in zip(preds, truth):
        try:
            close.append(abs(float(a) - float(b)) <= 0.01 * abs(float(b)) + 1e-9)
        except ValueError:
            close.append(False)
    lv = te_lv.cpu().numpy()
    return ([float(em[lv == i].mean()) for i in range(NL)], [float(np.array(close)[lv == i].mean()) for i in range(NL)], preds)


def lossfn(P, A, K):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.loss(P, A, K=K)


if args.compile != "none":
    lossfn = torch.compile(lossfn, mode=args.compile)
kset = [int(k) for k in args.kset.split(",")] if args.kset else list(range(args.K + 1))
g = torch.Generator(device=dev).manual_seed(args.seed)
gcpu = torch.Generator().manual_seed(args.seed)
log = open(f"{args.out}/{args.name}.jsonl", "w")
t0 = time.time()
model.train()
run_loss, n_loss = 0.0, 0
k_calls = collections.Counter()
for step in range(args.steps):
    for grp in opt.param_groups:
        grp["lr"] = lr_at(step)
    lv = torch.multinomial(level_probs(step), args.bs, replacement=True, generator=gcpu).to(dev)
    idx = lv * N + torch.randint(N, (args.bs,), device=dev, generator=g)
    P, A = tr_P[idx].long(), tr_A[idx].long()
    drop, K = coco_stage(step)
    if args.krand:
        K = kset[int(torch.randint(0, len(kset), (1,), generator=gcpu))]
    if args.coco:
        A = drop_items(A, drop)
    # A CUDA graph is recorded the first few times a new K is seen. If the cyclic GC frees a Triton kernel
    # in the middle of that recording, the capture is invalidated, so keep the GC out of those steps.
    fresh = k_calls[K] < 4
    k_calls[K] += 1
    if fresh:
        gc.collect()
        gc.disable()
    torch.compiler.cudagraph_mark_step_begin()
    loss = lossfn(P, A, K)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if fresh:
        gc.enable()
    run_loss += loss.item() if step % 20 == 0 else 0.0
    n_loss += step % 20 == 0
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        em, _ = evaluate(n_per_level=500, drop=drop, K=K if args.coco else None)
        rec = {"step": step + 1, "loss": run_loss / max(n_loss, 1), "em": em, "t": time.time() - t0, "lr": lr_at(step)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"{step+1:6d} loss {rec['loss']:.4f}  em " + " ".join(f"{x:.2f}" for x in em) + f"  mean {np.mean(em):.3f}  {rec['t']:.0f}s", flush=True)
        run_loss, n_loss = 0.0, 0

train_time = time.time() - t0
n_prompt = (te_P != mg.PAD).sum(1).float()
FINAL_DROP = 99 if args.coco else 0
n_ans = ((drop_items(te_C, FINAL_DROP) if cot else te_A) != mg.PAD).sum(1).float() - 1
res = {"name": args.name, "args": vars(args), "model": model.describe(), "train_time": train_time}
Ks = [int(k) for k in args.evalK.split(",")] if args.evalK else [args.K]
res["by_K"] = {}
for K in Ks:
    em_tf, ok = evaluate(K=K, drop=FINAL_DROP)
    em_gen, close, preds = evaluate_gen(K=K)
    tle = float(torch.tensor([model.tle(a, b, K) for a, b in zip(n_prompt.tolist(), n_ans.tolist())]).mean())
    res["by_K"][K] = {"em": em_tf, "em_gen": em_gen, "close": close, "tle": tle, "mean_em": float(np.mean(em_tf))}
    print(f"K={K} em {np.mean(em_tf):.4f} gen {np.mean(em_gen):.4f} close {np.mean(close):.4f} tle {tle:.1f}", flush=True)
    if K == args.K:
        res["preds"] = preds[:: len(preds) // 280]
        np.save(f"{args.out}/{args.name}.ok.npy", ok)
res["final"] = res["by_K"][args.K]
json.dump(res, open(f"{args.out}/{args.name}.json", "w"))
if args.save:
    torch.save({"cfg": model.describe(), "state": model.state_dict()}, f"{args.out}/{args.name}.pt")
