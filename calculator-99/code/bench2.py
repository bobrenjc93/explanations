"""Solo throughput of the bucketed trainer: eager vs compiled, and effective TFLOPS. Writes runs/ladder/bench.json."""
import gc, json, sys, time
import numpy as np, torch
import torch.nn.functional as Fn
import mathgen as mg, mathgen2 as m2
from model import Net, Cfg
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64
torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
Z = np.load("data23.npz")
W = [int(w) for w in Z["widths"]]
bs = 2048
N = sum(len(Z[f"L{b}"]) for b in range(len(W)))
nb = [max(2, round(bs * len(Z[f"L{b}"]) / N)) for b in range(len(W))]
nb[int(np.argmax(nb))] += bs - sum(nb)
batches = [(torch.from_numpy(Z[f"P{b}"][: nb[b]]).to(dev).long(), torch.from_numpy(Z[f"T{b}"][: nb[b]]).to(dev).long()) for b in range(len(W))]
positions = sum(n * (mg.P_MAX + w) for n, w in zip(nb, W))


def bench(d, L, heads, lo, hi, compiled, steps):
    model = Net(Cfg(vocab=m2.VOCAB, d=d, L=L, heads=heads, mode="ut", ut=2, lo=lo, hi=hi, max_pos=mg.P_MAX + W[-1] + 40)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    for q in model.parameters():
        q.grad = torch.zeros_like(q)

    def lossfn(P, T):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(P, T)
        return Fn.cross_entropy(logits.flatten(0, 1).float(), T.flatten(), ignore_index=0, reduction="sum") / 1e5

    fns = [torch.compile(lossfn, mode="reduce-overhead", dynamic=False) if compiled else lossfn for _ in batches]

    def step():
        opt.zero_grad(set_to_none=False)
        for f, (P, T) in zip(fns, batches):
            torch.compiler.cudagraph_mark_step_begin()
            f(P, T).backward()
        opt.step()

    gc.collect(); gc.disable()
    for _ in range(6):
        step()
    gc.enable()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    ms = (time.time() - t) / steps * 1000
    tflops = positions * len(model.sched) * 72 * d * d / (ms / 1000) / 1e12
    return ms, tflops


out = {"positions_per_step": positions}
out["eager_ms"], _ = bench(256, 4, 4, 1, 3, False, 10)
out["compiled_ms"], out["tflops_3m"] = bench(256, 4, 4, 1, 3, True, 60)
out["compiled_ms_11m"], out["tflops"] = bench(384, 6, 6, 2, 4, True, 40)
json.dump(out, open("runs/ladder/bench.json", "w"))
print(out)
