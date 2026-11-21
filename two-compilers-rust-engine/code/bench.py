"""bench.py --config FILE --out JSON [--probes]: throughput of TorchBench yolov3 inference, set up as the
performance dashboard sets it up (eval mode, batch 8 of 3x384x512, bfloat16, no_grad), through whatever
FILE's build(model, example_inputs) returns. The callable is checked against eager before and after it is
timed. This file is the same for both sides of the experiment and is not to be changed by either."""
import argparse, copy, gc, importlib.util, json, math, os, random, sys, time

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--probes", action="store_true")
args = parser.parse_args()
args.config, args.out = os.path.abspath(args.config), os.path.abspath(args.out)

BENCH = "/home/dev/local/g/benchmark"
os.chdir(BENCH)
sys.path.insert(0, BENCH)

import numpy as np
import torch

BATCH, POOL = 8, 4
MULTIPLIER, TOL = 10.0, 1e-4


def leaves(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for o in out for t in leaves(o)]
    if isinstance(out, dict):
        return [t for k in sorted(out) for t in leaves(out[k])]
    raise TypeError(f"output of type {type(out).__name__}")


def shape_of(out):
    if isinstance(out, torch.Tensor):
        return ("T", tuple(out.shape), str(out.dtype))
    if isinstance(out, (list, tuple)):
        return (type(out).__name__, [shape_of(o) for o in out])
    if isinstance(out, dict):
        return ("dict", [(k, shape_of(out[k])) for k in sorted(out)])
    return (type(out).__name__,)


def rmse(a, b):
    return torch.sqrt(torch.mean((a.double() - b.double()) ** 2)).item()


def keep(out):
    return [t.detach().to("cpu", copy=True) for t in leaves(out)]


def check(out, ref, gold):
    """The dashboard's rule for a result in half precision: equal within TOL, or no further from the float64
    result than MULTIPLIER times what eager is (the multiplier it allows where convolutions were folded)."""
    got = keep(out)
    if len(got) != len(ref):
        return False, float("inf")
    worst = 0.0
    for o, r, g in zip(got, ref, gold):
        if o.shape != r.shape or o.dtype != r.dtype:
            return False, float("inf")
        if torch.allclose(o.float(), r.float(), rtol=TOL, atol=TOL, equal_nan=True):
            continue
        eager_error, error = rmse(g, r), rmse(g, o)
        if math.isnan(error):
            return False, float("inf")
        ratio = error / max(eager_error, 1e-30)
        worst = max(worst, ratio)
        if error > MULTIPLIER * eager_error + TOL / 10:
            return False, worst
    return True, worst


def sync():
    torch.cuda.synchronize()


torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)

from torchbenchmark.models.yolov3 import Model

benchmark = Model(test="eval", device="cuda", batch_size=None, extra_args=[])
model, _ = benchmark.get_module()
model.eval()
model = model.to(torch.bfloat16)
generator = torch.Generator().manual_seed(1337)
pool = [torch.rand(BATCH, 3, 384, 512, generator=generator).to("cuda").to(torch.bfloat16) for _ in range(POOL)]

result = {"torch": os.path.dirname(torch.__file__), "config": os.path.abspath(args.config)}
with torch.no_grad():
    # What eager computes, and what float64 computes, before anything of the side under test is loaded.
    eager_model = copy.deepcopy(model)
    ref = [keep(eager_model(x)) for x in pool]
    layout = shape_of(eager_model(pool[0]))
    wide = copy.deepcopy(model).double()
    gold = [keep(wide(x.double())) for x in pool]
    del wide
    gc.collect()
    torch.cuda.empty_cache()

    for _ in range(30):
        eager_model(pool[0])
    sync()
    start = time.perf_counter()
    for i in range(100):
        eager_model(pool[i % POOL])
        sync()
    result["eager_qps"] = BATCH * 100 / (time.perf_counter() - start)

    spec = importlib.util.spec_from_file_location("hill_config", args.config)
    config = importlib.util.module_from_spec(spec)
    start = time.perf_counter()
    spec.loader.exec_module(config)
    fn = config.build(model, (pool[0],))
    fn(pool[0])
    sync()
    result["first_call_s"] = time.perf_counter() - start

    # Warm up: at least 300 calls and 20 s, then until the last 50 calls take what the 50 before them took.
    times, start = [], time.perf_counter()
    while True:
        t = time.perf_counter()
        fn(pool[len(times) % POOL])
        sync()
        times.append(time.perf_counter() - t)
        elapsed = time.perf_counter() - start
        if len(times) >= 300 and elapsed >= 20:
            a, b = sum(times[-100:-50]), sum(times[-50:])
            if abs(a - b) <= 0.02 * max(a, b) or elapsed > 240:
                break
    result["warmup_calls"], result["warmup_s"] = len(times), elapsed

    ok_before = [check(fn(x), r, g) for x, r, g in zip(pool, ref, gold)]

    blocks = []
    for _ in range(10):
        sync()
        start = time.perf_counter()
        for i in range(40):
            fn(pool[i % POOL])
            sync()
        blocks.append(BATCH * 40 / (time.perf_counter() - start))
    result["qps_blocks"] = blocks
    result["qps"] = len(blocks) / sum(1 / b for b in blocks)  # all 400 calls over all their time
    result["latency_ms"] = 1000 * BATCH / result["qps"]

    ok_after = [check(fn(x), r, g) for x, r, g in zip(pool, ref, gold)]
    same_layout = shape_of(fn(pool[0])) == layout
    result["accuracy_ok"] = bool(same_layout and all(ok for ok, _ in ok_before + ok_after))
    result["error_over_eager_error"] = max(w for _, w in ok_before + ok_after)

    if args.probes:
        probes = {}
        # 1. A weight changed in place after everything was compiled: does the callable compute with the new one?
        try:
            # The last parameter: the bias of the convolution an output is read from. (A convolution's
            # weight would not do: this model's batch norms take the scale of it out again.)
            name = list(dict(model.named_parameters()))[-1]
            new, old = dict(model.named_parameters())[name], dict(eager_model.named_parameters())[name]
            saved = new.detach().clone()
            new.add_(1.0)
            old.add_(1.0)
            want = keep(eager_model(pool[0]))
            got = keep(fn(pool[0]))
            moved = max(rmse(a, b) for a, b in zip(ref[0], want))
            off = max(rmse(a, b) for a, b in zip(got, want))
            probes["follows_weight_update"] = bool(moved > 0 and off < 0.1 * moved)
            probes["weight_update_moved_eager_by"], probes["callable_off_by"] = moved, off
            new.copy_(saved)
            old.copy_(saved)
            probes["right_after_weight_restored"] = check(fn(pool[0]), ref[0], gold[0])[0]
        except Exception as e:
            probes["follows_weight_update"] = f"raised {type(e).__name__}: {e}"[:300]
        # 2. A batch of another size.
        try:
            small = pool[1][:3].contiguous()
            want, got = keep(eager_model(small)), keep(fn(small))
            scale = max(torch.sqrt(torch.mean(w.double() ** 2)).item() for w in want)
            off = max(rmse(a, b) for a, b in zip(got, want))
            probes["right_for_another_batch_size"] = bool(off <= 0.05 * scale)
        except Exception as e:
            probes["right_for_another_batch_size"] = f"raised {type(e).__name__}: {e}"[:300]
        # 3. And back: the timed shape is still right.
        try:
            probes["right_afterwards"] = check(fn(pool[2]), ref[2], gold[2])[0]
        except Exception as e:
            probes["right_afterwards"] = f"raised {type(e).__name__}: {e}"[:300]
        result["probes"] = probes

with open(args.out, "w") as f:
    json.dump(result, f)
print(json.dumps({k: v for k, v in result.items() if k != "qps_blocks"}, indent=1))
