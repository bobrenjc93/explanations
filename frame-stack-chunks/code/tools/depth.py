"""depth.py SUITE --only MODEL: after the runner's own experiment, the compiled function (torch.rewrite, or
torch.compile with COMPILE=1) timed at the end of chains of small Python frames of growing length: the same call,
only deeper on the interpreter's stack of frames."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.environ["PYTORCH"], "benchmarks/dynamo"))  # a checkout of pytorch with the stack
import common  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

real = common.speedup_experiment
COMPILE = os.environ.get("COMPILE") == "1"
STEP = int(os.environ.get("STEP", "4"))
MOST = int(os.environ.get("MOST", "400"))


def deeper(n, thunk):
    return thunk() if n == 0 else deeper(n - 1, thunk)


def wrapped(args, model_iter_fn, model, example_inputs, **kwargs):
    out = real(args, model_iter_fn, model, example_inputs, **kwargs)
    fn = torch._dynamo.run(model_iter_fn) if COMPILE else common.rewritten(model_iter_fn)

    def once():
        return common.timed(model, fn, example_inputs, return_result=False, times=1, collect_outputs=False)

    def eager():
        with torch.compiler.set_stance("force_eager"):
            return common.timed(model, model_iter_fn, example_inputs, return_result=False, times=1, collect_outputs=False)

    print(f"  code of deeper: {deeper.__code__.co_nlocals} locals, stack {deeper.__code__.co_stacksize}")
    for depth in range(0, MOST, STEP):
        got = [deeper(depth, once) for _ in range(3)]
        base = deeper(depth, eager) if depth % (STEP * 8) == 0 else float("nan")
        print(f"  depth {depth:4d}: {np.median(got) * 1e3:6.2f} ms   eager {base * 1e3:6.2f} ms")
    return out


common.speedup_experiment = wrapped
suite = sys.argv[1]
module = importlib.import_module(suite)
how = ["--backend", "inductor"] if COMPILE else ["--rewrite"]
sys.argv = [suite + ".py", "--performance", "--cold-start-latency", "--training", "--amp", "--disable-cudagraphs", "--device", "cuda", *how,
            "--output", f"/tmp/depth_{os.getpid()}.csv", *sys.argv[2:]]
getattr(module, {"timm_models": "timm_main"}.get(suite, f"{suite}_main"))()
