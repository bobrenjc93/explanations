"""chain.py SUITE --only MODEL: the frames on the interpreter's stack when the forward graph launches its first
kernel, with their sizes, under torch.rewrite or (COMPILE=1) torch.compile, in the runner's own timed loop."""
import importlib
import os
import sys

sys.path.insert(0, os.path.join(os.environ["PYTORCH"], "benchmarks/dynamo"))  # a checkout of pytorch with the stack
import common  # noqa: E402
import torch  # noqa: E402
from torch._inductor.runtime.triton_heuristics import CachingAutotuner  # noqa: E402

COMPILE = os.environ.get("COMPILE") == "1"
real = common.speedup_experiment
inner_run = CachingAutotuner.run
state = {"armed": False, "seen": 0}


def size(code):
    # An argument that a nested function uses is a cell and a local in one slot.
    return (len(set(code.co_varnames) | set(code.co_cellvars)) + len(code.co_freevars) + code.co_stacksize + 9) * 8


def run(self, *args, **kwargs):
    if state["armed"]:
        state["armed"] = False
        frame = sys._getframe(1)
        rows = []
        while frame is not None:
            rows.append((frame.f_code.co_name, os.path.basename(frame.f_code.co_filename)[:28], size(frame.f_code)))
            frame = frame.f_back
        total = 0
        print(f"  frames above the first kernel launch ({'torch.compile' if COMPILE else 'torch.rewrite'}), outermost first:")
        for name, where, n in rows[::-1]:
            total += n
            print(f"    {total:7d}  {n:6d}  {name} ({where})")
    return inner_run(self, *args, **kwargs)


CachingAutotuner.run = run
inner_timed = common.timed


def timed(model, fn, *a, **k):
    # The compiled call of the tenth repetition.
    if fn is not model_iter[0]:
        state["seen"] += 1
        state["armed"] = state["seen"] == 10
    return inner_timed(model, fn, *a, **k)


model_iter = [None]


def wrapped(args, model_iter_fn, model, example_inputs, **kwargs):
    model_iter[0] = model_iter_fn
    common.timed = timed
    return real(args, model_iter_fn, model, example_inputs, **kwargs)


common.speedup_experiment = wrapped
suite = sys.argv[1]
module = importlib.import_module(suite)
how = ["--backend", "inductor"] if COMPILE else ["--rewrite"]
sys.argv = [suite + ".py", "--performance", "--cold-start-latency", "--training", "--amp", "--disable-cudagraphs", "--device", "cuda", *how,
            "--output", f"/tmp/chain_{os.getpid()}.csv", *sys.argv[2:]]
getattr(module, {"timm_models": "timm_main"}.get(suite, f"{suite}_main"))()
