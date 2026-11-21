"""after.py --tree TREE --config FILE --out JSON: does the speed survive the things probe.py does? The benchmark's
setup (random weights, batch 8, 384x512, bfloat16). Times the callable, then after each event again: a batch of
another size, images of another size, a weight updated in place."""
import argparse, sys

sys.path.insert(0, "/home/dev/local/hill2/acc")
import common

parser = argparse.ArgumentParser()
parser.add_argument("--tree", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--label", default="")
args = parser.parse_args()
import os as _os
args.tree, args.config, args.out = (_os.path.abspath(v) for v in (args.tree, args.config, args.out))
torch = common.use_tree(args.tree)

import importlib.util, json, time
from torchbenchmark.models.yolov3 import Model

torch.manual_seed(1337)
model, _ = Model(test="eval", device="cuda", batch_size=None, extra_args=[]).get_module()
model.eval()
model = model.to(torch.bfloat16)
g = torch.Generator().manual_seed(1337)
pool = [torch.rand(8, 3, 384, 512, generator=g).to("cuda").to(torch.bfloat16) for _ in range(4)]
other_batch = torch.rand(3, 3, 384, 512, generator=g).to("cuda").to(torch.bfloat16)
other_size = torch.rand(8, 3, 320, 416, generator=g).to("cuda").to(torch.bfloat16)


def qps(fn, warm):
    start, calls = time.perf_counter(), 0
    while calls < warm or time.perf_counter() - start < 15:
        fn(pool[calls % 4]); calls += 1
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(200):
        fn(pool[i % 4]); torch.cuda.synchronize()
    return 8 * 200 / (time.perf_counter() - start)


result = {"label": args.label}
with torch.no_grad():
    spec = importlib.util.spec_from_file_location("hill_config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    fn = config.build(model, (pool[0],))
    result["at first"] = qps(fn, 320)
    for _ in range(3):
        fn(other_batch)
    result["after a batch of 3 images"] = qps(fn, 320)
    for _ in range(3):
        fn(other_size)
    result["after images of another size"] = qps(fn, 320)
    list(model.parameters())[-1].add_(0.01)
    result["after a weight was updated in place"] = qps(fn, 320)
json.dump(result, open(args.out, "w"), indent=1)
print(json.dumps(result, indent=1))
