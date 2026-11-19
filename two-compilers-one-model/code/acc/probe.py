"""probe.py --tree TREE --config FILE --out JSON: what does the compiled callable still follow? The benchmark's
yolov3 with the pretrained weights (with random ones the output hardly depends on the convolutions), real images,
bfloat16. After the callable is warm, things a program may do to its model are done one at a time, to the
callable's model and to an eager copy, and the two outputs compared; then undone, and compared again."""
import argparse, sys

sys.path.insert(0, "/home/dev/local/hill/acc")
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

import copy, glob, importlib.util, json, time
import cv2
import numpy as np
from torchbenchmark.models.yolov3.yolo_models import Darknet, load_darknet_weights
from torchbenchmark.models.yolov3.yolo_utils.datasets import letterbox

torch.manual_seed(0)
model = Darknet(f"{common.YOLO}/cfg/yolov3-spp.cfg", 512)
load_darknet_weights(model, common.WEIGHTS)
model.cuda().eval().to(torch.bfloat16)


def images(h, w, n):
    out = []
    for path in sorted(glob.glob(f"{common.DATA}/images/train2017/*.jpg"))[:n]:
        img, _, _ = letterbox(cv2.imread(path), new_shape=(h, w), auto=False)
        out.append(torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).float() / 255.0)
    return torch.stack(out).cuda().to(torch.bfloat16)


x = images(384, 512, 16)
a, b = x[:8].contiguous(), x[8:].contiguous()
rmse = lambda p, q: torch.sqrt(torch.mean((p.double() - q.double()) ** 2)).item()
first = lambda out: (out[0] if isinstance(out, (tuple, list)) else out).float()


def close(got, want, scale):
    """Right: within a fifth of what the change moved eager by, or (for no change) within bfloat16's noise."""
    return rmse(got, want) <= scale


result = {"label": args.label, "tree": args.tree, "config": args.config}
with torch.no_grad():
    eager = copy.deepcopy(model)
    spec = importlib.util.spec_from_file_location("hill_config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    fn = config.build(model, (a,))
    start, calls = time.perf_counter(), 0
    while calls < 320 or time.perf_counter() - start < 20:
        fn(a if calls % 2 else b)
        calls += 1
    base_e, base_c = first(eager(a)), first(fn(a))
    noise = rmse(base_c, base_e)
    result["rmse_against_eager_before"] = noise
    slack = max(3 * noise, 0.2)

    params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    e_params, e_buffers = dict(eager.named_parameters()), dict(eager.named_buffers())
    norms = [n for n in params if "BatchNorm2d.bias" in n]
    convs = [n for n in params if "Conv2d.weight" in n]
    means = [n for n in buffers if "running_mean" in n]
    changes = {
        "bias of the last convolution += 1": (params, e_params, list(params)[-1], lambda t: t.add_(1.0)),
        "bias of a batch norm in the middle += 0.5": (params, e_params, norms[len(norms) // 2], lambda t: t.add_(0.5)),
        "weight of a convolution in the middle *= 0.5": (params, e_params, convs[len(convs) // 2], lambda t: t.mul_(0.5)),
        "running mean of a batch norm in the middle += 0.5": (buffers, e_buffers, means[len(means) // 2], lambda t: t.add_(0.5)),
    }
    probes = {}
    for what, (mine, theirs, name, change) in changes.items():
        entry = {"tensor": name}
        try:
            saved = mine[name].detach().clone()
            change(mine[name]); change(theirs[name])
            want, got = first(eager(a)), first(fn(a))
            moved = rmse(want, base_e)
            entry["eager_moved_by"], entry["off_by"] = moved, rmse(got, want)
            entry["followed"] = bool(moved > 5 * noise and entry["off_by"] <= max(0.2 * moved, slack))
            mine[name].copy_(saved); theirs[name].copy_(saved)
            entry["right_after_undo"] = bool(close(first(fn(a)), first(eager(a)), slack))
        except Exception as e:
            entry["raised"] = f"{type(e).__name__}: {e}"[:300]
        probes[what] = entry

    def other(what, make):
        entry = {}
        try:
            got, want = make()
            entry["off_by"] = rmse(got, want)
            entry["right"] = bool(got.shape == want.shape and entry["off_by"] <= slack)
        except Exception as e:
            entry["raised"] = f"{type(e).__name__}: {e}"[:300]
        probes[what] = entry

    other("a batch of 3 images", lambda: (first(fn(b[:3].contiguous())), first(eager(b[:3].contiguous()))))
    small = images(320, 416, 8)
    other("images of another size (320x416)", lambda: (first(fn(small)), first(eager(small))))
    other("the timed shape again", lambda: (first(fn(b)), first(eager(b))))

    # A parameter replaced by a new tensor (what load_state_dict does not do, but `module.weight = Parameter(...)` does).
    try:
        name = list(params)[-1]
        module_name, attr = name.rsplit(".", 1)
        new = torch.nn.Parameter(params[name].detach().clone() + 1.0)
        setattr(model.get_submodule(module_name), attr, new)
        setattr(eager.get_submodule(module_name), attr, torch.nn.Parameter(new.detach().clone()))
        want, got = first(eager(a)), first(fn(a))
        moved = rmse(want, base_e)
        probes["last bias replaced by a new Parameter"] = {"eager_moved_by": moved, "off_by": rmse(got, want), "followed": bool(rmse(got, want) <= max(0.2 * moved, slack))}
    except Exception as e:
        probes["last bias replaced by a new Parameter"] = {"raised": f"{type(e).__name__}: {e}"[:300]}

    # The model switched to training mode: eager then returns the three raw outputs, not the detections.
    try:
        model.train(); eager.train()
        want, got = eager(a), fn(a)
        same_kind = type(want) is type(got) and len(want) == len(got)
        probes["model.train()"] = {"followed": bool(same_kind and all(w.shape == g.shape for w, g in zip(want, got)))}
        model.eval(); eager.eval()
        probes["model.train()"]["right_after_undo"] = bool(close(first(fn(a)), first(eager(a)), 3 * slack))
    except Exception as e:
        probes["model.train()"] = {"raised": f"{type(e).__name__}: {e}"[:300]}
result["probes"] = probes
json.dump(result, open(args.out, "w"), indent=1)
print(json.dumps(result, indent=1))
