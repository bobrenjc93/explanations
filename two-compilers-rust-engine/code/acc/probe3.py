"""probe3.py --tree TREE --config FILE --out JSON: three probes that were not announced to the sides (rule 6 of the
checks): a weight written through .data, and a result kept across later calls, as a tensor and through DLPack.
(Derived from probe2.py, whose description follows.) One more probe than probe.py has, for what a guard filter on
module attributes gives up: after the callable is warm, a hyperparameter of a module that is not a tensor is
edited (the negative slope of one LeakyReLU in the middle of the model), on the callable's model and on an eager
copy, and the two outputs are compared; then it is set back. Pretrained weights, real images, bfloat16."""
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
args.tree, args.config, args.out = (_os.path.abspath(v) if not v.endswith("/live") else v for v in (args.tree, args.config, args.out))
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
out = []
for path in sorted(glob.glob(f"{common.DATA}/images/train2017/*.jpg"))[:16]:
    img, _, _ = letterbox(cv2.imread(path), new_shape=(384, 512), auto=False)
    out.append(torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).float() / 255.0)
x = torch.stack(out).cuda().to(torch.bfloat16)
a, b = x[:8].contiguous(), x[8:].contiguous()
rmse = lambda p, q: torch.sqrt(torch.mean((p.double() - q.double()) ** 2)).item()
first = lambda o: (o[0] if isinstance(o, (tuple, list)) else o).float()

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
    slack = max(3 * noise, 0.2)
    probes = {}
    # 1. A write through .data: it moves no version counter, and eager computes with the new values.
    convs_m = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)]
    convs_e = [m for m in eager.modules() if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)]
    k = len(convs_m) // 2
    entry = {"module": f"3x3 convolution {k} of {len(convs_m)}"}
    try:
        convs_m[k].weight.data.mul_(0.5)
        convs_e[k].weight.data.mul_(0.5)
        want, got = first(eager(a)), first(fn(a))
        moved = rmse(want, base_e)
        entry["eager_moved_by"], entry["off_by"] = moved, rmse(got, want)
        entry["followed"] = bool(moved > 5 * noise and entry["off_by"] <= max(0.2 * moved, slack))
        convs_m[k].weight.data.mul_(2.0)
        convs_e[k].weight.data.mul_(2.0)
        entry["right_after_undo"] = bool(rmse(first(fn(a)), first(eager(a))) <= slack)
    except Exception as e:
        entry["raised"] = f"{type(e).__name__}: {e}"[:300]
    probes["a 3x3 convolution's weight halved through .data"] = entry
    # 2. A result that the caller keeps is not changed by a later call.
    entry = {}
    try:
        kept = fn(a)
        kept0 = kept[0] if isinstance(kept, (tuple, list)) else kept
        copy0 = kept0.clone()
        fn(b)
        fn(a)
        entry["followed"] = bool(torch.equal(kept0, copy0))
    except Exception as e:
        entry["raised"] = f"{type(e).__name__}: {e}"[:300]
    probes["a result kept as a tensor across two more calls"] = entry
    # 3. The same through DLPack: the capsule keeps the memory, the tensor object is dropped.
    entry = {}
    try:
        out = fn(a)
        out0 = out[0] if isinstance(out, (tuple, list)) else out
        copy0 = out0.clone()
        capsule = torch.utils.dlpack.to_dlpack(out0)
        del out, out0
        fn(b)
        fn(a + 0.25)
        back = torch.utils.dlpack.from_dlpack(capsule)
        entry["followed"] = bool(torch.equal(back, copy0))
    except Exception as e:
        entry["raised"] = f"{type(e).__name__}: {e}"[:300]
    probes["a result kept through DLPack across two more calls"] = entry
    result["probes"] = probes
json.dump(result, open(args.out, "w"), indent=1)
print(json.dumps(result, indent=1))
