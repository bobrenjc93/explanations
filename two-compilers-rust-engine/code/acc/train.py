"""train.py --tree TREE --mode eager|tc|vm --steps N --out JSON [--amp]: the loss curve of training the benchmark's
yolov3 (yolov3-spp from a seeded random start) on the 128 COCO images that ship with it, with the model's own
detection loss and its SGD settings. The forward (and so its backward) goes through eager, torch.compile or
torch.vm with Inductor graphs; data order, seeds and cuDNN's deterministic algorithms are the same for all."""
import argparse, sys

sys.path.insert(0, "/home/dev/local/hill2/acc")
import common

parser = argparse.ArgumentParser()
parser.add_argument("--tree", required=True)
parser.add_argument("--mode", required=True, choices=("eager", "tc", "vm"))
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", required=True)
parser.add_argument("--amp", action="store_true")
parser.add_argument("--label", default="")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--nudge", type=float, default=0.0)
args = parser.parse_args()
if args.deterministic:
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import os as _os
args.tree, args.out = _os.path.abspath(args.tree), _os.path.abspath(args.out)
torch = common.use_tree(args.tree)

import json, random, time
import numpy as np
import torchbenchmark.models.yolov3  # sets cudnn.benchmark; undone below
from torchbenchmark.models.yolov3.yolo_models import Darknet
from torchbenchmark.models.yolov3.yolo_utils.datasets import LoadImagesAndLabels
from torchbenchmark.models.yolov3.yolo_utils.utils import compute_loss, labels_to_class_weights

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if args.deterministic:
    # Every kernel with a deterministic implementation uses it, and float32 convolutions and products are not
    # run in TF32 (the default on this GPU, where two algorithms for one convolution differ in the fourth digit).
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
BATCH, SIZE = 16, 416
hyp = {"giou": 3.54, "cls": 37.4, "cls_pw": 1.0, "obj": 64.3, "obj_pw": 1.0, "iou_t": 0.20, "lr0": 0.01, "lrf": 0.0005,
       "momentum": 0.937, "weight_decay": 0.0005, "fl_gamma": 0.0, "hsv_h": 0.0138, "hsv_s": 0.678, "hsv_v": 0.36,
       "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0}
model = Darknet(f"{common.YOLO}/cfg/yolov3-spp.cfg", SIZE).cuda()
dataset = LoadImagesAndLabels(f"{common.DATA}/coco128.txt", SIZE, BATCH, augment=False, hyp=hyp, rect=False)
batches = []
for first in range(0, len(dataset) - BATCH + 1, BATCH):
    imgs, targets, _, _ = dataset.collate_fn([dataset[i] for i in range(first, first + BATCH)])
    batches.append((imgs.cuda().float() / 255.0, targets.cuda()))
model.nc, model.hyp, model.gr = 80, hyp, 1.0
model.class_weights = labels_to_class_weights(dataset.labels, 80).cuda()

# The model's own parameter groups: weight decay on the convolutions' weights only.
decay, biases, rest = [], [], []
for name, p in model.named_parameters():
    (biases if ".bias" in name else decay if "Conv2d.weight" in name else rest).append(p)
LR = hyp["lr0"] * BATCH / 64
optimizer = torch.optim.SGD(rest, lr=LR, momentum=hyp["momentum"], nesterov=True)
optimizer.add_param_group({"params": decay, "weight_decay": hyp["weight_decay"]})
optimizer.add_param_group({"params": biases})
if args.nudge:
    # A change in the last digits of one layer's weights: how far does eager itself move for that?
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if "Conv2d.weight" in n).mul_(1.0 + args.nudge)
model.train()


def call(x):
    return model(x)


if args.mode == "tc":
    forward = torch.compile(call)
elif args.mode == "vm":
    forward = torch.vm(call, passes=(*torch.vm.DEFAULT_PASSES, torch.vm.passes.Graphs(), torch.vm.passes.Inductor()))
else:
    forward = call

losses, parts, times = [], [], []
for step in range(args.steps):
    imgs, targets = batches[step % len(batches)]
    for group in optimizer.param_groups:
        group["lr"] = LR * min(1.0, (step + 1) / 50)
    torch.cuda.synchronize()
    start = time.perf_counter()
    if args.amp:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = forward(imgs)
        pred = [p.float() for p in pred]
    else:
        pred = forward(imgs)
    loss, items = compute_loss(pred, targets, model)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - start)
    losses.append(loss.item())
    parts.append(items.tolist())
    if not np.isfinite(losses[-1]):
        break

result = {"label": args.label, "deterministic": args.deterministic, "mode": args.mode, "tree": args.tree, "amp": args.amp, "steps": len(losses), "loss": losses,
          "giou_obj_cls_total": parts, "ms_per_step_last_50": 1000 * float(np.median(times[-50:])),
          "weights_checksum": float(sum(p.double().abs().sum().item() for p in model.parameters()))}
if args.mode == "tc":
    result["compiled"] = dict(torch._dynamo.utils.counters["stats"])
if args.mode == "vm":
    regions = [r for r in torch.vm.regions(forward) if r.get("state") == "compiled" and not r.get("within")]
    result["compiled"] = {"graphs": len(regions), "variants": sum(r.get("variants", 0) for r in regions),
                          "operations": sum(r.get("ops", 0) for r in regions if isinstance(r.get("ops", 0), int)),
                          "hits": sum(r.get("hits", 0) for r in regions), "stats": forward.stats().get("graphs")}
json.dump(result, open(args.out, "w"))
print(json.dumps({k: v for k, v in result.items() if k not in ("loss", "giou_obj_cls_total")}, indent=1))
print("loss at steps 0, 10, 50, 100, last:", [round(losses[i], 4) for i in (0, 10, 50, 100, len(losses) - 1) if i < len(losses)])
