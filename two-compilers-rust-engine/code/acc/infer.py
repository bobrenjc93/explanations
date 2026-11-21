"""infer.py --tree TREE --config FILE --out JSON: are the predictions right? The benchmark's yolov3 model with the
original pretrained weights (yolov3-spp, Darknet release), on the 128 COCO images that ship with the benchmark,
letterboxed to the benchmark's 384x512 and run in its batches of 8 in bfloat16. Compares what FILE's
build(model, example_inputs) returns against eager: mAP@0.5 against the ground truth, the detections one by one,
and the raw output."""
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

import copy, glob, importlib.util, json, os, time
import cv2
import numpy as np
from torchbenchmark.models.yolov3.yolo_models import Darknet, load_darknet_weights
from torchbenchmark.models.yolov3.yolo_utils.datasets import letterbox
from torchbenchmark.models.yolov3.yolo_utils.utils import ap_per_class, box_iou, clip_coords, non_max_suppression

H, W, BATCH = 384, 512, 8
torch.manual_seed(0)
model = Darknet(f"{common.YOLO}/cfg/yolov3-spp.cfg", 512)
load_darknet_weights(model, common.WEIGHTS)
model.cuda().eval()

images, truth = [], []
for path in sorted(glob.glob(f"{common.DATA}/images/train2017/*.jpg")):
    img0 = cv2.imread(path)
    h0, w0 = img0.shape[:2]
    img, ratio, (dw, dh) = letterbox(img0, new_shape=(H, W), auto=False)
    images.append(torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).float() / 255.0)
    label = path.replace("/images/", "/labels/").replace(".jpg", ".txt")
    rows = np.loadtxt(label, ndmin=2) if os.path.exists(label) and os.path.getsize(label) else np.zeros((0, 5))
    boxes = np.zeros((len(rows), 5), np.float32)
    boxes[:, 0] = rows[:, 0]
    boxes[:, 1] = (rows[:, 1] - rows[:, 3] / 2) * w0 * ratio[0] + dw
    boxes[:, 2] = (rows[:, 2] - rows[:, 4] / 2) * h0 * ratio[1] + dh
    boxes[:, 3] = (rows[:, 1] + rows[:, 3] / 2) * w0 * ratio[0] + dw
    boxes[:, 4] = (rows[:, 2] + rows[:, 4] / 2) * h0 * ratio[1] + dh
    truth.append(torch.from_numpy(boxes))
batches = [torch.stack(images[i:i + BATCH]).cuda() for i in range(0, len(images), BATCH)]
half = [b.to(torch.bfloat16) for b in batches]


def mean_ap(outputs):
    """mAP@0.5 over the images, as the model's own test.py computes it."""
    stats = []
    for out, first in zip(outputs, range(0, len(images), BATCH)):
        for si, pred in enumerate(non_max_suppression(out.cuda(), conf_thres=0.001, iou_thres=0.6)):
            labels = truth[first + si].cuda()
            tcls = labels[:, 0].tolist()
            if pred is None:
                if len(labels):
                    stats.append((torch.zeros(0, 1, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            clip_coords(pred, (H, W))
            correct = torch.zeros(pred.shape[0], 1, dtype=torch.bool, device="cuda")
            detected = []
            for cls in torch.unique(labels[:, 0]):
                ti = (cls == labels[:, 0]).nonzero().view(-1)
                pi = (cls == pred[:, 5]).nonzero().view(-1)
                if pi.shape[0]:
                    ious, i = box_iou(pred[pi, :4], labels[ti, 1:5]).max(1)
                    for j in (ious > 0.5).nonzero():
                        d = ti[i[j]]
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = True
                            if len(detected) == len(labels):
                                break
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
    tp, conf, pcls, tcls = [np.concatenate(x, 0) for x in zip(*stats)]
    # The model's own ap_per_class, with numpy's trapezoid under its current name: per class, precision and
    # recall down the detections sorted by confidence, AP as the area under the precision envelope at 101 points.
    order = np.argsort(-conf)
    tp, conf, pcls = tp[order, 0], conf[order], pcls[order]
    aps, ps, rs = [], [], []
    for c in np.unique(tcls):
        mine = pcls == c
        n_gt, n_p = int((tcls == c).sum()), int(mine.sum())
        if n_p == 0:
            aps.append(0.0), ps.append(0.0), rs.append(0.0)
            continue
        tpc, fpc = tp[mine].cumsum(), (1 - tp[mine]).cumsum()
        recall, precision = tpc / (n_gt + 1e-16), tpc / (tpc + fpc)
        rs.append(float(np.interp(-0.1, -conf[mine], recall)))
        ps.append(float(np.interp(-0.1, -conf[mine], precision)))
        mrec = np.concatenate(([0.0], recall, [min(recall[-1] + 1e-3, 1.0)]))
        mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([0.0], precision, [0.0])))))
        x = np.linspace(0, 1, 101)
        aps.append(float(np.trapezoid(np.interp(x, mrec, mpre), x)))
    return {"mAP@0.5": float(np.mean(aps)), "precision": float(np.mean(ps)), "recall": float(np.mean(rs))}


def detections(outputs, conf=0.3):
    found = []
    for out in outputs:
        for pred in non_max_suppression(out.cuda(), conf_thres=conf, iou_thres=0.6):
            found.append(torch.zeros(0, 6) if pred is None else pred.cpu())
    return found


def agreement(a, b, overlap=0.9):
    """Detections of A (the reference) found in B: same class, boxes overlapping by at least OVERLAP."""
    total = matched = extra = 0
    score = shift = 0.0
    for x, y in zip(detections(a), detections(b)):
        total += len(x)
        used = set()
        for row in x:
            best, at = 0.0, None
            for k, other in enumerate(y):
                if k in used or other[5] != row[5]:
                    continue
                iou = box_iou(row[None, :4], other[None, :4]).item()
                if iou > best:
                    best, at = iou, k
            if at is not None and best >= overlap:
                used.add(at)
                matched += 1
                score = max(score, abs(row[4] - y[at][4]).item())
                shift = max(shift, (row[:4] - y[at][:4]).abs().max().item())
        extra += len(y) - len(used)
    return {"detections": total, "found": matched, "missing": total - matched, "extra": extra,
            "largest_score_difference": score, "largest_box_shift_px": shift}


def raw(a, b):
    a, b = torch.cat(a).double(), torch.cat(b).double()
    return {"rmse": torch.sqrt(torch.mean((a - b) ** 2)).item(), "largest": (a - b).abs().max().item(),
            "rms_of_reference": torch.sqrt(torch.mean(a ** 2)).item()}


with torch.no_grad():
    eager32 = [model(b)[0].float().cpu() for b in batches]
    model16 = copy.deepcopy(model).to(torch.bfloat16)
    eager16 = [model16(b)[0].float().cpu() for b in half]
    spec = importlib.util.spec_from_file_location("hill_config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    under = copy.deepcopy(model).to(torch.bfloat16)
    fn = config.build(under, (half[0],))
    start, calls = time.perf_counter(), 0
    while calls < 320 or time.perf_counter() - start < 20:
        fn(half[calls % len(half)])
        calls += 1
    torch.cuda.synchronize()
    start = time.perf_counter()
    compiled = [fn(b)[0].float().cpu() for b in half]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    again = [fn(b)[0].float().cpu() for b in half]

result = {
    "label": args.label, "tree": args.tree, "config": args.config, "images": len(images),
    "eager_float32": mean_ap(eager32), "eager_bfloat16": mean_ap(eager16), "compiled_bfloat16": mean_ap(compiled),
    "compiled_against_eager_bfloat16": {"detections": agreement(eager16, compiled), "detections_iou_0.5": agreement(eager16, compiled, 0.5), "raw_output": raw(eager16, compiled)},
    "compiled_against_float32": {"detections": agreement(eager32, compiled), "detections_iou_0.5": agreement(eager32, compiled, 0.5), "raw_output": raw(eager32, compiled)},
    "eager_bfloat16_against_float32": {"detections": agreement(eager32, eager16), "detections_iou_0.5": agreement(eager32, eager16, 0.5), "raw_output": raw(eager32, eager16)},
    "compiled_twice": raw(compiled, again),
    "ms_per_batch_with_copy_out": 1000 * seconds / len(half),
}
json.dump(result, open(args.out, "w"), indent=1)
print(json.dumps(result, indent=1))
