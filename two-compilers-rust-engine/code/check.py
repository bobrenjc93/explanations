#!/usr/bin/env python3
"""check.py SIDE: the checks the coordinator runs on recorded states, on SIDE's current state: what the callable
still follows once it is warm (probe.py, probe2.py), and its predictions with the pretrained weights on 128 COCO images
(infer.py). About seven minutes of the GPU, behind the lock. Changes nothing and records no result."""
import json, os, subprocess, sys, time

H = "/home/dev/local/hill2"
side = sys.argv[1]
if side not in ("tc", "vm"):
    sys.exit("check: SIDE is tc or vm")
tree = f"{H}/{side}/tree"
stamp = time.strftime("%m%d_%H%M%S", time.gmtime())
env = dict(os.environ, HILL_CACHE=f"{H}/{side}/cache", HILL_TIMEOUT="900")
out = {}
for kind in ("probe", "probe2", "infer"):
    path = f"{H}/{side}/out/check_{stamp}_{kind}.json"
    with open(path.replace(".json", ".log"), "w") as log:
        subprocess.run([f"{H}/acc/gpu", "python", "-I", f"{H}/acc/{kind}.py", "--tree", tree, "--config", f"{H}/{side}/config.py", "--out", path],
                       stdout=log, stderr=subprocess.STDOUT, env=env)
    out[kind] = json.load(open(path)) if os.path.exists(path) else None
    if out[kind] is None:
        print(f"{kind}: FAILED, see {path.replace('.json', '.log')}")
if out["probe"]:
    print("after the callable is warm (pretrained weights):")
    for name, v in out["probe"]["probes"].items():
        word = "raises: " + v["raised"][:120] if "raised" in v else ("followed" if v.get("followed", v.get("right")) else "NOT followed")
        print(f"  {name}: {word}")
if out["probe2"]:
    for name, v in out["probe2"]["probes"].items():
        word = "raises: " + v["raised"][:120] if "raised" in v else ("followed" if v.get("followed") else "NOT followed")
        print(f"  {name}: {word}")
if out["infer"]:
    r = out["infer"]
    print(f"predictions, mAP@0.5 on 128 COCO images: compiled {r['compiled_bfloat16']['mAP@0.5']:.4f}, eager bfloat16 {r['eager_bfloat16']['mAP@0.5']:.4f}, eager float32 {r['eager_float32']['mAP@0.5']:.4f}")
    print(f"error against float32 (RMSE of the raw output): compiled {r['compiled_against_float32']['raw_output']['rmse']:.3f}, eager bfloat16 {r['eager_bfloat16_against_float32']['raw_output']['rmse']:.3f}")
