"""Shared by infer.py and train.py: picking the tree torch is imported from, and the benchmark's directory."""
import os, sys

BENCH = "/home/dev/local/g/benchmark"
YOLO = f"{BENCH}/torchbenchmark/models/yolov3"
DATA = f"{BENCH}/torchbenchmark/data/.data/coco128"
WEIGHTS = "/home/dev/local/hill/acc/dl/yolov3-spp/yolov3-spp.weights"


def use_tree(tree):
    """torch from TREE (a standalone tree as tools/overlay.py makes) instead of the build tree."""
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "ScikitBuildRedirectingFinder"]
    sys.path.insert(0, tree)
    os.chdir(BENCH)
    sys.path.insert(1, BENCH)
    import torch

    if not torch.__file__.startswith(tree):
        raise AssertionError(f"torch came from {torch.__file__}, not {tree}")
    return torch
