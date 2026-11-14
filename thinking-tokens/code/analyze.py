"""Per-sub-skill accuracy of saved checkpoints on the held-out test set."""
import sys, collections, numpy as np, torch
import mathgen as mg
from model import Net, Cfg
from subtypes import subtype

dev = "cuda"
D = np.load("data.npz")
P = torch.from_numpy(D["test_P"]).to(dev).long(); A = torch.from_numpy(D["test_A"]).to(dev).long()
lv = D["test_level"]
exprs = [mg.decode(r)[:-1] for r in D["test_P"]]
keys = [(int(l), subtype(int(l), e)) for l, e in zip(lv, exprs)]


def load(path):
    ck = torch.load(path, map_location=dev)
    c = {k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__}
    m = Net(Cfg(**c)).to(dev).eval()
    m.load_state_dict(ck["state"])
    return m


@torch.no_grad()
def exact(m, **kw):
    out = []
    for i in range(0, len(P), 3500):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m.exact(P[i : i + 3500], A[i : i + 3500], **kw))
    return torch.cat(out).cpu().numpy()


if __name__ == "__main__":
    cols = {p.split("/")[-1][:-3]: exact(load(p)) for p in sys.argv[1:]}
    n = collections.Counter(keys)
    print(f"{'sub-skill':44s} {'n':>5s} " + " ".join(f"{c:>10s}" for c in cols))
    for k in sorted(n):
        idx = np.array([kk == k for kk in keys])
        print(f"{mg.LEVELS[k[0]][:2]} {k[1]:41s} {n[k]:5d} " + " ".join(f"{ok[idx].mean():10.3f}" for ok in cols.values()))
