"""Where do worked solutions go wrong? Generate every test solution, find the first step that differs from the
reference and bucket it by the kind of step. Usage: errs.py <checkpoint> [n_show] [data file]"""
import collections, json, re, sys
import numpy as np, torch
import mathgen as mg, mathgen2 as m2
from model import Net, Cfg
from subtypes import subtype

path = sys.argv[1]
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dev = "cuda"
ck = torch.load(path, map_location=dev)
model = Net(Cfg(**{k: v for k, v in ck["cfg"].items() if k in Cfg.__dataclass_fields__})).to(dev).eval()
model.load_state_dict(ck["state"])
Z = np.load(sys.argv[3] if len(sys.argv) > 3 else "data2.npz")
W = [int(w) for w in Z["widths"]]
lv = Z["test_level"]
truth = [mg.decode(r[: list(r).index(mg.EOS)]) for r in Z["test_A"]]
exprs = [mg.decode(r)[:-1] for r in np.load("data.npz")["test_P"]]
preds, refs = [""] * len(truth), [""] * len(truth)
with torch.no_grad():
    for b in range(len(W)):
        P = torch.from_numpy(Z[f"tP{b}"]).to(dev).long(); I = Z[f"tI{b}"]; T = Z[f"tT{b}"]
        for i in range(0, len(P), 1024):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                toks = model.generate(P[i : i + 1024], min(2 * W[b] + 96, W[-1] + 24)).cpu().tolist()
            for j, row, ref in zip(I[i : i + 1024].tolist(), toks, T[i : i + 1024].tolist()):
                row = row[: row.index(mg.EOS)] if mg.EOS in row else row
                preds[j] = m2.decode(row); refs[j] = m2.decode(ref[: ref.index(mg.EOS)])


def kind(step):
    """The kind of step, by its syntax."""
    if step.startswith("]") or step == "":
        return "final answer"
    head = step.split("=")[0].split(":")[0]
    if "~" in step:
        return "sqrt estimate"
    if re.match(r"^(sin|cos|tan|asin|acos|atan)\(", head):
        return "trig table"
    if re.match(r"^(ln|log)\(", head):
        return "ln/log (table or sum)"
    if re.match(r"^(sqrt|gcd|lcm|mod|abs)\(", head):
        return head.split("(")[0] + " summary"
    if "!" in head or "nCr" in head or "nPr" in head:
        return "n!, nCr, nPr table"
    if "^" in head:
        return "power (table or summary)"
    if "%" in head:
        return "integer division (%)"
    if "/" in head:
        return "long division"
    if "*" in head:
        return "multiplication with work" if ":" in step else "multiplication, one-digit"
    return "add/subtract"


first = collections.Counter(); total = collections.Counter(); examples = collections.defaultdict(list)
ok_ans = np.array([p.split("]")[-1] == t if "]" in p else False for p, t in zip(preds, truth)])
ok_trace = np.array([p == r for p, r in zip(preds, refs)])
wrong_part = collections.Counter()
for i, (p, r) in enumerate(zip(preds, refs)):
    rs = r[1:].replace("]", ";]").split(";")
    for st in rs:
        total[kind(st)] += 1
    if p == r:
        continue
    ps = p[1:].replace("]", ";]").split(";")
    for k, st in enumerate(rs):
        if k >= len(ps) or ps[k] != st:
            kd = kind(st)
            first[kd] += 1
            got = ps[k] if k < len(ps) else "<missing>"
            # which part of the step broke: the restated operation, the work, or the value
            hp, hr = got.split("=")[0].split(":")[0], st.split("=")[0].split(":")[0]
            part = "restating the operation" if hp != hr else ("the work" if ":" in st and got.split("=")[0] != st.split("=")[0] else "the value")
            wrong_part[(kd, part)] += 1
            if len(examples[kd]) < n_show:
                examples[kd].append((exprs[i], st, got))
            break
print(f"answer exact (vs true answers): {np.mean([ok_ans[lv == l].mean() for l in range(7)]):.4f}   by level: " + " ".join(f"{ok_ans[lv == l].mean():.3f}" for l in range(7)))
print(f"whole solution identical to reference: {np.mean([ok_trace[lv == l].mean() for l in range(7)]):.4f}")
print(f"\n{'first wrong step':34s} {'errors':>7s} {'steps of this kind':>19s} {'error rate':>11s}")
for kd, n in first.most_common():
    print(f"{kd:34s} {n:7d} {total[kd]:19d} {n / total[kd]:11.4f}")
print("\nwhat broke inside the step:")
for (kd, part), n in wrong_part.most_common(14):
    print(f"  {kd:32s} {part:26s} {n}")
keys = [(int(l), subtype(int(l), e)) for l, e in zip(lv, exprs)]
print("\nsub-skills below 97%:")
for k in sorted(set(keys)):
    idx = np.array([kk == k for kk in keys])
    if ok_ans[idx].mean() < 0.97:
        print(f"  {mg.LEVELS[k[0]][:2]} {k[1]:36s} n={idx.sum():5d}  {ok_ans[idx].mean():.3f}")
for kd in examples:
    for e, want, got in examples[kd]:
        print(f"\n[{kd}] {e}\n   want {want[:150]}\n   got  {got[:150]}")
json.dump({"preds": preds, "refs": refs}, open(path.replace(".pt", ".preds.json"), "w"))
# long division: how often is a division the first thing to go wrong, by divisor length
div_len = collections.defaultdict(lambda: [0, 0])
num = r"\(?-?[0-9.]+\)?"
for p, r in zip(preds, refs):
    ps, rs_ = p[1:].split(";"), r[1:].split(";")
    for k, st in enumerate(rs_):
        bad = k >= len(ps) or ps[k] != st
        m = re.match(rf"^{num}/({num}):", st)
        if m:
            n_dig = len(m.group(1).strip("()-").replace(".", "").lstrip("0").rstrip("0") or "0")
            div_len[n_dig][1] += 1
            div_len[n_dig][0] += bad
        if bad:
            break
# roots: does the answer survive a starting estimate that differs from the reference one?
root = {"est_same": [0, 0], "est_diff": [0, 0]}
for i, (p, r) in enumerate(zip(preds, refs)):
    if exprs[i].startswith("sqrt(") and lv[i] == 4:
        key = "est_same" if p.split(";")[0] == r.split(";")[0] else "est_diff"
        root[key][1] += 1
        root[key][0] += int(ok_ans[i])
print("\ndivision as first error, by divisor digits:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(div_len.items())})
print("L5 sqrt problems, answer right when the model's estimate equals / differs from the reference:", root)
json.dump({"first": dict(first), "total": dict(total), "div_len": {str(k): v for k, v in div_len.items()}, "root": root,
           "sub": {f"{k[0]}|{k[1]}": float(ok_ans[np.array([kk == k for kk in keys])].mean()) for k in sorted(set(keys))}, "part": {f"{k[0]}|{k[1]}": v for k, v in wrong_part.items()},
           "answer_exact": float(np.mean([ok_ans[lv == l].mean() for l in range(7)])),
           "trace_exact": float(np.mean([ok_trace[lv == l].mean() for l in range(7)])),
           "examples": {k: v for k, v in examples.items()}}, open(path.replace(".pt", ".errs.json"), "w"))
