"""Mean number of target tokens per training row of each pool, counted over every row.
(train2.py estimated this from the head of each length bucket, which is all level 1 when a pool has few buckets.)
Usage: token_stats.py <out.json> <data file>..."""
import json, sys
import numpy as np

out = {}
for path in sys.argv[2:]:
    Z = np.load(path)
    rows = tok = 0
    for b in range(len(Z["widths"])):
        T = Z[f"T{b}"]
        rows += len(T)
        tok += int(np.count_nonzero(T))
    out[path] = tok / rows
    print(path, rows, round(out[path], 3), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
