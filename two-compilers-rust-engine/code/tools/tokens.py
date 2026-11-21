"""tokens.py [--json OUT]: for each side, what its agent had spent (tokens, time) when each official measurement
was taken. Reads the agents' transcripts (one usage record per model response, with a timestamp) and the
sides' log.jsonl."""
import datetime, glob, json, os, sys

H = "/home/dev/local/hill2"
SUB = "/path/to/the/session/subagents"  # where the agent harness keeps the two agents' transcripts
WANT = {"tc": "Hill-climb torch.compile on yolov3 (run 2)", "vm": "Hill-climb torch.vm on yolov3 (run 2)"}


def stamp(text):
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def transcript(side):
    for meta in glob.glob(os.path.join(SUB, "agent-*.meta.json")):
        if json.load(open(meta)).get("description") == WANT[side]:
            return meta.replace(".meta.json", ".jsonl")
    raise SystemExit(f"no transcript for {side}")


def usage(path):
    """(time, input, cache write, cache read, output) per response, in order; a response is written in several
    records with the same id, the last of which has its final counts."""
    last, first = {}, None
    for line in open(path):
        row = json.loads(line)
        if first is None and row.get("timestamp"):
            first = stamp(row["timestamp"])
        message = row.get("message")
        if row.get("type") == "assistant" and isinstance(message, dict) and message.get("usage"):
            u = message["usage"]
            last[message["id"]] = (stamp(row["timestamp"]), u.get("input_tokens", 0), u.get("cache_creation_input_tokens", 0),
                                   u.get("cache_read_input_tokens", 0), u.get("output_tokens", 0))
    return first, sorted(last.values())


def agent_minutes(t, start):
    """Minutes of the agent's time at moment T: the pause between the two halves is not its time."""
    try:
        first_end = float(open(f"{H}/phase1_deadline").read())
        second_start = float(open(f"{H}/phase2_start").read())
    except FileNotFoundError:
        return max(0.0, (t - start) / 60)
    if t <= first_end:
        return max(0.0, (t - start) / 60)
    return (first_end - start) / 60 + max(0.0, t - second_start) / 60


def spent(records, until):
    total = [0, 0, 0, 0]
    for t, *counts in records:
        if t <= until:
            total = [a + b for a, b in zip(total, counts)]
    return dict(zip(("input", "cache_write", "cache_read", "output"), total))


out = {}
for side in ("tc", "vm"):
    start, records = usage(transcript(side))
    rows = [json.loads(line) for line in open(os.path.join(H, side, "log.jsonl"))]
    waits = [(r["time"], r["lock_wait_s"]) for r in rows]
    for line in open(os.path.join(H, side, "gpu.log")) if os.path.exists(os.path.join(H, side, "gpu.log")) else []:
        parts = line.split()
        waits.append((stamp(parts[0]), float(parts[3])))
    points, best = [], 0.0
    for r in rows:
        if r["kind"] != "official":
            continue
        s = spent(records, r["time"])
        waited = sum(w for t, w in waits if t <= r["time"])
        counts = r["eligible"] and r["time"] >= start - 3600
        best = max(best, r["qps"]) if counts else best
        points.append({
            "note": r["note"], "utc": r["utc"], "qps": r["qps"], "eligible": r["eligible"], "best": best,
            "minutes": agent_minutes(r["time"], start), "minutes_not_waiting": max(0.0, agent_minutes(r["time"], start) - waited / 60),
            "half": 2 if os.path.exists(f"{H}/phase2_start") and r["time"] > float(open(f"{H}/phase2_start").read()) else 1,
            "tokens": s, "follows_weights": (r["probes"] or {}).get("follows_weight_update"),
            "changed": len(r["changed"] or []), "sha": r["sha"], "error_ratio": r["error_over_eager_error"],
        })
    end = records[-1][0] if records else start
    out[side] = {"start": start, "responses": len(records), "total": spent(records, float("inf")), "minutes": agent_minutes(end, start), "points": points}
    print(f"== {side}: {len(records)} responses over {agent_minutes(end, start):.0f} min of agent time; tokens so far {out[side]['total']}")
    for p in points:
        t = p["tokens"]
        print(f"  {p['utc'][11:19]}  {p['minutes']:6.1f} min  out {t['output'] / 1e3:7.1f}k  cache-read {t['cache_read'] / 1e6:7.1f}M  write {t['cache_write'] / 1e6:5.2f}M"
              f"  {p['qps'] or 0:7.0f} img/s  {'ok ' if p['eligible'] else 'NO '} weights {str(p['follows_weights'])[:5]:5s} files {p['changed']:2d}  {p['note'][:70]}")
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
