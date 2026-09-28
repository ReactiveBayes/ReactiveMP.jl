#!/usr/bin/env python3
"""Summarise driver.sh output: the median over rounds of each round's value (bench_models.jl
already records a per-round minimum), per model, variant and stage; and the same for the micro
suite. Writes Markdown tables and a JSON file the report page reads.
    python3 summarise.py <outdir>
"""
import sys, json, statistics, collections, os

out = sys.argv[1]
ORDER = ["v6", "A", "P12", "E4", "FULL", "FULLT", "FULLPC", "FULLTPC"]

def med(xs):
    xs = [x for x in xs if x == x]
    return statistics.median(xs) if xs else float("nan")

# bench_models writes infer(I=it1) and infer(I=2it1); keep them apart by their label
models2 = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(os.path.join(out, "models.tsv")):
    f = line.rstrip("\n").split("\t")
    if len(f) < 7:
        continue
    v, tag, model, stage, tmin, tmed, b = f[:7]
    gc = float(f[7]) if len(f) > 7 else float("nan")
    models2[(model, stage)][v].append((float(tmin), float(b), gc))

summary = {}
for (model, stage), byv in sorted(models2.items()):
    summary.setdefault(model, {})[stage] = {
        v: {"t": med([x[0] for x in xs]), "bytes": med([x[1] for x in xs]), "gc": med([x[2] for x in xs]), "n": len(xs)}
        for v, xs in byv.items()
    }

micro = collections.defaultdict(lambda: collections.defaultdict(list))
mpath = os.path.join(out, "micro.tsv")
if os.path.exists(mpath):
    for line in open(mpath):
        f = line.rstrip("\n").split("\t")
        if len(f) < 7:
            continue
        v, tag, case, tmin, tmed, allocs, b = f[:7]
        micro[case][v].append((float(tmin), float(allocs), float(b)))
msum = {case: {v: {"ns": med([x[0] for x in xs]), "allocs": med([x[1] for x in xs]), "bytes": med([x[2] for x in xs])} for v, xs in byv.items()} for case, byv in micro.items()}

json.dump({"models": summary, "micro": msum}, open(os.path.join(out, "summary.json"), "w"), indent=1)

def fmt_t(t):
    if t != t:
        return "—"
    if t >= 1:
        return f"{t:.2f} s"
    if t >= 1e-3:
        return f"{t*1e3:.1f} ms"
    return f"{t*1e6:.0f} µs"

lines = ["| model | stage | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 2) + "|"]
for model in summary:
    for stage in summary[model]:
        if stage in ("load",):
            continue
        row = summary[model][stage]
        lines.append(f"| {model} | {stage} | " + " | ".join(fmt_t(row[v]["t"]) if v in row else "—" for v in ORDER) + " |")
lines += ["", "| micro case | " + " | ".join(ORDER[1:]) + " |", "|---" * len(ORDER) + "|"]
for case in msum:
    row = msum[case]
    lines.append(f"| {case} | " + " | ".join((f"{row[v]['ns']:.0f} ns / {row[v]['allocs']:.0f}" if v in row else "—") for v in ORDER[1:]) + " |")
open(os.path.join(out, "tables.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
