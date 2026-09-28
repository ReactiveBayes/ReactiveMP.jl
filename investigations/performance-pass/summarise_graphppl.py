# Summarises results/graphppl_compare.tsv: per (model, n, stage) the median over rounds of each
# round's minimum, for each variant, and the ratio to the baseline P12.
import sys, statistics, collections
path = sys.argv[1] if len(sys.argv) > 1 else "results/graphppl_compare.tsv"
order = ["P12", "Ga", "Gb", "Gc", "G1"]
d = collections.defaultdict(list)
for line in open(path):
    v, tag, m, n, stage, tmin, tmed, b = line.rstrip("\n").split("\t")
    d[(m, int(n), stage, v)].append((float(tmin), int(b)))
keys = sorted({k[:3] for k in d}, key=lambda k: (k[0], k[1], ["first_full", "graph0", "graph", "full"].index(k[2])))
print("| model | n | stage | " + " | ".join(order) + " | G1/P12 |")
print("|---|---|---|" + "---|" * (len(order) + 1))
for (m, n, stage) in keys:
    cells = []; base = None; last = None
    for v in order:
        xs = d.get((m, n, stage, v))
        if not xs: cells.append("–"); continue
        t = statistics.median(x[0] for x in xs)
        if v == "P12": base = t
        if v == "G1": last = t
        cells.append(("%.3g s" % t) if t >= 1 else ("%.3g ms" % (t * 1e3)))
    ratio = ("%.2f" % (last / base)) if base and last else "–"
    print(f"| {m} | {n} | {stage} | " + " | ".join(cells) + f" | {ratio} |")
