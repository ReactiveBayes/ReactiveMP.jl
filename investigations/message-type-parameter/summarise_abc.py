# Tables for the five variants from micro_abc.tsv, model_abc.tsv and spec_abc.tsv: for each case
# the median over rounds of each round's minimum, the spread of those minima, and X/A ratios.
import statistics as st, collections, sys
VARS = ['A', 'B', 'C', 'D', 'E']

def load(path, micro):
    d = collections.defaultdict(list); order = []
    for line in open(path):
        p = line.rstrip('\n').split('\t')
        if len(p) < 7: continue
        if micro:
            v, tag, name, tmin, tmed, allocs, byts = p
        else:
            v, tag, model, what, tmin, tmed, byts = p; name = model + ' ' + what; allocs = ''
        if name not in order: order.append(name)
        d[(name, v)].append((float(tmin), allocs, float(byts)))
    return d, order

def fmt(x, micro):
    if micro:
        return f"{x/1e6:.2f} ms" if x >= 1e6 else (f"{x/1e3:.2f} µs" if x >= 1e3 else f"{x:.0f} ns")
    return f"{x*1e3:.1f} ms" if x < 10 else f"{x:.2f} s"

for path, micro in (('micro_abc.tsv', True), ('model_abc.tsv', False)):
    d, order = load(path, micro)
    print(f"\n## {path}  (median of per-round minima; spread = max/min of the minima; allocs from round 1)\n")
    print("| case | " + " | ".join(VARS) + " | B/A | C/A | D/A | E/A | spread A,C | allocs A/B/C/D/E |")
    print("|---" * (len(VARS) + 7) + "|")
    for n in order:
        if not all(d.get((n, v)) for v in VARS): continue
        med = {v: st.median(x[0] for x in d[(n, v)]) for v in VARS}
        spread = {v: max(x[0] for x in d[(n, v)]) / max(min(x[0] for x in d[(n, v)]), 1e-12) for v in VARS}
        allocs = "/".join(d[(n, v)][0][1] if micro else f"{d[(n, v)][0][2]/2**20:.0f}M" for v in VARS)
        a = med['A'] or 1e-12
        print(f"| {n} | " + " | ".join(fmt(med[v], micro) for v in VARS) +
              f" | {med['B']/a:.2f} | {med['C']/a:.2f} | {med['D']/a:.2f} | {med['E']/a:.2f} | {spread['A']:.3f},{spread['C']:.3f} | {allocs} |")

try:
    rows = collections.defaultdict(list)
    for line in open('spec_abc.tsv'):
        p = line.rstrip('\n').split('\t')
        if len(p) >= 4 and p[2] == 'first-infer-s':
            rows[(p[1] + ' compile s', p[0])].append(float(p[5]))
        elif len(p) == 4 and p[2] == 'specialisations':
            rows[(p[1] + ' specialisations', p[0])].append(float(p[3]))
    names = []
    for (n, v) in rows:
        if n not in names: names.append(n)
    print("\n## spec_abc.tsv (median over 2 runs)\n\n| | " + " | ".join(VARS) + " | C/A |\n|---" * 1 + "|---" * (len(VARS) + 1) + "|")
    for n in names:
        med = {v: st.median(rows[(n, v)]) for v in VARS if rows.get((n, v))}
        if len(med) < len(VARS): continue
        print(f"| {n} | " + " | ".join(f"{med[v]:.2f}" if med[v] < 100 else f"{med[v]:.0f}" for v in VARS) + f" | {med['C']/med['A']:.3f} |")
except FileNotFoundError:
    pass
