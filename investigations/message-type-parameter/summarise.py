# Tables from micro_results.tsv and model_results.tsv: per case, the median across runs of
# each run's minimum (and the spread of those minima), A vs B, and B/A.
import sys, statistics as st, collections
def load(path):
    d = collections.defaultdict(list)
    for line in open(path):
        p = line.rstrip('\n').split('\t')
        if len(p) < 7: continue
        if path.endswith('micro_results.tsv'):
            v, tag, name, tmin, tmed, allocs, byts = p
        else:
            v, tag, model, what, tmin, tmed, byts = p; name = model + ' ' + what; allocs = ''
        d[(name, v)].append((float(tmin), float(tmed), allocs, float(byts)))
    return d
def fmt(x, unit):
    if unit == 'ns':
        return f"{x/1e6:.2f} ms" if x > 1e6 else (f"{x/1e3:.2f} µs" if x > 1e3 else f"{x:.0f} ns")
    return f"{x*1e3:.1f} ms" if x < 10 else f"{x:.2f} s"
for path, unit in (('micro_results.tsv', 'ns'), ('model_results.tsv', 's')):
    try: d = load(path)
    except FileNotFoundError: continue
    names = []
    for (n, v) in d:
        if n not in names: names.append(n)
    print(f"\n## {path}\n| case | A min (median of runs; range) | B min | B/A | A allocs/bytes | B allocs/bytes |\n|---|---|---|---|---|---|")
    for n in names:
        A, B = d.get((n, 'A'), []), d.get((n, 'B'), [])
        if not A or not B: continue
        am = st.median(x[0] for x in A); bm = st.median(x[0] for x in B)
        ar = f"{fmt(min(x[0] for x in A), unit)}–{fmt(max(x[0] for x in A), unit)}"
        br = f"{fmt(min(x[0] for x in B), unit)}–{fmt(max(x[0] for x in B), unit)}"
        ab = f"{A[0][2]} / {int(min(x[3] for x in A))}"; bb = f"{B[0][2]} / {int(min(x[3] for x in B))}"
        print(f"| {n} | {fmt(am, unit)} ({ar}) | {fmt(bm, unit)} ({br}) | {bm/am:.3f} | {ab} | {bb} |")
