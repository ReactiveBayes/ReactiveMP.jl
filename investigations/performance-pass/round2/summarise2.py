# Summarises bench_models2.jl's raw samples: every figure a minimum.
#   python3 summarise2.py <models.tsv> <reference variant> [<variant>...]
# Per model and variant:
# - T(I), T(2I): the minimum over every sample of every round, of wall time, of wall time less GC
#   time (`nogc`), and of thread CPU time;
# - per iteration: (min nogc T(2I) - min nogc T(I)) / I, both minima of GC-free-equivalent time, so
#   the estimate does not depend on where a pause fell; setup: min nogc T(I) - I * per iteration;
# - ttfx: the minimum over rounds of the first `infer`; bytes: of T(I), identical across samples;
# - ratio to the reference: min over rounds of (variant's round minimum / reference's round
#   minimum), with its range over rounds, of T(I) wall; a ratio whose range excludes 1 is marked.
import sys, csv, collections

path, ref = sys.argv[1], sys.argv[2]
variants = sys.argv[3:]
rows = collections.defaultdict(list)  # (variant, model, iters) -> [(round, wall, cpu, gc, bytes, pauses, full)]
for r in csv.reader(open(path), delimiter='\t'):
    v, tag, model, iters, k = r[0], r[1], r[2], int(r[3]), int(r[4])
    wall, cpu, gc, b, p, f = float(r[5]), float(r[6]), float(r[7]), float(r[8]), int(r[9]), int(r[10])
    rows[(v, model, iters)].append((tag, wall, cpu, gc, b, p, f))

models = []
for (v, m, it) in rows:
    if m not in models:
        models.append(m)
allvariants = [ref] + [v for v in variants if v != ref] if variants else sorted({v for (v, _, _) in rows})

def iters_of(v, m):
    return sorted({it for (vv, mm, it) in rows if vv == v and mm == m and it > 0})

def mins(v, m, it):
    s = rows.get((v, m, it), [])
    if not s:
        return None
    return dict(wall=min(x[1] for x in s), nogc=min(x[1] - x[3] for x in s), cpu=min(x[2] for x in s),
                bytes=min(x[4] for x in s), pauses=min(x[5] for x in s))

def round_mins(v, m, it):
    d = collections.defaultdict(lambda: float('inf'))
    for x in rows.get((v, m, it), []):
        d[x[0]] = min(d[x[0]], x[1])
    return d

def fmt(t):
    if t is None:
        return '—'
    return f"{t * 1e3:.2f} ms" if t < 1 else f"{t:.2f} s"

def summary(v, m):
    its = iters_of(v, m)
    out = {}
    first = mins(v, m, 0)
    out['ttfx'] = first['wall'] if first else None
    if not its:
        return out
    I1 = its[0]
    a = mins(v, m, I1)
    out.update(I1=I1, T1=a['wall'], T1nogc=a['nogc'], T1cpu=a['cpu'], bytes1=a['bytes'], pauses1=a['pauses'])
    if len(its) > 1:
        I2 = its[1]; b = mins(v, m, I2)
        out.update(I2=I2, T2=b['wall'], T2nogc=b['nogc'], pauses2=b['pauses'])
        per = (b['nogc'] - a['nogc']) / (I2 - I1)
        out.update(per=per, setup=a['nogc'] - I1 * per, bytesper=(b['bytes'] - a['bytes']) / (I2 - I1))
    return out

def paired(v, m):
    its = iters_of(v, m)
    if not its:
        return None
    a, b = round_mins(v, m, its[0]), round_mins(ref, m, its[0])
    rs = [a[t] / b[t] for t in a if t in b]
    return (min(rs), sorted(rs)[len(rs) // 2], max(rs)) if rs else None

cols = ['T(I)', 'T(I) no GC', 'T(2I) no GC', 'per iteration (no GC)', 'setup (no GC)', 'bytes T(I)', 'first inference']
print('| model | variant | I | ' + ' | '.join(cols) + ' | T(I) / ' + ref + ' per round: min, median, max |')
print('|' + '---|' * (len(cols) + 4))
for m in models:
    for v in allvariants:
        s = summary(v, m)
        if not s:
            continue
        pr = paired(v, m) if v != ref else None
        mark = ''
        if pr:
            mark = f"{pr[0]:.3f}, {pr[1]:.3f}, {pr[2]:.3f}" + (' **faster**' if pr[2] < 1 else (' **slower**' if pr[0] > 1 else ''))
        print(f"| {m} | {v} | {s.get('I1', '—')} | {fmt(s.get('T1'))} | {fmt(s.get('T1nogc'))} | {fmt(s.get('T2nogc'))} | "
              f"{fmt(s.get('per'))} | {fmt(s.get('setup'))} | {s.get('bytes1', 0) / 1e6:.1f} MB | {fmt(s.get('ttfx'))} | {mark} |")
