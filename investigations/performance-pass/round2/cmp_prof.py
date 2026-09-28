# Compares two prof_setup.jl outputs: inclusive milliseconds per function, ranked by difference.
#   python3 cmp_prof.py <a.tsv> <b.tsv> [n]
import sys
def load(p):
    d = {}; hdr = ''
    for l in open(p):
        if l.startswith('#'):
            hdr = l.strip(); continue
        k, c, pct, ms = l.rstrip('\n').split('\t'); d[k] = float(ms)
    return d, hdr
skip = ('C:jl_apply', 'C:ijl_toplevel', 'C:jl_eval_toplevel', 'C:jl_repl', 'C:jl_toplevel')
(a, ha), (b, hb) = load(sys.argv[1]), load(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 45
print(f"  {ha}\n  {hb}\n  a ms | b ms | diff")
keys = sorted(set(a) | set(b), key=lambda k: -abs(b.get(k, 0) - a.get(k, 0)))
for k in [k for k in keys if not k.startswith(skip)][:n]:
    print(f"{a.get(k, 0):7.2f} {b.get(k, 0):7.2f} {b.get(k, 0) - a.get(k, 0):+7.2f}  {k}")
