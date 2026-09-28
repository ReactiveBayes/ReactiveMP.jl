#!/usr/bin/env bash
# GraphPPL model-creation comparison: the baseline (P12: P1+P2, GraphPPL 4.8.0) and the cumulative
# GraphPPL variants Ga (C1), Gb (C1-C3), Gc (C1-C4), G1 (C1-C6). Fresh process per run; the
# variant order rotates each round. VROOT is the variants' directory.
set -u
cd "$(dirname "$0")"
: "${VROOT:?set VROOT}"
OUT=${OUT:-results/graphppl_compare.tsv}
rounds=("P12 Ga Gb Gc G1" "G1 Gc Gb Ga P12")
sizes() {
  case "$1" in
    nl) echo "300 1000 3000" ;;
    mat) echo "1024 10000 40000" ;;
    *) echo "1000 10000" ;;
  esac
}
r=0
for vs in "${rounds[@]}"; do
  r=$((r+1))
  for m in ssm1 iid hmm nl ssmsub mat; do
    for v in $vs; do
      julia --startup-file=no --project="$VROOT/$v/env" bench_creation.jl "$v" "round$r" "$OUT" "$m" $(sizes $m) > "logs/creation_${v}_${m}_$r.log" 2>&1
    done
  done
done
echo DONE
