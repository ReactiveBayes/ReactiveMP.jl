#!/usr/bin/env bash
# Interleaved A/B of RxInfer-level variants: P12 (ReactiveMP P1+P2, RxInfer HEAD) vs X1 (the same
# + RxInfer P6). Fresh process per run, order alternating per round. Results: results/rxinfer_ab.tsv
cd "$(dirname "$0")"
S=/private/tmp/claude-501/-Users-bvdmitri-Projects-Julia-ReactiveMP-jl/928b8021-d60e-4193-8fce-2d309cac34fa/scratchpad
mkdir -p logs
for r in 1 2 3 4 5 6; do
  if [ $((r % 2)) -eq 1 ]; then order="P12 X1"; else order="X1 P12"; fi
  for v in $order; do
    BENCH_ROUNDS=1 julia --startup-file=no --project=$S/v/$v/env bench_rxinfer_opts.jl $v round$r results/rxinfer_ab.tsv > logs/rxinfer_ab_${v}_$r.log 2>&1
  done
done
echo DONE
