#!/bin/bash
# Alternates A and B: micro runs A B B A A B, then models per rep with alternating order.
cd "$(dirname "$0")"
set -x
order=(A B B A A B)
for i in 0 1 2 3 4 5; do
  v=${order[$i]}; julia --project=env$v bench_micro.jl run$((i/2+1)) micro_results.tsv > logs/micro_${v}_$i.log 2>&1
done
for rep in 1 2 3; do
  for m in ssm1 ssm2 iid nl hmm; do
    if [ $((rep % 2)) -eq 1 ]; then vs="A B"; else vs="B A"; fi
    for v in $vs; do
      julia --project=env$v bench_models.jl $m run$rep model_results.tsv post > logs/model_${v}_${m}_$rep.log 2>&1
    done
  done
done
echo DONE
