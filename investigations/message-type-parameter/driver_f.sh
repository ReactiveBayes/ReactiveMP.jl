#!/bin/bash
# Second batch: A (HEAD), C (B + barriers), E (C + lazy events), F (A + lazy events + constructor barriers).
# Fresh process per run; the order rotates each round so no variant always runs first or last.
cd "$(dirname "$0")"
set -x
rounds=("A C E F" "F E C A" "C F A E")
r=0
for vs in "${rounds[@]}"; do
  r=$((r+1))
  for v in $vs; do julia --project=env$v bench_micro.jl run$r micro_f.tsv > logs_f/micro_${v}_$r.log 2>&1; done
done
r=0
for vs in "${rounds[@]}"; do
  r=$((r+1))
  for m in ssm1 ssm2 iid nl hmm; do
    for v in $vs; do julia --project=env$v bench_models.jl $m run$r model_f.tsv post_f > logs_f/model_${v}_${m}_$r.log 2>&1; done
  done
done
for vs in "A C E F" "F E C A"; do
  for v in $vs; do julia --project=env$v spec_count.jl spec_f.tsv > logs_f/spec_$v.log 2>&1; done
done
echo DONE
