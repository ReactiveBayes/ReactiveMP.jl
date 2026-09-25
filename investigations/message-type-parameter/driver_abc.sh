#!/bin/bash
# A (HEAD), B (untyped), C (B + barriers), D (A + lazy events), E (C + lazy events).
# Fresh process per run; the order rotates each round so no variant always runs first or last.
cd "$(dirname "$0")"
set -x
rounds=("A B C D E" "E D C B A" "C A E B D")
r=0
for vs in "${rounds[@]}"; do
  r=$((r+1))
  for v in $vs; do julia --project=env$v bench_micro.jl run$r micro_abc.tsv > logs_abc/micro_${v}_$r.log 2>&1; done
done
r=0
for vs in "${rounds[@]}"; do
  r=$((r+1))
  for m in ssm1 ssm2 iid nl hmm; do
    for v in $vs; do julia --project=env$v bench_models.jl $m run$r model_abc.tsv post_abc > logs_abc/model_${v}_${m}_$r.log 2>&1; done
  done
done
for vs in "A B C D E" "E D C B A"; do
  for v in $vs; do julia --project=env$v spec_count.jl spec_abc.tsv > logs_abc/spec_$v.log 2>&1; done
done
echo DONE
