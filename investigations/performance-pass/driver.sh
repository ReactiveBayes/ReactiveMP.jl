#!/usr/bin/env bash
# The final, sequential measurement: every variant in a fresh process, the order rotating each
# round so that no variant always runs first or last. Nothing else should run on the machine.
#   VROOT=<variants dir> ./driver.sh <outdir> "<variants>" [rounds]
# Variants are directories under $VROOT with an `env/`; `v6` is RxInfer 5.5.2 over ReactiveMP 6.5.0.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$1"; read -ra variants <<< "$2"; rounds="${3:-3}"
: "${VROOT:?set VROOT}"
mkdir -p "$out/logs" "$out/post"
models=(ssm1 ssm2 iid betabern nl hmm gmm linreg filter ssm1+defaults iid+defaults)
scaling=(ssm1@100 ssm1@10000 iid@100 iid@10000 iid@100000 betabern@50000)
rotate() { local n=$1; shift; local a=("$@"); local k=$(( n % ${#a[@]} )); echo "${a[@]:k}" "${a[@]:0:k}"; }
for r in $(seq 1 "$rounds"); do
  read -ra order <<< "$(rotate "$r" "${variants[@]}")"
  for m in "${models[@]}" "${scaling[@]}"; do
    for v in "${order[@]}"; do
      # the micro suite needs the v7 API
      julia --startup-file=no --project="$VROOT/$v/env" "$here/bench_models.jl" "$v" "$m" "run$r" "$out/models.tsv" "$out/post" \
        > "$out/logs/model_${v}_${m}_$r.log" 2>&1 || echo "FAILED $v $m $r" >> "$out/failures.txt"
    done
  done
  for v in "${order[@]}"; do
    [ "$v" = "v6" ] && continue
    julia --startup-file=no --project="$VROOT/$v/env" "$here/bench_micro.jl" "$v" "run$r" "$out/micro.tsv" \
      > "$out/logs/micro_${v}_$r.log" 2>&1 || echo "FAILED micro $v $r" >> "$out/failures.txt"
  done
done
echo DONE > "$out/DONE"
