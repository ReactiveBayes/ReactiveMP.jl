#!/usr/bin/env bash
# Round 2's driver: for every model, the variants run back to back in fresh processes, one round
# after another, the order rotating each round, so that a ratio is always between samples taken
# minutes apart (machine load cancels in the pair). One Julia thread, one BLAS thread, one process
# at a time.
#   VROOT=<variants dir> ./driver2.sh <outdir> "<variants>" [rounds] ["<models>"]
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="$1"; read -ra variants <<< "$2"; rounds="${3:-5}"
default_models="ssm1 ssm2 iid betabern nl hmm gmm linreg filter ssm1+defaults iid+defaults ssm1@100 ssm1@10000 iid@100 iid@10000 iid@100000 betabern@50000"
read -ra models <<< "${4:-$default_models}"
: "${VROOT:?set VROOT}"
mkdir -p "$out/logs" "$out/post"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
rotate() { local n=$1; shift; local a=("$@"); local k=$(( n % ${#a[@]} )); echo "${a[@]:k}" "${a[@]:0:k}"; }
for m in "${models[@]}"; do
  for r in $(seq 1 "$rounds"); do
    read -ra order <<< "$(rotate "$r" "${variants[@]}")"
    for v in "${order[@]}"; do
      julia --startup-file=no --project="$VROOT/$v/env" "$here/bench_models2.jl" "$v" "$m" "run$r" "$out/models.tsv" "$out/post" \
        > "$out/logs/model_${v}_${m}_$r.log" 2>&1 || echo "FAILED $v $m $r" >> "$out/failures.txt"
    done
  done
  echo "done $m" >> "$out/progress.txt"
done
echo DONE > "$out/DONE"
