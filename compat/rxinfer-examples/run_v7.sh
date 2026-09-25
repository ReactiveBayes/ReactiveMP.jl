#!/usr/bin/env bash
# Runs `<name>_v7.jl` in this directory's environment (RxInfer's v7 branch).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec julia --startup-file=no --project="$here" "$here/$1_v7.jl"
