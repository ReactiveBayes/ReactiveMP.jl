#!/usr/bin/env bash
# Runs `<name>_v6.jl` in the pinned v6 environment, with v6-extras (StableRNGs only) stacked
# after it; the pinned environment itself is not modified.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
JULIA_LOAD_PATH="$here/../v6-comparison:$here/v6-extras:@stdlib" \
    exec julia --startup-file=no --project="$here/../v6-comparison" "$here/$1_v6.jl"
