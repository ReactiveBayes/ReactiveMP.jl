#!/usr/bin/env bash
# Creates a variant: sibling git worktrees of ReactiveMP.jl, RxInfer.jl, Rocket.jl and GraphPPL.jl
# under $VROOT/<name>/, applies diffs/<name>-<pkg>.diff when present, and resolves an environment
# $VROOT/<name>/env that develops all of them.
#   VROOT=<dir> ./mkvariant.sh <name> [<base-variant>]
# RMP_REV, RX_REV default to the revisions recorded in the README; Rocket and GraphPPL come from
# clones under $VROOT/../src (master, i.e. Rocket 1.10.0 and GraphPPL 4.8.0 plus CI-only commits).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
name="$1"
: "${VROOT:?set VROOT}"
RMP="${RMP_SRC:-$HOME/Projects/Julia/ReactiveMP.jl}"
RX="${RX_SRC:-$HOME/Projects/Julia/RxInfer.jl}"
SRC="$VROOT/../src"
RMP_REV="${RMP_REV:-d50fac9c}"
RX_REV="${RX_REV:-4ca9eb88}"
d="$VROOT/$name"
mkdir -p "$d"
[ -d "$d/ReactiveMP.jl" ] || git -C "$RMP" worktree add -q --detach "$d/ReactiveMP.jl" "$RMP_REV"
[ -d "$d/RxInfer.jl" ]    || git -C "$RX" worktree add -q --detach "$d/RxInfer.jl" "$RX_REV"
[ -d "$d/Rocket.jl" ]     || git -C "$SRC/Rocket.jl" worktree add -q --detach "$d/Rocket.jl" HEAD
[ -d "$d/GraphPPL.jl" ]   || git -C "$SRC/GraphPPL.jl" worktree add -q --detach "$d/GraphPPL.jl" HEAD
IFS='+' read -ra parts <<< "$name"
for p in "${parts[@]}"; do
  for pkg in ReactiveMP RxInfer Rocket GraphPPL; do
    f="$here/diffs/$p-$pkg.diff"
    if [ -f "$f" ]; then git -C "$d/$pkg.jl" apply --3way "$f"; echo "applied $f"; fi
  done
done
julia --startup-file=no "$here/mkenv.jl" "$d/env" "$d"
