#!/usr/bin/env bash
# Round 2's variant builder: sibling git worktrees of ReactiveMP.jl, RxInfer.jl, Rocket.jl and
# GraphPPL.jl under $VROOT/<name>/, taken from clones under $SRC at the round-1 revisions, with the
# given diffs applied in order, and an environment $VROOT/<name>/env developing them (../mkenv.jl).
#   VROOT=<dir> SRC=<clones dir> ./mkvariant2.sh <name> <diff>...
# A diff's package is the suffix of its file name, `<anything>-<Pkg>.diff` (ReactiveMP, RxInfer,
# Rocket or GraphPPL). The clones' HEADs must be the revisions recorded in ../README.md
# (ReactiveMP d50fac9c, RxInfer 4ca9eb88, Rocket eba18dfc, GraphPPL 4cc7c6c).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
name="$1"; shift
: "${VROOT:?set VROOT}"; : "${SRC:?set SRC}"
rev() { case "$1" in ReactiveMP) echo d50fac9c ;; RxInfer) echo 4ca9eb88 ;; Rocket) echo eba18dfc ;; GraphPPL) echo 4cc7c6c ;; esac; }
d="$VROOT/$name"
mkdir -p "$d"
for pkg in ReactiveMP RxInfer Rocket GraphPPL; do
  [ -d "$d/$pkg.jl" ] || git -C "$SRC/$pkg.jl" worktree add -q --detach "$d/$pkg.jl" "$(rev "$pkg")"
done
for f in "$@"; do
  base="$(basename "$f" .diff)"; pkg="${base##*-}"
  case "$pkg" in ReactiveMP|RxInfer|Rocket|GraphPPL) ;; *) echo "no package in $f" >&2; exit 1 ;; esac
  git -C "$d/$pkg.jl" apply --3way "$f"; echo "applied $f to $pkg"
done
julia --startup-file=no "$here/../mkenv.jl" "$d/env" "$d"
