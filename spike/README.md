# `spike/` — Phase 0, throwaway

Hand-written, no macros. This directory exists to answer the questions in `PHASES.md`
§ Phase 0 before the macro surface is built, and **is deleted in the commit that closes
Phase 0**. The findings survive in `DISCUSSION.md` §3.15; the code does not.

It is committed while it lives so that the measurements are reproducible and reviewable
rather than being numbers someone once saw in a terminal.

## Layout

Two environments, because the dispatch gate wants a clean process with a fast load time and
the semantics work needs the live v6 engine.

| file | Phase 0 criterion it answers |
|---|---|
| `dispatch/01_machinery.jl` | targets, algorithms, context, annotations sink, keyed `args`, `RuleSpec`, `find_rule` |
| `dispatch/02_rules.jl` | ten representative rules written by hand in the decided syntax |
| `dispatch/03_devirt.jl` | **the gate** — inference and allocations through the `RuleSpec` |
| `dispatch/04_fallback.jl` | rule-fallback contract, and the ruleset-axis decision (open item #4) |
| `dispatch/05_allocate.jl` | one worked `preallocate` lowering, end to end |
| `dispatch/06_services.jl` | the two hard context services (open item #12) |
| `dispatch/07_measure.jl` | cold, warm, allocations, specialization growth |
| `semantics/08_layouts.jl` | the four delta layouts expressed as dependency declarations |
| `semantics/09_execution.jl` | execution semantics: emissions *and* numbers against v6 |

`results/` holds captured output, one file per Julia version.

## Running

```bash
julia +1.10 --startup-file=no --project=spike/dispatch  spike/dispatch/03_devirt.jl
julia +1.13 --startup-file=no --project=spike/dispatch  spike/dispatch/03_devirt.jl
julia +1.10 --startup-file=no --project=spike/semantics spike/semantics/09_execution.jl
```

`--startup-file=no` is not optional here: a personal `~/.julia/config/startup.jl` loading
packages that are not in this environment will abort the run before anything is measured.

Measurements are taken on the **1.10 floor** and on 1.13. A gate that passes only on the
newest Julia says nothing about the version this package supports — and in this case the two
disagree, which is the whole argument for measuring on the floor.

The manifests are committed (`git add -f`, since `**/Manifest.toml` is ignored) and resolved
on **1.10**. Running on 1.13 reuses that resolution, under which JET does not precompile; the
gate reports its JET check as skipped rather than failing, because the routing assertions
matter more than the JET pass.

## Measuring allocations: four ways to get it wrong

All four were hit while building `03_devirt.jl`, and each silently changes the answer:

| mistake | effect |
|---|---|
| measuring against a non-`const` global | `+16` bytes of boxing |
| a varargs helper that splats, `f(xs...)` | `+48` bytes of its own, in every figure |
| constructing the spec inline in an inlinable `find_rule` | constant-folded away — `0` for *every* representation |
| closing over the node in a loop, so it is a `DataType` rather than `Type{Node}` | the call goes dynamic and reports `Any` |

The gate caught the first two by failing. That is what it is for.

## Two things the spike deliberately does not do

- **It does not implement macros.** Everything here is what a macro would *lower to*,
  written out by hand. Body slot selection — `PLAN.md`'s `(output, algo, ctx, args, ann,
  node)`, where a body names only the slots it wants — is therefore not modelled at runtime:
  it is a macro-expansion-time concern. The macro wraps the user's lambda in a full-arity
  adapter closure at definition time, so nothing at run time depends on which slots were
  named. Bodies here take the full canonical list.
- **It does not aim to be correct numerically** except where a criterion says so
  (`09_execution.jl`). Rule bodies elsewhere are shaped like the real ones so that dispatch,
  inference and allocation behaviour are representative; they are not a port.
