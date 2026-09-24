# The v6/v7 comparison environment

A pinned environment holding **`ReactiveMP@6.5.0` from the registry** — deliberately not
this checkout, which becomes v7 and would collide by name.

## What it is for

Phase 4's migration checker runs a v6 rule and a v7 rule on identical inputs and asserts
they agree. `MIGRATION.md`'s before/after doctests will execute here too, once Phase 5 writes it. Both work in a single
process because the new rule packages are differently named and do not depend on
ReactiveMP, so v6 and they coexist.

What does **not** work here is ReactiveMP v7 against v6 — same package name, same UUID, and
Julia loads one version of a package per process. Engine-level comparisons (scheduling,
free-energy trajectories, retained values) therefore need separate pinned processes or
fixtures recorded from full v6 runs. That is why the Phase 4 checker must *capture* results
rather than only assert equality: Phases 4.5 and 7 consume those recordings.

## The Julia version

**The committed `Manifest.toml` is resolved on Julia 1.13**, the version all work targets
for now (Phase 4.5 step 4; `DISCUSSION.md` §3.22). Until then it was pinned to 1.10.12, the old
floor, because a manifest resolved on a newer Julia can select versions an older one cannot
load. ReactiveMP 6.5.0 and RxInfer 5.5.2 run on 1.13, and the engine fixtures recorded on
1.10.12 reproduce there exactly (`record_engine_fixtures.jl --check`).

Regenerate it, developing the five lib packages the comparisons load in one call so Pkg never
resolves with only some of them (`DeltaMessagePassingRules` joined in Phase 6 step 2, with
`compare_delta.jl`):

```bash
julia --startup-file=no --project=compat/v6-comparison -e '
using Pkg
Pkg.develop([PackageSpec(path = joinpath("lib", p)) for p in ("MessagePassingRulesBase", "MessagePassingRulesTestUtils", "StandardMessagePassingRules", "MessagePassingRulesApproximations", "DeltaMessagePassingRules")])
Pkg.instantiate()'
```

Then confirm that ReactiveMP comes from the registry:

```bash
julia --startup-file=no --project=compat/v6-comparison -e '
using ReactiveMP
println(pkgversion(ReactiveMP), "  ", pathof(ReactiveMP))'
```

The path must be under `~/.julia/packages/`. If it points into this repository, something
has dev-linked the local copy and the comparison is measuring v7 against itself.

`Pkg.develop` resolves its path against the working directory, not the activated project,
so run this from the repository root. A lib package a comparison starts to load joins the
same single `develop` call.

## Engine fixtures (Phase 4.5)

Phase 4.5 replaces v6's rule-call and node-creation paths rather than bridging to them, so
before any v6 code is deleted this environment records engine-level fixtures from full v6
runs, through **RxInfer 5.5.2** (pinned, `=5.5.2`; it accepts ReactiveMP 6.5):

```bash
julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl          # record
julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl --check  # compare
julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl delta_unscented_static  # only the named models
```

The fixtures are `fixtures/engine/<model>.toml`, one per model (the seven slice models,
`delta_unscented_static` from Phase 4.5 case (d), `logic_bp` from Phase 5 step 4 and
`mixture_bp` from step 8), written by
`MessagePassingRulesTestUtils.save_engine_fixture`. Each holds the free energy per iteration,
the final posteriors, and every message-rule call in the order v6 made it, which is
materialisation order, with its result and log scale. They are **TOML, not `Serialization`**,
so ReactiveMP's tests can read them on a Julia minor other than the one that recorded them
(recorded on 1.10.12, read on 1.13). `--check` re-records the fixtures and compares them with
the committed files; it is run locally, since the `v6-comparison` CI job still targets 1.10
and is stale until the workflows are updated (`PHASES.md` § Phase 7). The header's
`notes` say what the recording could not capture: log scales are recorded only where v6
produces them, which is `bp_iid` alone. See `PHASES.md` § Phase 4.5, Step 0.

## The standing constraint

This environment resolves only while shared dependencies stay inside v6.5.0's caret bounds
— `BayesBase = "1.5"`, `ExponentialFamily = "2.5.0"`. New **minor** versions are fine.

`PLAN.md` instructs that anything the base package needs be **added to BayesBase**. Those
additions must therefore ship as non-breaking 1.x releases. A BayesBase 2.0 released for
this rewrite would make this environment unresolvable and silently cost us the main
instrument for verifying 490 ported rules. If a breaking BayesBase change becomes
unavoidable, redesign this harness first, not afterwards.

## The migration checker, and the comparisons

These run here and are verified locally (no CI runs until a PR):

```bash
julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/check.jl
julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_standard.jl
julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/compare_approximations.jl
julia --startup-file=no --project=compat/v6-comparison compat/v6-comparison/slice_rule_inventory.jl
```

- `check.jl`: the Phase 4 checker's own demonstration, and v6 rules verified against their
  node definitions.
- `compare_standard.jl`: every rule ported into `StandardMessagePassingRules` against its v6
  original, with #669 declared as a correction.
- `compare_approximations.jl`: `MessagePassingRulesApproximations` against v6's
  `src/approximations/`.
- `slice_rule_inventory.jl`: every v6 rule the slice models select, with its declared types.

How the checker works:

- `V6Oracle.jl` calls a v6 rule from the inputs a v7 rule takes, and returns its result and
  log scale. It is the only code in the repository that names ReactiveMP v6's internals.
- Ported rules are compared with their v6 originals through `compare_with_reference` from
  `MessagePassingRulesTestUtils`. An undeclared disagreement fails; a declared one must say
  whether it is a `:migration_bug` or a `:correction`, and why.
- v6 rules are verified against their own node definitions with `verify_message_update`.
  Failures there are findings about v6, pinned in `KNOWN_V6_FINDINGS` with their
  explanation, so a new one fails the run until it is understood.

The disposition inventory runs here too, `julia --project=compat/v6-comparison
scripts/inventory.jl --check`: it records where everything in ReactiveMP 6.5.0 goes, and since
Phase 4.5 step 4 only v6.5.0 still has all of it. The root suite's `:quality` item runs it.
The engine fixtures in `fixtures/engine/` are what `test/engine/` in the root suite compares
the new engine with.

The four `lib/` packages the comparisons load are dev'd into this environment by relative path and recorded in the
committed manifest, resolved on 1.13.
