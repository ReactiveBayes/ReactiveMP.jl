# The v6/v7 comparison environment

A pinned environment holding **`ReactiveMP@6.5.0` from the registry** — deliberately not
this checkout, which becomes v7 and would collide by name.

## What it is for

Phase 4's migration checker runs a v6 rule and a v7 rule on identical inputs and asserts
they agree. `MIGRATION.md`'s before/after doctests execute here too. Both work in a single
process because the new rule packages are differently named and do not depend on
ReactiveMP, so v6 and they coexist.

What does **not** work here is ReactiveMP v7 against v6 — same package name, same UUID, and
Julia loads one version of a package per process. Engine-level comparisons (scheduling,
free-energy trajectories, retained values) therefore need separate pinned processes or
fixtures recorded from full v6 runs. That is why the Phase 4 checker must *capture* results
rather than only assert equality: Phases 4.5 and 7 consume those recordings.

## Why the manifest is pinned to Julia 1.10

**The committed `Manifest.toml` is generated on Julia 1.10.12, matching the declared floor,
and this is load-bearing.** An earlier version was generated on 1.13 and failed to load on
1.10 with a `PrecompileTools`/`StaticData` `UndefVarError`. Manifests are not portable
downwards: resolving on a newer Julia can select stdlib versions and package versions that
the floor cannot load.

Since the whole point of this environment is reproducibility, regenerate it on the floor:

```bash
julia +1.10 --startup-file=no --project=compat/v6-comparison -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

Verify it afterwards, on the floor, and confirm the source is the registry:

```bash
julia +1.10 --startup-file=no --project=compat/v6-comparison -e '
using ReactiveMP
println(pkgversion(ReactiveMP), "  ", pathof(ReactiveMP))'
```

The path must be under `~/.julia/packages/`. If it points into this repository, something
has dev-linked the local copy and the comparison is measuring v7 against itself.

## Adding the new packages

As Phase 3 onwards creates them, add them by relative path from the repository root:

```julia
julia> using Pkg
julia> Pkg.activate("compat/v6-comparison")
julia> Pkg.develop(path = "lib/MessagePassingRulesBase")
julia> Pkg.develop(path = "lib/StandardMessagePassingRules")   # once Phase 5 creates it
```

`Pkg.develop` resolves its path against the working directory, not the activated project,
so run this from the repository root.

## Engine fixtures (Phase 4.5)

Phase 4.5 replaces v6's rule-call and node-creation paths rather than bridging to them, so
before any v6 code is deleted this environment records engine-level fixtures from full v6
runs, through **RxInfer 5.5.2** (pinned, `=5.5.2`; it accepts ReactiveMP 6.5):

```bash
julia +1.10 --startup-file=no --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl          # record
julia +1.10 --startup-file=no --project=compat/v6-comparison compat/v6-comparison/record_engine_fixtures.jl --check  # compare
```

The fixtures are `fixtures/engine/<model>.toml`, one per slice model, written by
`MessagePassingRulesTestUtils.save_engine_fixture`. Each holds the free energy per iteration,
the final posteriors, and every message-rule call in the order v6 made it, which is
materialisation order, with its result and log scale. They are **TOML, not `Serialization`**,
so ReactiveMP's tests can read them on every Julia version in the CI matrix. `--check`
re-records the fixtures and compares them with the committed files; CI runs it. The header's
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

## The migration checker

`check.jl` runs here, on the 1.10 floor, and in CI (`LibTests.yml`, job `v6-comparison`):

```bash
julia +1.10 --startup-file=no --project=compat/v6-comparison compat/v6-comparison/check.jl
```

- `V6Oracle.jl` calls a v6 rule from the inputs a v7 rule takes, and returns its result and
  log scale. It is the only code in the repository that names ReactiveMP v6's internals.
- Ported rules are compared with their v6 originals through `compare_with_reference` from
  `MessagePassingRulesTestUtils`. An undeclared disagreement fails; a declared one must say
  whether it is a `:migration_bug` or a `:correction`, and why.
- v6 rules are verified against their own node definitions with `verify_message_update`.
  Failures there are findings about v6, pinned in `KNOWN_V6_FINDINGS` with their
  explanation, so a new one fails the run until it is understood.

`MessagePassingRulesBase` and `MessagePassingRulesTestUtils` are dev'd into this
environment by relative path and recorded in the committed manifest, resolved on 1.10.
