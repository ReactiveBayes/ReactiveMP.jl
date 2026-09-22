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
julia> Pkg.develop(path = "lib/StandardMessagePassingRules")
```

`Pkg.develop` resolves its path against the working directory, not the activated project,
so run this from the repository root.

## The standing constraint

This environment resolves only while shared dependencies stay inside v6.5.0's caret bounds
— `BayesBase = "1.5"`, `ExponentialFamily = "2.5.0"`. New **minor** versions are fine.

`PLAN.md` instructs that anything the base package needs be **added to BayesBase**. Those
additions must therefore ship as non-breaking 1.x releases. A BayesBase 2.0 released for
this rewrite would make this environment unresolvable and silently cost us the main
instrument for verifying 490 ported rules. If a breaking BayesBase change becomes
unavoidable, redesign this harness first, not afterwards.
