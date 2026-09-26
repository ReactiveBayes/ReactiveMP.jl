# Comparing with a reference

A rule that reimplements another implementation, such as a rule ported from an earlier release,
is compared with it on the same inputs. [`compare_with_reference`](@ref) makes the comparison a
test: it passes when the two agree, results and log scales, and fails otherwise, unless the
difference was investigated and declared.

A [`DeclaredDisagreement`](@ref) records such a difference with its kind and its reasons:
`:correction` when the reference is wrong and the new implementation deliberately differs,
`:migration_bug` when the new one is wrong and awaits a fix. Neither side is ever silently
preferred: every difference is either an agreement within tolerance or a declaration with its
reasoning, logged whenever it is met.

```julia
declared = [DeclaredDisagreement("NormalMeanVariance:out:q"; kind = :correction, reasoning = "the reference uses E[v], not 1/E[1/v]")]
record = compare_with_reference(
    "NormalMeanVariance:out:q", actual, reference;
    node = NormalMeanVariance, target = :out, inputs, declared,
)
```

Each comparison returns a [`MigrationRecord`](@ref). Records can be saved and read back with
[`save_migration_fixtures`](@ref) and [`load_migration_fixtures`](@ref), which keep every value
exactly but only for the Julia minor version that wrote them; for fixtures read across Julia
versions, see [Engine trajectories](@ref).

## API

```@docs
compare_with_reference
DeclaredDisagreement
MigrationRecord
save_migration_fixtures
load_migration_fixtures
```
