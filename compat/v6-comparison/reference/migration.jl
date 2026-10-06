"""
    DeclaredDisagreement(id; kind, reasoning)

A known, investigated difference between a result and its reference, for the comparison named
`id`. Passed to [`compare_with_reference`](@ref) or [`compare_engine_trajectory`](@ref) in
`declared`, it turns that comparison's failure into a recorded outcome, and its reasoning is
logged with `@info` whenever the difference is met. A declaration whose comparison agrees has no
effect.

# Arguments

- `id`: the comparison's name, as given to [`compare_with_reference`](@ref), or the
  reference trajectory's `id` for [`compare_engine_trajectory`](@ref).

# Keywords

- `kind`: `:migration_bug` when the implementation under test is wrong and awaits a fix, or
  `:correction` when the reference is wrong and the implementation deliberately differs.
  Required.
- `reasoning`: why they differ, in words. Required.

# Throws

- `ArgumentError` when `kind` is neither `:migration_bug` nor `:correction`, or `reasoning` is
  blank.
"""
struct DeclaredDisagreement
    id::String
    kind::Symbol
    reasoning::String
    function DeclaredDisagreement(id::AbstractString; kind::Symbol, reasoning::AbstractString)
        kind in (:migration_bug, :correction) ||
            throw(ArgumentError("a disagreement is `:migration_bug` or `:correction`, got `:$kind`"))
        isempty(strip(reasoning)) && throw(ArgumentError("a declared disagreement needs its reasoning"))
        return new(String(id), kind, String(reasoning))
    end
end

"""
    MigrationRecord(id, node, target, inputs, actual, reference, actual_logscale, reference_logscale, outcome)

One comparison of a result with a reference implementation's, on the same inputs, as
[`compare_with_reference`](@ref) returns it and [`save_migration_fixtures`](@ref) stores it.

# Fields

- `id::String`: the comparison's name.
- `node::String`, `target::String`: the node and the target, as text, for reading the record.
- `inputs`: the inputs both sides ran on, as the caller gave them, or `nothing`.
- `actual`, `reference`: the two results.
- `actual_logscale`, `reference_logscale`: their log scales, or `nothing`.
- `outcome::Symbol`: `:agree`; the `kind` of a matching [`DeclaredDisagreement`](@ref),
  `:migration_bug` or `:correction`; or `:disagree` when the results differ and no
  disagreement was declared.
"""
struct MigrationRecord
    id::String
    node::String
    target::String
    inputs::Any
    actual::Any
    reference::Any
    actual_logscale::Any
    reference_logscale::Any
    outcome::Symbol
end

"""
    compare_with_reference(id, actual, reference; inputs = nothing, node = "", target = "",
        actual_logscale = nothing, reference_logscale = nothing, atol = 1e-6, rtol = 0,
        declared = DeclaredDisagreement[]) -> MigrationRecord

Compare a result with a reference implementation's result on the same inputs, as a `Test`
assertion, and return the comparison as a [`MigrationRecord`](@ref).

The two agree when the results are equal by [`approximately_equal`](@ref), types included, and
the log scales are both `nothing` or both numbers within the tolerances. The assertion passes
when they agree, or when they differ and `declared` holds a [`DeclaredDisagreement`](@ref) for
`id`, whose reasoning is then logged; an undeclared difference fails, reporting both sides.
Neither side is ever silently preferred.

# Arguments

- `id`: the comparison's name, unique within `declared`.
- `actual`: the result of the implementation under test.
- `reference`: the reference implementation's result.

# Keywords

- `inputs`, `node`, `target`: what was compared, stored in the record and not checked.
  Defaults: `nothing`, `""` and `""`.
- `actual_logscale`, `reference_logscale`: the two log scales. Default: `nothing`.
- `atol`, `rtol`: the tolerances for the results and the log scales. Defaults: `1e-6` and `0`.
- `declared`: the known disagreements, looked up by `id`. Default: none.
- `source`: the line the check is reported against. Default: `LineNumberNode(0, :unknown)`.

# Examples

```jldoctest; setup = :(using Distributions)
julia> compare_with_reference("normal", Normal(0.0, 1.0), Normal(0.0, 1.0)).outcome
:agree

julia> declared = [DeclaredDisagreement("shifted"; kind = :correction, reasoning = "the reference drops a term")];

julia> compare_with_reference("shifted", Normal(0.0, 1.0), Normal(1.0, 1.0); declared).outcome
[ Info: known disagreement `shifted` (correction): the reference drops a term
:correction
```
"""
function compare_with_reference(id::AbstractString, actual, reference; inputs = nothing, node = "", target = "", actual_logscale = nothing, reference_logscale = nothing, atol = 1.0e-6, rtol = 0.0, declared = DeclaredDisagreement[], source = LineNumberNode(0, :unknown))
    values_agree = approximately_equal(actual, reference; atol, rtol)
    logscales_agree = (actual_logscale === nothing && reference_logscale === nothing) ||
        (actual_logscale !== nothing && reference_logscale !== nothing && isapprox(actual_logscale, reference_logscale; atol, rtol))
    agree = values_agree && logscales_agree
    declaration = findfirst(d -> d.id == id, declared)
    outcome = if agree
        :agree
    elseif declaration !== nothing
        @info "known disagreement `$id` ($(declared[declaration].kind)): $(declared[declaration].reasoning)"
        declared[declaration].kind
    else
        :disagree
    end
    record_check(
        outcome !== :disagree, :(actual ≈ reference),
        () -> "`$id`: the result is $(repr(actual)) (log scale $(repr(actual_logscale))), the reference is $(repr(reference)) (log scale $(repr(reference_logscale))); investigate, then declare it a :migration_bug or a :correction",
        source,
    )
    return MigrationRecord(String(id), string(node), string(target), inputs, actual, reference, actual_logscale, reference_logscale, outcome)
end

const FIXTURE_FORMAT = 1

"""
    save_migration_fixtures(path, records; packages = Dict{String, Any}()) -> path

Write [`MigrationRecord`](@ref)s to `path`, to be read back with
[`load_migration_fixtures`](@ref), with a header of the file format, the Julia version and
`packages`. The file is Julia's `Serialization`, which keeps every value exactly, types
included, but is tied to the Julia version that wrote it; [`save_engine_fixture`](@ref) writes
portable text instead.

# Arguments

- `path`: the file to write, replaced if it exists.
- `records`: a vector of [`MigrationRecord`](@ref)s.

# Keywords

- `packages`: the versions of the packages the records were made with, such as
  `Dict("ReactiveMP" => v"6.5.0")`, stored in the header. Default: empty.
"""
function save_migration_fixtures(path::AbstractString, records::AbstractVector{MigrationRecord}; packages = Dict{String, Any}())
    header = (format = FIXTURE_FORMAT, julia = VERSION, packages = Dict{String, Any}(packages))
    open(io -> serialize(io, (header, collect(records))), path, "w")
    return path
end

"""
    load_migration_fixtures(path) -> (; header, records)

Read what [`save_migration_fixtures`](@ref) wrote, as a `NamedTuple`: `header`, itself a
`NamedTuple` of `format`, `julia` (the `VersionNumber` that wrote the file) and `packages`, and
`records`, the vector of [`MigrationRecord`](@ref)s.

# Throws

- `ArgumentError` when the file's format is not the one this version of the package writes, or
  when it was written by a Julia of another minor version, whose serialization cannot be read
  reliably: regenerate the file.
"""
function load_migration_fixtures(path::AbstractString)
    header, records = open(deserialize, path)
    header.format == FIXTURE_FORMAT || throw(ArgumentError("fixture format $(header.format) is not $FIXTURE_FORMAT"))
    (header.julia.major, header.julia.minor) == (VERSION.major, VERSION.minor) ||
        throw(ArgumentError("fixtures were written by Julia $(header.julia) and cannot be read reliably by $VERSION; regenerate them"))
    return (header = header, records = records)
end
