"""
    DeclaredDisagreement(id; kind, reasoning)

A known, investigated difference between the actual and the reference result of the
comparison `id`: `kind = :migration_bug` when the implementation under test is wrong and awaits
a fix, `:correction` when the reference is wrong and the implementation deliberately differs.
`reasoning` says why; it is printed whenever the disagreement is met.
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
    MigrationRecord

One comparison of a rule against a reference implementation on the same inputs: both
results and log scales, and the `outcome` — `:agree`, a declared `:migration_bug` or
`:correction`, or `:disagree` when the difference was never declared.
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
    compare_with_reference(id, actual, reference; inputs, node, target, actual_logscale, reference_logscale, atol, rtol, declared)

Compare a result with the reference implementation's and return a
[`MigrationRecord`](@ref). The check passes when they agree — values, types and log scales —
or when the difference is one of the `declared` disagreements; an undeclared difference fails.
Neither side is ever silently preferred.
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
    save_migration_fixtures(path, records; packages = Dict{String, Any}())

Write comparison records to be read back later, stamped with the Julia version and the
given package versions. Serialization is exact but tied to the Julia version, so
[`load_migration_fixtures`](@ref) refuses a file written by another minor release.
"""
function save_migration_fixtures(path::AbstractString, records::AbstractVector{MigrationRecord}; packages = Dict{String, Any}())
    header = (format = FIXTURE_FORMAT, julia = VERSION, packages = Dict{String, Any}(packages))
    open(io -> serialize(io, (header, collect(records))), path, "w")
    return path
end

"""
    load_migration_fixtures(path)

The records and header written by [`save_migration_fixtures`](@ref).
"""
function load_migration_fixtures(path::AbstractString)
    header, records = open(deserialize, path)
    header.format == FIXTURE_FORMAT || throw(ArgumentError("fixture format $(header.format) is not $FIXTURE_FORMAT"))
    (header.julia.major, header.julia.minor) == (VERSION.major, VERSION.minor) ||
        throw(ArgumentError("fixtures were written by Julia $(header.julia) and cannot be read reliably by $VERSION; regenerate them"))
    return (header = header, records = records)
end
