"""
    encode_fixture_value(x)

The portable form of a value recorded in an engine fixture. Numbers become `Float64`, a
distribution becomes its family name and parameters (`Dict("type" => "Normal", "params" =>
[0.5, 2.0])`), a matrix becomes its rows, a tuple or vector is encoded element by element,
and `nothing` and `missing` are `Dict("type" => "nothing")` and
`Dict("type" => "missing")`. Encoded values encode to themselves, so both
sides of a comparison can be given either way. It holds no Julia types, which is what lets a
fixture recorded on one Julia version be read by another.
"""
encode_fixture_value(x::Real) = Float64(x)
encode_fixture_value(x::AbstractString) = String(x)
encode_fixture_value(::Nothing) = Dict{String, Any}("type" => "nothing")
encode_fixture_value(::Missing) = Dict{String, Any}("type" => "missing")
encode_fixture_value(x::AbstractDict) = Dict{String, Any}(x)
encode_fixture_value(x::AbstractVector{<:Real}) = Float64.(x)
encode_fixture_value(x::AbstractVector) = Any[encode_fixture_value(v) for v in x]
encode_fixture_value(x::AbstractMatrix) = Any[encode_fixture_value(collect(row)) for row in eachrow(x)]
encode_fixture_value(x::Tuple) = Any[encode_fixture_value(v) for v in x]
encode_fixture_value(x::FactorizedCluster) = Dict{String, Any}(
    "type" => "FactorizedCluster",
    "blocks" => Any[String.(collect(block)) for block in cluster_blocks(x)],
    "components" => encode_fixture_value(BayesBase.components(x)),
)
encode_fixture_value(x::PointMass) = Dict{String, Any}("type" => "PointMass", "params" => Any[encode_fixture_value(mean(x))])
encode_fixture_value(x::BayesBase.MixtureDistribution) = Dict{String, Any}(
    "type" => "MixtureDistribution",
    "components" => encode_fixture_value(collect(BayesBase.components(x))),
    "weights" => encode_fixture_value(collect(BayesBase.weights(x))),
)
encode_fixture_value(x::Distribution) =
    Dict{String, Any}("type" => string(nameof(typeof(x))), "params" => Any[encode_fixture_value(p) for p in params(x)])

"""
    RuleCallRecord(iteration, node, target, result, logscale)

One rule call as an engine made it: the iteration it happened in, the node and the target
edge as text, the result, and its log scale (`nothing` when none was recorded).
"""
struct RuleCallRecord
    iteration::Int
    node::String
    target::String
    result::Any
    logscale::Union{Nothing, Float64}
    RuleCallRecord(iteration::Integer, node, target, result, logscale) =
        new(Int(iteration), string(node), string(target), result, logscale === nothing ? nothing : Float64(logscale))
end

"""
    EngineTrajectory(id; description = "", free_energy, posteriors, trace)

What a full inference run produced, for comparing engines rather than rules: the free
energy per iteration, the final posteriors by variable name, and every rule call in the
order it happened. Values may be distributions or their [`encode_fixture_value`](@ref) form.
"""
struct EngineTrajectory
    id::String
    description::String
    free_energy::Vector{Float64}
    posteriors::Dict{String, Any}
    trace::Vector{RuleCallRecord}
end

EngineTrajectory(id::AbstractString; description = "", free_energy, posteriors, trace) =
    EngineTrajectory(String(id), String(description), Float64.(free_energy), Dict{String, Any}(string(k) => v for (k, v) in posteriors), collect(RuleCallRecord, trace))

const ENGINE_FIXTURE_FORMAT = 1

"""
    save_engine_fixture(path, trajectory; packages = Dict(), notes = "")

Write an [`EngineTrajectory`](@ref) as TOML, with a header carrying the format, the Julia and
package versions it was recorded with, and `notes` — the place to state anything the
recording deviates from, so that whoever compares against it knows. Unlike
[`save_migration_fixtures`](@ref) the file is text and readable on any Julia version.
"""
function save_engine_fixture(path::AbstractString, trajectory::EngineTrajectory; packages = Dict{String, Any}(), notes::AbstractString = "")
    header = Dict{String, Any}(
        "format" => ENGINE_FIXTURE_FORMAT,
        "julia" => string(VERSION),
        "notes" => String(notes),
        "packages" => Dict{String, Any}(string(k) => string(v) for (k, v) in packages),
    )
    body = Dict{String, Any}(
        "id" => trajectory.id,
        "description" => trajectory.description,
        "free_energy" => trajectory.free_energy,
        "posteriors" => Dict{String, Any}(k => encode_fixture_value(v) for (k, v) in trajectory.posteriors),
        "trace" => Any[encode_rule_call(r) for r in trajectory.trace],
    )
    open(io -> TOML.print(io, Dict{String, Any}("header" => header, "trajectory" => body); sorted = true), path, "w")
    return path
end

function encode_rule_call(r::RuleCallRecord)
    encoded = Dict{String, Any}("iteration" => r.iteration, "node" => r.node, "target" => r.target, "result" => encode_fixture_value(r.result))
    r.logscale === nothing || (encoded["logscale"] = r.logscale)
    return encoded
end

"""
    load_engine_fixture(path) -> (header, trajectory)

Read what [`save_engine_fixture`](@ref) wrote. The trajectory's values come back encoded.
"""
function load_engine_fixture(path::AbstractString)
    data = TOML.parsefile(path)
    h = data["header"]
    h["format"] == ENGINE_FIXTURE_FORMAT || throw(ArgumentError("engine fixture format $(h["format"]) is not $ENGINE_FIXTURE_FORMAT"))
    header = (format = h["format"], julia = h["julia"], notes = h["notes"], packages = Dict{String, String}(h["packages"]))
    t = data["trajectory"]
    trace = [RuleCallRecord(r["iteration"], r["node"], r["target"], r["result"], get(r, "logscale", nothing)) for r in t["trace"]]
    trajectory = EngineTrajectory(t["id"]; description = t["description"], free_energy = t["free_energy"], posteriors = t["posteriors"], trace)
    return (header, trajectory)
end

encoded_close(a::Real, b::Real; atol, rtol) = isapprox(a, b; atol, rtol)
encoded_close(a::AbstractString, b::AbstractString; atol, rtol) = a == b
encoded_close(a::AbstractVector, b::AbstractVector; atol, rtol) =
    length(a) == length(b) && all(((x, y),) -> encoded_close(x, y; atol, rtol), zip(a, b))
encoded_close(a::AbstractDict, b::AbstractDict; atol, rtol) =
    keys(a) == keys(b) && all(k -> encoded_close(a[k], b[k]; atol, rtol), keys(a))
encoded_close(a, b; atol, rtol) = false

values_agree(a, b; atol, rtol) = encoded_close(encode_fixture_value(a), encode_fixture_value(b); atol, rtol)

describe_call(r::RuleCallRecord) = "$(r.node)($(r.target)) in iteration $(r.iteration) → $(repr(encode_fixture_value(r.result)))"

"""
    compare_engine_trajectory(v7, v6; atol = 1e-6, rtol = 0, declared = [], trace_order = :exact, collapse_repeats = false)

Compare a v7 run with its recorded v6 [`EngineTrajectory`](@ref) and return the outcome:
`:agree`, the kind of a matching [`DeclaredDisagreement`](@ref) (looked up by the v6 id), or
`:disagree`. The free energy, each posterior, and the trace — every rule call with its result
and log scale — are checked and reported separately.

With `trace_order = :exact` the rule calls must come **in the same order**, because emission
order is part of what an engine must reproduce. `trace_order = :within_iteration` declares
the order of the calls inside an iteration free: each iteration must make the same calls with
the same results, in any order. It is for a schedule that is changed deliberately where the
calls it reorders do not depend on each other; say why where it is used.

`collapse_repeats = true` declares that v6 computed a message once per subscriber where an engine
shares it: a v6 call that repeats an earlier call of its own iteration, with an agreeing result,
is dropped before the traces are compared. A repeat with another result stays, and so does one in
a later iteration.
"""
function compare_engine_trajectory(v7::EngineTrajectory, v6::EngineTrajectory; atol = 1.0e-6, rtol = 0.0, declared = DeclaredDisagreement[], trace_order = :exact, collapse_repeats = false, source = LineNumberNode(0, :unknown))
    trace_order in (:exact, :within_iteration) || throw(ArgumentError("`trace_order` is `:exact` or `:within_iteration`, got $(repr(trace_order))"))
    checks = Tuple{Bool, Any, Function}[]

    fe_ok = length(v7.free_energy) == length(v6.free_energy) && all(((a, b),) -> isapprox(a, b; atol, rtol), zip(v7.free_energy, v6.free_energy))
    push!(checks, (fe_ok, :(v7.free_energy ≈ v6.free_energy), () -> "`$(v6.id)`: free energy per iteration differs: v7 $(v7.free_energy), v6 $(v6.free_energy)"))

    names = sort!(collect(union(keys(v7.posteriors), keys(v6.posteriors))))
    for name in names
        present = haskey(v7.posteriors, name) && haskey(v6.posteriors, name)
        ok = present && values_agree(v7.posteriors[name], v6.posteriors[name]; atol, rtol)
        push!(checks, (ok, :(v7.posteriors[$name] ≈ v6.posteriors[$name]), () -> "`$(v6.id)`: posterior `$name` differs: v7 $(repr(present ? encode_fixture_value(v7.posteriors[name]) : get(v7.posteriors, name, missing))), v6 $(repr(get(v6.posteriors, name, missing)))"))
    end

    v6_trace = collapse_repeats ? without_repeats(v6.trace; atol, rtol) : v6.trace
    mismatch = trace_order === :exact ? first_trace_mismatch(v7.trace, v6_trace; atol, rtol) : first_unmatched_call(v7.trace, v6_trace; atol, rtol)
    push!(checks, (mismatch === nothing, :(v7.trace ≈ v6.trace), () -> "`$(v6.id)`: $mismatch"))

    agree = all(first, checks)
    declaration = findfirst(d -> d.id == v6.id, declared)
    outcome = if agree
        :agree
    elseif declaration !== nothing
        @info "known disagreement `$(v6.id)` ($(declared[declaration].kind)): $(declared[declaration].reasoning)"
        declared[declaration].kind
    else
        :disagree
    end
    for (ok, expression, describe) in checks
        record_check(ok || outcome !== :disagree, expression, describe, source)
    end
    return outcome
end

# The calls of `trace` that repeat no earlier call of their iteration.
function without_repeats(trace; atol, rtol)
    kept = eltype(trace)[]
    for call in trace
        any(earlier -> calls_agree(earlier, call; atol, rtol), kept) || push!(kept, call)
    end
    return kept
end

calls_agree(a, b; atol, rtol) =
    (a.iteration, a.node, a.target) == (b.iteration, b.node, b.target) && values_agree(a.result, b.result; atol, rtol) &&
    ((a.logscale === nothing && b.logscale === nothing) || (a.logscale !== nothing && b.logscale !== nothing && isapprox(a.logscale, b.logscale; atol, rtol)))

# Matches every v6 call with a v7 call of its iteration that agrees, each used once.
function first_unmatched_call(v7, v6; atol, rtol)
    unmatched = collect(eachindex(v7))
    for b in v6
        position = findfirst(i -> calls_agree(v7[i], b; atol, rtol), unmatched)
        position === nothing && return "v6 made $(describe_call(b)), which no call of v7 in iteration $(b.iteration) matches"
        deleteat!(unmatched, position)
    end
    isempty(unmatched) && return nothing
    return "v7 made $(describe_call(v7[first(unmatched)])), which v6 did not make"
end

function first_trace_mismatch(v7, v6; atol, rtol)
    for i in 1:min(length(v7), length(v6))
        a, b = v7[i], v6[i]
        if (a.iteration, a.node, a.target) != (b.iteration, b.node, b.target) || !values_agree(a.result, b.result; atol, rtol)
            return "rule call $i differs: v7 made $(describe_call(a)), v6 made $(describe_call(b))"
        end
        logscales_agree = (a.logscale === nothing && b.logscale === nothing) ||
            (a.logscale !== nothing && b.logscale !== nothing && isapprox(a.logscale, b.logscale; atol, rtol))
        logscales_agree || return "rule call $i, $(describe_call(b)): log scale v7 $(a.logscale), v6 $(b.logscale)"
    end
    length(v7) == length(v6) && return nothing
    return "the traces have $(length(v7)) (v7) and $(length(v6)) (v6) rule calls; the first $(min(length(v7), length(v6))) agree"
end
