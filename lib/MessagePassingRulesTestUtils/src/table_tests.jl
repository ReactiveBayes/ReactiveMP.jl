"""
    ExpectedWithAnnotations(value; annotations...)

An expected rule output together with the annotations the rule must write, e.g.
`ExpectedWithAnnotations(NormalMeanVariance(0.0, 1.0); logscale = -0.5)`.
"""
struct ExpectedWithAnnotations{V, A <: NamedTuple}
    value::V
    annotations::A
end

ExpectedWithAnnotations(value; annotations...) = ExpectedWithAnnotations(value, NamedTuple(annotations))

expected_value(expected::ExpectedWithAnnotations) = expected.value
expected_value(expected) = expected
expected_annotations(expected::ExpectedWithAnnotations) = expected.annotations
expected_annotations(expected) = NamedTuple()

const DEFAULT_ATOL = Dict{Type, Float64}(Float32 => 1.0e-4, Float64 => 1.0e-6, BigFloat => 1.0e-8)

tolerance_for(tolerance::Real, ::Type) = tolerance
tolerance_for(tolerance::AbstractDict, ::Type{T}) where {T} = get(tolerance, T, get(DEFAULT_ATOL, T, 1.0e-6))
tolerance_for(::Nothing, ::Type{T}) where {T} = get(DEFAULT_ATOL, T, 1.0e-6)

value_float_type(value::Number) = float(typeof(value))
value_float_type(value::AbstractArray) = float(BayesBase.deep_eltype(value))
value_float_type(value::Tuple) = promote_type(map(value_float_type, filter(!isnothing, value))...)
value_float_type(value::NamedTuple) = value_float_type(values(value))
value_float_type(value) = BayesBase.paramfloattype(value)

output_float_type(value) = value_float_type(value)

# A case's inputs, flattened to (container, key, value) so promotion can convert any subset.
struct CaseInputs
    m::NamedTuple
    q::NamedTuple
    clusters::Tuple
    ctx::Any
end

function CaseInputs(inputs::NamedTuple)
    for key in keys(inputs)
        key in (:m, :q, :clusters, :ctx) ||
            throw(ArgumentError("a case's inputs take `m`, `q`, `clusters` and `ctx`; got `$key`"))
    end
    return CaseInputs(
        get(inputs, :m, NamedTuple()), get(inputs, :q, NamedTuple()), Tuple(get(inputs, :clusters, ())),
        get(inputs, :ctx, MessagePassingRulesBase.RuleContext()),
    )
end

rule_args(inputs::CaseInputs) = MessagePassingRulesBase.interactive_args(inputs.m, inputs.q, inputs.clusters)

input_count(inputs::CaseInputs) = length(inputs.m) + length(inputs.q) + length(inputs.clusters)

function input_values(inputs::CaseInputs)
    return Any[values(inputs.m)..., values(inputs.q)..., map(last, inputs.clusters)...]
end

convert_input(::Type{T}, value::Tuple) where {T} = map(v -> convert_input(T, v), value)
convert_input(::Type{T}, ::Nothing) where {T} = nothing
convert_input(::Type{T}, value) where {T} = BayesBase.convert_paramfloattype(T, value)


function converted(inputs::CaseInputs, ::Type{T}, selected) where {T}
    position = Ref(0)
    convert_if_selected(value) = (position[] += 1; position[] in selected ? convert_input(T, value) : value)
    m = map(convert_if_selected, inputs.m)
    q = map(convert_if_selected, inputs.q)
    clusters = map(pair -> first(pair) => convert_if_selected(last(pair)), inputs.clusters)
    return CaseInputs(m, q, clusters, inputs.ctx)
end

function promotion_subsets(n, mode)
    n == 0 && return Vector{Int}[]
    mode === :exhaustive && return [findall(i -> (mask >> (i - 1)) & 1 == 1, 1:n) for mask in 1:(2^n - 1)]
    subsets = [collect(1:n)]
    n > 1 && append!(subsets, [[i] for i in 1:n])
    return subsets
end

struct TableContext
    kind::Symbol
    node::Any
    target::Any
    algorithm::Any
    source::LineNumberNode
end

function resolve(table::TableContext, args)
    table.kind === :message && return MessagePassingRulesBase.find_message_rule(table.node, table.target, table.algorithm, args)
    table.kind === :marginal && return MessagePassingRulesBase.find_marginal_rule(table.node, table.target, table.algorithm, args)
    return MessagePassingRulesBase.find_average_energy(table.node, table.algorithm, args)
end

function describe_case(table::TableContext, index, inputs::CaseInputs)
    parts = String[]
    isempty(inputs.m) || push!(parts, "m = $(inputs.m)")
    isempty(inputs.q) || push!(parts, "q = $(inputs.q)")
    isempty(inputs.clusters) || push!(parts, "clusters = $(inputs.clusters)")
    target_text = table.target === nothing ? "" : " towards $(table.target)"
    return "case $index, $(table.kind) rule for $(table.node)$target_text under $(table.algorithm), inputs ($(join(parts, "; ")))"
end

function run_case(table::TableContext, args, ctx, output = nothing)
    spec = resolve(table, args)
    spec isa MessagePassingRulesBase.RuleSpec || return spec, nothing, nothing
    annotations = MessagePassingRulesBase.RuleAnnotations(out = MessagePassingRulesBase.AnnotationStore())
    result = MessagePassingRulesBase.execute_rule(spec, output, MessagePassingRulesBase.rule_algorithm(spec, table.algorithm), ctx, args, annotations, table.target)
    return spec, result, annotations
end

function check_value(table, label, result, expected, atol, rtol)
    T = output_float_type(expected)
    ok = approximately_equal(result, expected; atol = tolerance_for(atol, T), rtol = tolerance_for(rtol, T))
    description = "$label: got $(repr(result))::$(typeof(result)), expected $(repr(expected))::$(typeof(expected))"
    return record_check(ok, :(rule_output ≈ expected), description, table.source)
end

function check_case(table::TableContext, index, inputs::CaseInputs, expected; atol, rtol, check_type_promotion, float_types, check_nonallocating)
    label = describe_case(table, index, inputs)
    args = rule_args(inputs)
    spec, result, annotations = run_case(table, args, inputs.ctx)
    record_check(spec isa MessagePassingRulesBase.RuleSpec, :(rule_found), () -> sprint(showerror, MessagePassingRulesBase.RuleNotFoundError(spec)), table.source) || return nothing
    record_selected_rule!(spec, table.source)

    check_value(table, label, result, expected_value(expected), atol, something(rtol, 0.0))
    for (key, value) in pairs(expected_annotations(expected))
        actual = MessagePassingRulesBase.getannotation(annotations, key, nothing)
        T = output_float_type(value)
        ok = actual !== nothing && values_close(actual, value; atol = tolerance_for(atol, T), rtol = tolerance_for(something(rtol, 0.0), T))
        record_check(ok, :(annotation == expected), "$label: annotation `$key` is $(repr(actual)), expected $(repr(value))", table.source)
    end

    if spec.inplace
        buffer = spec.prealloc(MessagePassingRulesBase.rule_algorithm(spec, table.algorithm), inputs.ctx, args, table.target)
        _, inplace_result, _ = run_case(table, args, inputs.ctx, buffer)
        T = output_float_type(result)
        record_check(inplace_result === buffer, :(rule!(buffer) === buffer), "$label: `rule!` returned a new object instead of writing into its buffer", table.source)
        record_check(
            approximately_equal(inplace_result, result; atol = tolerance_for(atol, T), rtol = tolerance_for(something(rtol, 0.0), T)),
            :(rule! == rule), () -> "$label: `rule!` into a preallocated buffer gave $(repr(inplace_result)), `rule` gave $(repr(result))", table.source,
        )
    end

    spec.scratch === nothing || check_scratch(table, label, spec, args, inputs.ctx, result, atol, rtol)

    if check_nonallocating
        bytes = measure_allocations(spec, table, args, inputs.ctx)
        record_check(bytes == 0, :(allocated_bytes == 0), "$label: allocated $bytes bytes", table.source)
    end

    if check_type_promotion !== false
        mode = check_type_promotion === :exhaustive ? :exhaustive : :default
        for T in float_types, selected in promotion_subsets(input_count(inputs), mode)
            promoted = converted(inputs, T, selected)
            values_in = input_values(promoted)
            output_type = promote_type(map(value_float_type, values_in)...)
            promoted_expected = convert_input(output_type, expected_value(expected))
            promoted_args = rule_args(promoted)
            promoted_spec, promoted_result, _ = run_case(table, promoted_args, promoted.ctx)
            promoted_label = "$label, with input(s) $(selected) converted to $T"
            if promoted_spec isa MessagePassingRulesBase.RuleSpec
                check_value(table, promoted_label, promoted_result, promoted_expected, atol, something(rtol, 0.0))
            else
                record_check(false, :(rule_found), "$promoted_label: no rule accepts the converted inputs", table.source)
            end
        end
    end
    return nothing
end

# A rule with scratch runs again on a scratch an engine would reuse: one call on it, then one
# after `poison!`, each into a fresh output. Both must agree with the fresh run, or the rule
# reads its scratch before writing it.
function check_scratch(table::TableContext, label, spec, args, ctx, result, atol, rtol)
    algorithm = MessagePassingRulesBase.rule_algorithm(spec, table.algorithm)
    scratch = MessagePassingRulesBase.rule_scratch(spec, algorithm, ctx, args, table.target)
    fresh_output() = spec.inplace ? spec.prealloc(algorithm, ctx, args, table.target) : nothing
    run() = MessagePassingRulesBase.execute_rule(spec, fresh_output(), scratch, algorithm, ctx, args, MessagePassingRulesBase.NoAnnotations(), table.target)
    reused = run()
    poison!(scratch)
    poisoned = run()
    T = output_float_type(result)
    agree(value) = approximately_equal(value, result; atol = tolerance_for(atol, T), rtol = tolerance_for(something(rtol, 0.0), T))
    return record_check(
        agree(reused) && agree(poisoned), :(rule_with_reused_scratch == rule),
        () -> "$label: the rule reads its scratch before writing it: fresh $(repr(result)), on a reused scratch $(repr(reused)), after poisoning it $(repr(poisoned))",
        table.source,
    )
end

"""
    poison!(scratch)

Overwrite a rule's scratch with values no correct computation reads: NaN in every
floating-point array, through tuples and named tuples; other storage is left as it is. Table
tests run a rule with scratch again after poisoning it, and it must give the same result. A
scratch of another kind extends this function. Returns `scratch`.
"""
poison!(x::AbstractArray{<:AbstractFloat}) = fill!(x, NaN)
poison!(x::AbstractArray) = (foreach(poison!, x); x)
poison!(x::Union{Tuple, NamedTuple}) = (foreach(poison!, x); x)
poison!(x) = x

# Measured through the public entry points with concrete arguments, so resolution is static
# and the figure is the rule's own, as an engine calling it would see it. `node::N` forces
# specialisation on a node that is a type; without it Julia passes a `DataType` through and
# resolution turns dynamic.
function measure_allocations(spec, table, args, ctx)
    output = spec.inplace ? spec.prealloc(MessagePassingRulesBase.rule_algorithm(spec, table.algorithm), ctx, args, table.target) : nothing
    kind = Val(table.kind)
    run_measured(kind, output, table.node, table.target, table.algorithm, args, ctx)
    return run_measured(kind, output, table.node, table.target, table.algorithm, args, ctx)
end

@noinline run_measured(::Val{:message}, ::Nothing, node::N, target, algorithm, args, ctx) where {N} =
    @allocated MessagePassingRulesBase.message_passing_rule(node, target, algorithm, args, ctx)
@noinline run_measured(::Val{:message}, output, node::N, target, algorithm, args, ctx) where {N} =
    @allocated MessagePassingRulesBase.message_passing_rule!(output, node, target, algorithm, args, ctx)
@noinline run_measured(::Val{:marginal}, ::Nothing, node::N, target, algorithm, args, ctx) where {N} =
    @allocated MessagePassingRulesBase.message_passing_marginalrule(node, target, algorithm, args, ctx)
@noinline run_measured(::Val{:marginal}, output, node::N, target, algorithm, args, ctx) where {N} =
    @allocated MessagePassingRulesBase.message_passing_marginalrule!(output, node, target, algorithm, args, ctx)
@noinline run_measured(::Val{:average_energy}, ::Nothing, node::N, target, algorithm, args, ctx) where {N} =
    @allocated MessagePassingRulesBase.message_passing_average_energy(node, algorithm, args, ctx)

function run_table(table::TableContext, cases; atol = nothing, rtol = nothing, check_type_promotion = true, float_types = (Float32, Float64, BigFloat), check_nonallocating = false)
    check_type_promotion in (true, false, :exhaustive) ||
        throw(ArgumentError("`check_type_promotion` is `true`, `false` or `:exhaustive`"))
    for (index, case) in enumerate(cases)
        case isa Pair || throw(ArgumentError("each case is `inputs => expected`, got $(typeof(case))"))
        check_case(table, index, CaseInputs(first(case)), last(case); atol, rtol, check_type_promotion, float_types, check_nonallocating)
    end
    return nothing
end

"""
    test_message_update_rule(node, target; cases, algorithm, atol, rtol, check_type_promotion = true, float_types, check_nonallocating = false)

Check a node's message rule towards `target` against a table of `inputs => expected`
cases. `inputs` is a named tuple of `m`, `q`, `clusters` (`(:y, :x) => value` pairs) and
`ctx`; `expected` is the output, or an [`ExpectedWithAnnotations`](@ref).

Each case checks the output and its type. With `check_type_promotion` (the default), it
also checks the output after converting all the inputs, and each input alone, to every type
in `float_types`; `:exhaustive` converts every subset of the inputs. An in-place rule is also
checked for agreement between `rule` and `rule!`, and `check_nonallocating = true` asserts
the call allocates nothing.

`atol` and `rtol` are a number, or a dictionary from float type to tolerance; by default
`atol` is `1e-4`, `1e-6` and `1e-8` for `Float32`, `Float64` and `BigFloat`.
"""
test_message_update_rule(node, target; cases, algorithm = MessagePassingRulesBase.default_algorithm(node), source = LineNumberNode(0, :unknown), kwargs...) =
    run_table(TableContext(:message, node, MessagePassingRulesBase.as_target(target), algorithm, source), cases; kwargs...)

"""
    test_marginal_update_rule(node, target; cases, ...)

As [`test_message_update_rule`](@ref), for the marginal of the cluster `target`.
"""
test_marginal_update_rule(node, target; cases, algorithm = MessagePassingRulesBase.default_algorithm(node), source = LineNumberNode(0, :unknown), kwargs...) =
    run_table(TableContext(:marginal, node, MessagePassingRulesBase.as_cluster(target), algorithm, source), cases; kwargs...)

"""
    test_average_energy(node; cases, ...)

As [`test_message_update_rule`](@ref), for a node's average energy.
"""
test_average_energy(node; cases, algorithm = MessagePassingRulesBase.default_algorithm(node), source = LineNumberNode(0, :unknown), kwargs...) =
    run_table(TableContext(:average_energy, node, nothing, algorithm, source), cases; kwargs...)

function table_macro_call(name, f, positional, args, source)
    keywords = Dict{Symbol, Any}()
    rest = Expr[Expr(:kw, :source, QuoteNode(source))]
    for arg in args
        (arg isa Expr && arg.head === :(=) && arg.args[1] isa Symbol) ||
            error("@$name takes keyword arguments only, like `node = ...`; got `$arg`")
        arg.args[1] in positional ? (keywords[arg.args[1]] = arg.args[2]) : push!(rest, Expr(:kw, arg.args[1], arg.args[2]))
    end
    for key in positional
        haskey(keywords, key) || error("@$name: `$key` is required")
    end
    return Expr(:call, f, Expr(:parameters, rest...), (keywords[key] for key in positional)...)
end

"""
    @test_message_update_rule(node = ..., target = ..., cases = [inputs => expected, ...], ...)

[`test_message_update_rule`](@ref), written with keywords; failures point at this line.

```julia
@test_message_update_rule(
    node    = NormalMeanVariance,
    target = :out,
    cases   = [(m = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0)],
)
```
"""
macro test_message_update_rule(args...)
    return esc(table_macro_call("test_message_update_rule", test_message_update_rule, (:node, :target), args, __source__))
end

"""
    @test_marginal_update_rule(node = ..., target = (:y, :x), cases = [...], ...)

[`test_marginal_update_rule`](@ref), written with keywords.
"""
macro test_marginal_update_rule(args...)
    return esc(table_macro_call("test_marginal_update_rule", test_marginal_update_rule, (:node, :target), args, __source__))
end

"""
    @test_average_energy(node = ..., cases = [...], ...)

[`test_average_energy`](@ref), written with keywords.
"""
macro test_average_energy(args...)
    return esc(table_macro_call("test_average_energy", test_average_energy, (:node,), args, __source__))
end
