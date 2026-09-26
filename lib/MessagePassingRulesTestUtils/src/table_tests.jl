"""
    ExpectedWithAnnotations(value; annotations...)
    ExpectedWithAnnotations(value, annotations::NamedTuple)

The expected result of a table case together with the annotations the rule must write:
`ExpectedWithAnnotations(NormalMeanVariance(0.0, 1.0); input_count = 2)`. The result is
compared as a plain expected value is; each annotation is looked up by name among those the
rule wrote with [`annotate!`](@extref MessagePassingRulesBase.annotate!), must be present, and
is compared by value within the table's tolerances, not by type. Annotations are checked on the
case's own inputs, not on its promoted runs.

# Throws

- `ArgumentError` when `logscale` is among the annotations: a log scale is part of the
  message, not an annotation, and is expected with [`ExpectedWithLogScale`](@ref).
"""
struct ExpectedWithAnnotations{V, A <: NamedTuple}
    value::V
    annotations::A
    function ExpectedWithAnnotations(value::V, annotations::A) where {V, A <: NamedTuple}
        haskey(annotations, :logscale) &&
            throw(ArgumentError("a log scale is not an annotation; expect it with `ExpectedWithLogScale(value, logscale)`"))
        return new{V, A}(value, annotations)
    end
end

ExpectedWithAnnotations(value; annotations...) = ExpectedWithAnnotations(value, NamedTuple(annotations))

"""
    ExpectedWithLogScale(value, logscale::Real)

The expected message of a table case together with the log scale its rule must declare:
`ExpectedWithLogScale(Beta(2.0, 1.0), -log(2))`. For message rules only: marginal rules and
average energies have no log scale, and a case expecting one fails.

The log scale must be a number, so a rule that declares none fails, and it is compared with
`logscale` within the table's tolerances for the float type of `logscale`. On every promoted
run of the case (see `check_type_promotion`), a floating-point log scale must also be of the
inputs' promoted float type, as the message is: a computed log scale follows the inputs'
precision. A declared integer or irrational constant, `logscale = 0`, passes that check.
"""
struct ExpectedWithLogScale{V, L <: Real}
    value::V
    logscale::L
end

expected_value(expected::Union{ExpectedWithAnnotations, ExpectedWithLogScale}) = expected.value
expected_value(expected) = expected
expected_annotations(expected::ExpectedWithAnnotations) = expected.annotations
expected_annotations(expected) = NamedTuple()
expected_logscale(expected::ExpectedWithLogScale) = expected.logscale
expected_logscale(expected) = nothing

const DEFAULT_ATOL = Dict{Type, Float64}(Float32 => 1.0e-4, Float64 => 1.0e-6, BigFloat => 1.0e-8)

tolerance_for(tolerance::Real, ::Type) = tolerance
tolerance_for(tolerance::AbstractDict, ::Type{T}) where {T} = get(tolerance, T, get(DEFAULT_ATOL, T, 1.0e-6))
tolerance_for(::Nothing, ::Type{T}) where {T} = get(DEFAULT_ATOL, T, 1.0e-6)
# A relative tolerance: a float type a dictionary does not name gets none.
relative_tolerance_for(tolerance::Real, ::Type) = tolerance
relative_tolerance_for(tolerance::AbstractDict, ::Type{T}) where {T} = get(tolerance, T, 0.0)

value_float_type(value::Number) = float(typeof(value))
value_float_type(value::AbstractArray) = float(BayesBase.deep_eltype(value))
value_float_type(value::Tuple) = promote_type(map(value_float_type, filter(!isnothing, value))...)
value_float_type(value::NamedTuple) = value_float_type(values(value))
value_float_type(value) = BayesBase.paramfloattype(value)

output_float_type(value) = value_float_type(value)

# A case's inputs, flattened to (container, key, value) so promotion can convert any subset.
# `logscale` holds the log scales that arrived with the messages, for a rule that reads them;
# they are not converted by promotion.
struct CaseInputs
    m::NamedTuple
    q::NamedTuple
    clusters::Tuple
    ctx::Any
    logscale::Union{Nothing, NamedTuple}
end

function CaseInputs(inputs::NamedTuple)
    for key in keys(inputs)
        key in (:m, :q, :clusters, :ctx, :logscale) ||
            throw(ArgumentError("a case's inputs take `m`, `q`, `clusters`, `ctx` and `logscale`; got `$key`"))
    end
    return CaseInputs(
        get(inputs, :m, NamedTuple()), get(inputs, :q, NamedTuple()), Tuple(get(inputs, :clusters, ())),
        get(inputs, :ctx, MessagePassingRulesBase.RuleContext()), get(inputs, :logscale, nothing),
    )
end

rule_args(inputs::CaseInputs) = MessagePassingRulesBase.interactive_args(inputs.m, inputs.q, inputs.clusters, inputs.logscale)

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
    return CaseInputs(m, q, clusters, inputs.ctx, inputs.logscale)
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
    inputs.logscale === nothing || push!(parts, "logscale = $(inputs.logscale)")
    target_text = table.target === nothing ? "" : " towards $(table.target)"
    return "case $index, $(table.kind) rule for $(table.node)$target_text under $(table.algorithm), inputs ($(join(parts, "; ")))"
end

# The rule's result, the annotations it wrote and, for a message rule, its log scale.
function run_case(table::TableContext, args, ctx, output = nothing)
    spec = resolve(table, args)
    spec isa MessagePassingRulesBase.RuleSpec || return spec, nothing, nothing, nothing
    MessagePassingRulesBase.check_reads_logscale(spec, args)
    annotations = MessagePassingRulesBase.RuleAnnotations(out = MessagePassingRulesBase.AnnotationStore())
    algorithm = MessagePassingRulesBase.rule_algorithm(spec, table.algorithm)
    result, logscale = if table.kind === :message
        MessagePassingRulesBase.execute_rule_with_logscale(spec, output, nothing, algorithm, ctx, args, annotations, table.target)
    else
        MessagePassingRulesBase.execute_rule(spec, output, algorithm, ctx, args, annotations, table.target), nothing
    end
    return spec, result, annotations, logscale
end

function check_logscale(table, label, actual, expected, atol, rtol)
    T = output_float_type(expected)
    ok = actual isa Real && values_close(actual, expected; atol = tolerance_for(atol, T), rtol = relative_tolerance_for(rtol, T))
    return record_check(ok, :(logscale ≈ expected), () -> "$label: the log scale is $(repr(actual)), expected $(repr(expected))", table.source)
end

# A computed log scale follows the precision of the inputs, as the message does: a wider float
# type than the output's loses nothing but is not what the inputs asked for, and a narrower one
# loses precision. A declared integer or irrational constant takes the type of whatever it is
# added to.
function check_logscale_type(table, label, logscale, output_type)
    logscale isa AbstractFloat || return nothing
    return record_check(
        typeof(logscale) === output_type, :(typeof(logscale) == output_type),
        () -> "$label: the log scale is a $(typeof(logscale)), the inputs ask for $output_type", table.source,
    )
end

function check_value(table, label, result, expected, atol, rtol)
    T = output_float_type(expected)
    ok = approximately_equal(result, expected; atol = tolerance_for(atol, T), rtol = relative_tolerance_for(rtol, T))
    description = "$label: got $(repr(result))::$(typeof(result)), expected $(repr(expected))::$(typeof(expected))"
    return record_check(ok, :(rule_output ≈ expected), description, table.source)
end

function check_case(table::TableContext, index, inputs::CaseInputs, expected; atol, rtol, check_type_promotion, float_types, check_nonallocating)
    label = describe_case(table, index, inputs)
    args = rule_args(inputs)
    spec, result, annotations, logscale = run_case(table, args, inputs.ctx)
    record_check(spec isa MessagePassingRulesBase.RuleSpec, :(rule_found), () -> sprint(showerror, MessagePassingRulesBase.RuleNotFoundError(spec)), table.source) || return nothing
    record_selected_rule!(spec, table.source)

    check_value(table, label, result, expected_value(expected), atol, something(rtol, 0.0))
    for (key, value) in pairs(expected_annotations(expected))
        actual = MessagePassingRulesBase.getannotation(annotations, key, nothing)
        T = output_float_type(value)
        ok = actual !== nothing && values_close(actual, value; atol = tolerance_for(atol, T), rtol = relative_tolerance_for(something(rtol, 0.0), T))
        record_check(ok, :(annotation == expected), "$label: annotation `$key` is $(repr(actual)), expected $(repr(value))", table.source)
    end
    expected_scale = expected_logscale(expected)
    expected_scale === nothing || table.kind === :message || throw(
        ArgumentError("$label expects a log scale, but a $(table.kind === :marginal ? "marginal rule" : "average energy") has none: `ExpectedWithLogScale` is for message rules"),
    )
    expected_scale === nothing || check_logscale(table, label, logscale, expected_scale, atol, something(rtol, 0.0))

    if spec.inplace
        buffer = spec.prealloc(MessagePassingRulesBase.rule_algorithm(spec, table.algorithm), inputs.ctx, args, table.target)
        _, inplace_result, _, _ = run_case(table, args, inputs.ctx, buffer)
        T = output_float_type(result)
        record_check(inplace_result === buffer, :(rule!(buffer) === buffer), "$label: `rule!` returned a new object instead of writing into its buffer", table.source)
        record_check(
            approximately_equal(inplace_result, result; atol = tolerance_for(atol, T), rtol = relative_tolerance_for(something(rtol, 0.0), T)),
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
            promoted_spec, promoted_result, _, promoted_logscale = run_case(table, promoted_args, promoted.ctx)
            promoted_label = "$label, with input(s) $(selected) converted to $T"
            if promoted_spec isa MessagePassingRulesBase.RuleSpec
                check_value(table, promoted_label, promoted_result, promoted_expected, atol, something(rtol, 0.0))
                expected_scale === nothing || check_logscale_type(table, promoted_label, promoted_logscale, output_type)
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
    agree(value) = approximately_equal(value, result; atol = tolerance_for(atol, T), rtol = relative_tolerance_for(something(rtol, 0.0), T))
    return record_check(
        agree(reused) && agree(poisoned), :(rule_with_reused_scratch == rule),
        () -> "$label: the rule reads its scratch before writing it: fresh $(repr(result)), on a reused scratch $(repr(reused)), after poisoning it $(repr(poisoned))",
        table.source,
    )
end

"""
    poison!(scratch) -> scratch

Overwrite a rule's scratch with values no correct computation reads: `NaN` in every
floating-point array, found through arrays of arrays, tuples and named tuples; any other
storage is left as it is. Table tests run a rule that has scratch once more after poisoning its
scratch, and the result must not change: a rule that reads its scratch before writing it
fails. A package whose scratch holds another kind of storage adds a method.

# Examples

```jldoctest
julia> scratch = (work = zeros(2), counts = [1, 2]);

julia> MessagePassingRulesTestUtils.poison!(scratch) === scratch
true

julia> all(isnan, scratch.work), scratch.counts
(true, [1, 2])
```
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

# Parts of the table docstrings that several share, written once and interpolated.

const DOC_TABLE_INPUTS = rstrip(
    """
    - `cases`: the table, a vector of `inputs => expected` pairs. Required. `inputs` is a
      `NamedTuple` with any of these entries:
      - `m`: the inbound messages, keyed by the interfaces' declared names:
        `m = (μ = PointMass(1.0), v = PointMass(2.0))`. A group is a tuple of its members in
        order, with `nothing` for a member the rule does not take.
      - `q`: the marginals of single interfaces, keyed the same way.
      - `clusters`: the joint marginals of clusters, a tuple of pairs from the cluster's
        members, in interface order, to its joint: `clusters = ((:out, :μ) => q_outμ,)`.
      - `ctx`: the [`RuleContext`](@extref MessagePassingRulesBase.RuleContext) of services the
        rule reads. Default: an empty one. Its services are not checked: one the rule declares
        and `ctx` lacks reads as `nothing` inside the rule.
      - `logscale`: the log scales that arrived with the messages, keyed like `m`:
        `logscale = (in = 0.5,)`, for a rule declared with `reads_logscale = true`.
    """
)

const DOC_TABLE_KEYWORDS = rstrip(
    """
    - `algorithm`: the algorithm value to run under. Default:
      [`default_algorithm`](@extref MessagePassingRulesBase.default_algorithm)`(node)`. Under a
      [`DefaultAlgorithmExtension`](@extref MessagePassingRulesBase.DefaultAlgorithmExtension)
      the extension's rules are selected, and the default's where it has none; an inherited
      rule runs with the value it was written for, as an engine runs it.
    - `atol`: the absolute tolerance, a number, or a `Dict` from float type to number, looked up
      by the float type of the expected value. Default: `nothing`, which is `1e-4` for
      `Float32`, `1e-6` for `Float64`, `1e-8` for `BigFloat` and `1e-6` for any other type; a
      `Dict` without an entry for a type falls back to the same values.
    - `rtol`: the relative tolerance, in the same forms. Default: `nothing`, which is `0`. A
      `Dict` without an entry for a type falls back to the default `atol` values above.
    - `check_type_promotion`: whether to run each case again with inputs of other float types.
      `true` (the default) runs it, for every type `T` in `float_types`, with all its inputs
      converted to `T` and, when it has more than one, with each input alone converted;
      `:exhaustive` converts every non-empty subset of the inputs instead; `false` skips it. An
      input is one entry of `m`, `q` or `clusters`, a group's tuple counting as one, converted
      with `BayesBase.convert_paramfloattype`; `ctx` and `logscale` stay as they are. The
      expected value is converted to the promoted float type of all the inputs, and the result
      must equal it, type included.
    - `float_types`: the types promotion converts to. Default: `(Float32, Float64, BigFloat)`.
    - `check_nonallocating`: whether the rule must allocate nothing. Default: `false`. When
      `true`, the call through the engine's entry point,
      [`message_passing_rule`](@extref MessagePassingRulesBase.message_passing_rule) or its
      in-place, marginal or average-energy sibling, is measured with `@allocated` on its second
      run, after compilation, and must allocate 0 bytes.
    - `source`: the line checks are reported against. Default: `LineNumberNode(0, :unknown)`;
      the macro form passes its own line.
    """
)

const DOC_TABLE_CHECKS = rstrip(
    """
    Each case makes these checks, each a `Test` assertion:

    1. A rule is found for the inputs; if none is, the case fails with the
       [`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError) text, which
       lists the closest rules, and its other checks are skipped. The rule found is recorded as
       selected for [`check_rule_coverage`](@ref).
    2. The result equals the expected value by [`approximately_equal`](@ref): same type, values
       within the tolerances.
    3. The annotations an [`ExpectedWithAnnotations`](@ref) names, and the log scale an
       [`ExpectedWithLogScale`](@ref) gives.
    4. For an in-place rule, `rule!` into a buffer from the rule's `preallocate` returns that
       buffer, and agrees with `rule`.
    5. For a rule with scratch, two more runs on one scratch from the rule's `scratch` function,
       the first on the scratch as it is and the second after [`poison!`](@ref), each into a
       fresh output, agree with the first result.
    6. With `check_nonallocating = true`, the rule allocates nothing.
    7. With `check_type_promotion`, every promoted run finds a rule and returns the promoted
       expected value, and a floating-point log scale of the promoted type when the case expects
       one. Annotations, `rule!` and scratch are checked on the case's own inputs only.
    """
)

const DOC_TABLE_THROWS = rstrip(
    """
    - `ArgumentError` when `check_type_promotion` is not `true`, `false` or `:exhaustive`, when a
      case is not a `Pair`, or when a case's inputs have an entry other than `m`, `q`,
      `clusters`, `ctx` and `logscale`. A rule that is missing or wrong is a test failure, not
      an error.
    """
)

doc_keyword_macro(f, required) = """
The macro takes keyword arguments only, written `name = value`, and passes them to
[`$(f)`](@ref); $(required) required, and a positional argument or a missing required
keyword is an error when the macro expands. Every check is reported against the line of the
macro call, so a failure points at the call that made it.
"""

"""
    test_message_update_rule(node, target; cases, algorithm = default_algorithm(node), atol = nothing,
        rtol = nothing, check_type_promotion = true, float_types = (Float32, Float64, BigFloat),
        check_nonallocating = false) -> nothing

Check the message rule of `node` towards `target` against a table of cases, each the inputs of
one call and the message it must return. [`@test_message_update_rule`](@ref) is the same check
written with keywords, and reports failures at its own line.

# Arguments

- `node`: the node, as declared with
  [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node).
- `target`: the interface the message goes to: `:out`, or `(:m, k)` for member `k` of the group
  `m`.

# Keywords

$(DOC_TABLE_INPUTS)

  `expected` is the message: a plain value, an [`ExpectedWithLogScale`](@ref) to check the log
  scale the rule declares as well, or an [`ExpectedWithAnnotations`](@ref) to check what it
  annotates.

$(DOC_TABLE_KEYWORDS)

# Checks

$(DOC_TABLE_CHECKS)

# Throws

$(DOC_TABLE_THROWS)

# Examples

```julia
test_message_update_rule(
    NormalMeanVariance, :out;
    cases = [
        (m = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
        (m = (μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) => ExpectedWithLogScale(NormalMeanVariance(0.0, 3.0), 0),
    ],
)
```

See also [`test_marginal_update_rule`](@ref), [`test_average_energy`](@ref),
[`check_rule_coverage`](@ref).
"""
test_message_update_rule(node, target; cases, algorithm = MessagePassingRulesBase.default_algorithm(node), source = LineNumberNode(0, :unknown), kwargs...) =
    run_table(TableContext(:message, node, MessagePassingRulesBase.as_target(target), algorithm, source), cases; kwargs...)

"""
    test_marginal_update_rule(node, target; cases, algorithm = default_algorithm(node), atol = nothing,
        rtol = nothing, check_type_promotion = true, float_types = (Float32, Float64, BigFloat),
        check_nonallocating = false) -> nothing

Check the marginal rule of `node` for the cluster `target` against a table of cases, each the
inputs of one call and the joint marginal it must return. [`@test_marginal_update_rule`](@ref)
is the same check written with keywords, and reports failures at its own line.

# Arguments

- `node`: the node, as declared with
  [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node).
- `target`: the cluster whose joint marginal the rule computes, its members in interface order:
  `(:out, :μ)`, or `(:out, (:T, 1))` with a group member.

# Keywords

$(DOC_TABLE_INPUTS)

  `expected` is the joint marginal, a plain value or an [`ExpectedWithAnnotations`](@ref); a
  cluster that factorises is expected as a
  [`FactorizedCluster`](@extref MessagePassingRulesBase.FactorizedCluster). A marginal has no
  log scale, so an [`ExpectedWithLogScale`](@ref) fails.

$(DOC_TABLE_KEYWORDS)

# Checks

$(DOC_TABLE_CHECKS)

# Throws

$(DOC_TABLE_THROWS)

See also [`test_message_update_rule`](@ref), [`test_average_energy`](@ref).
"""
test_marginal_update_rule(node, target; cases, algorithm = MessagePassingRulesBase.default_algorithm(node), source = LineNumberNode(0, :unknown), kwargs...) =
    run_table(TableContext(:marginal, node, MessagePassingRulesBase.as_cluster(target), algorithm, source), cases; kwargs...)

"""
    test_average_energy(node; cases, algorithm = default_algorithm(node), atol = nothing,
        rtol = nothing, check_type_promotion = true, float_types = (Float32, Float64, BigFloat),
        check_nonallocating = false) -> nothing

Check the average energy of `node` against a table of cases, each the marginals of one call and
the energy it must return. [`@test_average_energy`](@ref) is the same check written with
keywords, and reports failures at its own line.

# Arguments

- `node`: the node, as declared with
  [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node).

# Keywords

$(DOC_TABLE_INPUTS)

  An average energy reads marginals, `q` and `clusters`, one per cluster of the factorisation.
  `expected` is the energy, a number, or an [`ExpectedWithAnnotations`](@ref); an average
  energy has no log scale, so an [`ExpectedWithLogScale`](@ref) fails.

$(DOC_TABLE_KEYWORDS)

# Checks

$(DOC_TABLE_CHECKS)

# Throws

$(DOC_TABLE_THROWS)

See also [`test_message_update_rule`](@ref), [`test_marginal_update_rule`](@ref).
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

Check the message rule of a node towards one of its interfaces against a table of cases, each
the inputs of one call and the message it must return. It takes the arguments and keywords of
[`test_message_update_rule`](@ref), where they are described, all by name.

$(doc_keyword_macro("test_message_update_rule", "`node` and `target` are"))

# Examples

```julia
@test_message_update_rule(
    node = NormalMeanVariance,
    target = :out,
    cases = [
        (m = (μ = PointMass(1.0), v = PointMass(2.0)),) => NormalMeanVariance(1.0, 2.0),
        (q = (μ = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)),) => NormalMeanVariance(1.0, 4 / 3),
        (m = (μ = NormalMeanVariance(0.0, 1.0), v = PointMass(2.0)),) => ExpectedWithLogScale(NormalMeanVariance(0.0, 3.0), 0),
    ],
)
```
"""
macro test_message_update_rule(args...)
    return esc(table_macro_call("test_message_update_rule", test_message_update_rule, (:node, :target), args, __source__))
end

"""
    @test_marginal_update_rule(node = ..., target = (:out, :μ), cases = [inputs => expected, ...], ...)

Check the marginal rule of a node for one of its clusters against a table of cases, each the
inputs of one call and the joint marginal it must return. It takes the arguments and keywords of
[`test_marginal_update_rule`](@ref), where they are described, all by name.

$(doc_keyword_macro("test_marginal_update_rule", "`node` and `target` are"))

# Examples

```julia
@test_marginal_update_rule(
    node = NormalMeanVariance,
    target = (:out, :μ),
    cases = [
        (m = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(2.0, 1.0)), q = (v = PointMass(1.0),)) => q_outμ,
    ],
)
```
"""
macro test_marginal_update_rule(args...)
    return esc(table_macro_call("test_marginal_update_rule", test_marginal_update_rule, (:node, :target), args, __source__))
end

"""
    @test_average_energy(node = ..., cases = [inputs => expected, ...], ...)

Check the average energy of a node against a table of cases, each the marginals of one call and
the energy it must return. It takes the arguments and keywords of
[`test_average_energy`](@ref), where they are described, all by name.

$(doc_keyword_macro("test_average_energy", "`node` is"))

# Examples

```julia
@test_average_energy(
    node = NormalMeanVariance,
    cases = [
        (q = (out = NormalMeanVariance(0.0, 1.0), μ = NormalMeanVariance(1.0, 1.0), v = PointMass(1.0)),) => 1.5 + log(2π) / 2,
    ],
)
```
"""
macro test_average_energy(args...)
    return esc(table_macro_call("test_average_energy", test_average_energy, (:node,), args, __source__))
end
