# The reference update a rule must agree with, computed from the node's log-density.

const QUADRATURE_TAIL = 1.0e-9
const MAX_INTEGRATED_DIMENSIONS = 2

abstract type InputTreatment end
struct FixedAt{T} <: InputTreatment
    value::T
end
struct Enumerated{V, W} <: InputTreatment
    values::V
    logweights::W
end
struct Integrated{D} <: InputTreatment
    distribution::D
    lower::Float64
    upper::Float64
end

function input_treatment(name, input)
    input isa PointMass && return FixedAt(mean(input))
    if input isa DiscreteUnivariateDistribution
        values = collect(support(input))
        length(values) < 10_000 || throw(ArgumentError("`$name` has an unbounded discrete support, which verification cannot enumerate"))
        return Enumerated(values, map(x -> logpdf(input, x), values))
    end
    input isa ContinuousUnivariateDistribution &&
        return Integrated(input, Float64(quantile(input, QUADRATURE_TAIL)), Float64(quantile(input, 1 - QUADRATURE_TAIL)))
    throw(ArgumentError("verification supports point masses and univariate distributions; `$name` is a $(typeof(input))"))
end

# log Σ_discrete ∫_continuous g(x) ∏ w(x) dx, for `g` returning log values (BP) or a plain
# expectation Σ ∫ g(x) ∏ w(x) dx (VMP).
function reference_value(g, names, treatments, mode::Symbol)
    integrated = [i for (i, t) in enumerate(treatments) if t isa Integrated]
    length(integrated) <= MAX_INTEGRATED_DIMENSIONS ||
        throw(ArgumentError("verification integrates at most $MAX_INTEGRATED_DIMENSIONS continuous inputs; this case has $(length(integrated))"))
    enumerated = [i for (i, t) in enumerate(treatments) if t isa Enumerated]
    combinations = Iterators.product((1:length(treatments[i].values) for i in enumerated)...)
    terms = Float64[]
    for combination in combinations
        assignment = Any[t isa FixedAt ? t.value : nothing for t in treatments]
        logweight = 0.0
        for (i, choice) in zip(enumerated, combination)
            assignment[i] = treatments[i].values[choice]
            logweight += treatments[i].logweights[choice]
        end
        function integrand(point)
            local_assignment = copy(assignment)
            density = 0.0
            for (position, i) in enumerate(integrated)
                local_assignment[i] = point[position]
                density += logpdf(treatments[i].distribution, point[position])
            end
            value = g(NamedTuple{Tuple(names)}(Tuple(local_assignment)))
            return mode === :bp ? exp(value + density) : value * exp(density)
        end
        inner = if isempty(integrated)
            integrand(Float64[])
        else
            lower = [treatments[i].lower for i in integrated]
            upper = [treatments[i].upper for i in integrated]
            first(hcubature(integrand, lower, upper; rtol = 1.0e-10, maxevals = 10^6))
        end
        push!(terms, mode === :bp ? log(inner) + logweight : inner * exp(logweight))
    end
    return mode === :bp ? logsumexp_of(terms) : sum(terms)
end

logsumexp_of(xs) = (m = maximum(xs); m + log(sum(x -> exp(x - m), xs)))

function default_points(output)
    output isa DiscreteUnivariateDistribution && return collect(support(output))
    return [quantile(output, p) for p in range(0.05, 0.95; length = 9)]
end

"""
    verify_message_update(message, logdensity, interfaces, target; m, q, points, atol, source)

Check a message against the node definition it comes from. `message(m, q)` returns the
outgoing message and its log scale (or `nothing`); `logdensity(; interfaces...)` is the
node's log-density; `interfaces` lists its interface names.

With messages as inputs (`m`), the reference is belief propagation, `log ∫ f ∏ mⱼ`; with
marginals (`q`), naive variational message passing, `E_q[log f]`. Point masses are
substituted, finite discrete inputs enumerated, and at most two continuous univariate inputs
integrated.

Two separate assertions:
- **shape** — `logpdf(message, x) − reference(x)` is the same at every test point;
- **scale**, for belief propagation when a log scale is given — that constant is
  `−logscale`, i.e. the message times `exp(logscale)` is the reference itself. Checked only
  when the shape holds, since otherwise there is no single constant to compare.
"""
function verify_message_update(message, logdensity, interfaces, target::Symbol; m = NamedTuple(), q = NamedTuple(), points = nothing, atol = 1.0e-6, source = LineNumberNode(0, :unknown))
    (isempty(m) || isempty(q)) ||
        throw(ArgumentError("verification takes messages (belief propagation) or marginals (variational), not both"))
    mode = isempty(q) ? :bp : :vmp
    inputs = mode === :bp ? m : q
    others = Tuple(name for name in interfaces if name !== target)
    for name in others
        haskey(inputs, name) || throw(ArgumentError("verification needs an input for every other interface; `$name` is missing"))
    end
    treatments = [input_treatment(name, inputs[name]) for name in others]
    output, logscale = message(m, q)
    test_points = something(points, default_points(output))
    names = (target, others...)

    differences = map(test_points) do x
        g = assignment -> logdensity(; assignment...)
        reference = reference_value(nt -> g(merge(NamedTuple{(target,)}((x,)), nt)), others, treatments, mode)
        logpdf(output, x) - reference
    end
    label = "$(mode === :bp ? "belief propagation" : "variational") message target :$target"
    spread = maximum(differences) - minimum(differences)
    shape = record_check(spread <= atol, :(shape_matches_node_definition), () -> "$label: log-ratio to the node definition varies by $spread over the test points; differences $(differences)", source)
    if shape && mode === :bp && logscale !== nothing
        offset = mean(differences)
        record_check(abs(offset + logscale) <= atol, :(logscale_matches_node_definition), () -> "$label: the node definition implies log scale $(-offset), the rule declares $logscale", source)
    end
    return differences
end

"""
    verify_message_update_rule(node, target; m, q, algorithm, points, atol)

[`verify_message_update`](@ref) for a rule defined with the base package: its output
against `nodefunction(node)`, with the log scale it declares.
"""
function verify_message_update_rule(node, target::Symbol; m = NamedTuple(), q = NamedTuple(), algorithm = MessagePassingRulesBase.default_algorithm(node), points = nothing, atol = 1.0e-6, source = LineNumberNode(0, :unknown))
    resolved_target = MessagePassingRulesBase.as_target(target)
    function message(m, q)
        args = MessagePassingRulesBase.RuleArgs(m = m, q = q)
        spec = MessagePassingRulesBase.find_message_rule(node, resolved_target, algorithm, args)
        spec isa MessagePassingRulesBase.RuleSpec || throw(MessagePassingRulesBase.RuleNotFoundError(spec))
        record_selected_rule!(spec, source)
        output, logscale = MessagePassingRulesBase.execute_rule_with_logscale(
            spec, nothing, nothing, MessagePassingRulesBase.rule_algorithm(spec, algorithm), MessagePassingRulesBase.RuleContext(), args,
            MessagePassingRulesBase.NoAnnotations(), resolved_target,
        )
        return output, logscale isa Real ? logscale : nothing
    end
    return verify_message_update(message, MessagePassingRulesBase.nodefunction(node), MessagePassingRulesBase.interfaces(node), target; m, q, points, atol, source)
end

"""
    @verify_message_update_rule(node = ..., target = ..., m = (...), q = (...), ...)

[`verify_message_update_rule`](@ref), written with keywords; failures point at this line.
"""
macro verify_message_update_rule(args...)
    return esc(table_macro_call("verify_message_update_rule", verify_message_update_rule, (:node, :target), args, __source__))
end
