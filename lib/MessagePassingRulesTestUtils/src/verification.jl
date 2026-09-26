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

# Parts of the verification docstrings that several share, written once and interpolated.

const DOC_VERIFY_REFERENCE = rstrip(
    """
    The reference depends on the kind of inputs, which is one or the other:

    - with messages, `m`, belief propagation: ``\\log \\int f(x, y) \\prod_j m_j(y_j) \\, dy``,
      the node's density `f` integrated against the normalised inbound messages;
    - with marginals, `q`, naive variational message passing:
      ``\\mathrm{E}_{q}[\\log f(x, y)]``.

    Each input is treated by its kind: a `PointMass` is substituted; a discrete univariate
    distribution with fewer than 10 000 values in its support is enumerated; a continuous
    univariate distribution is integrated by `HCubature` between its `1e-9` and `1 - 1e-9`
    quantiles, for at most two such inputs per call.

    Two checks are made, each a `Test` assertion:

    - **shape**: `logpdf(message, x) - reference(x)` is the same at every test point, within
      `atol`, so the message is the reference up to a constant;
    - **scale**, for belief propagation when the message comes with a log scale: that constant
      is `-logscale`, within `atol`, so the message times `exp(logscale)` is the reference
      itself. It is checked only when the shape holds, since otherwise there is no single
      constant to compare.
    """
)

const DOC_VERIFY_KEYWORDS = rstrip(
    """
    - `m`: the inbound messages, a `NamedTuple` keyed by interface name, with one entry for
      every interface but `target`. Default: none.
    - `q`: the marginals, keyed the same way. Default: none. Give `m` or `q`, not both.
    - `points`: the values of the target at which to compare. Default: `nothing`, which is the
      support of a discrete message, or 9 quantiles of the message, from `0.05` to `0.95`.
    - `atol`: the absolute tolerance on the log-ratio's spread and on the log scale. Default:
      `1e-6`.
    - `source`: the line checks are reported against. Default: `LineNumberNode(0, :unknown)`;
      the macro form passes its own line.
    """
)

const DOC_VERIFY_THROWS = rstrip(
    """
    - `ArgumentError` when both `m` and `q` are given, when an interface other than `target`
      has no input, when an input is neither a point mass nor a univariate distribution, when a
      discrete input's support has 10 000 values or more, or when more than two inputs need
      integrating.
    """
)

"""
    verify_message_update(message, logdensity, interfaces, target::Symbol; m = (;), q = (;),
        points = nothing, atol = 1e-6) -> Vector{Float64}

Check a message against the node definition it comes from, for any implementation of a rule:
the message must be, up to a constant, the reference computed numerically from the node's
log-density. [`verify_message_update_rule`](@ref) does this for a rule defined with the base
package.

$(DOC_VERIFY_REFERENCE)

# Arguments

- `message`: a function `(m, q) -> (message, logscale)`, called once with the keywords `m` and
  `q`, returning the outbound message, a univariate distribution, and its log scale, a number,
  or `nothing` to leave the scale unchecked.
- `logdensity`: the node's log-density, called with one keyword per interface:
  `logdensity(; out, μ, σ)`.
- `interfaces`: the node's interface names, `target` among them.
- `target`: the interface the message goes to.

# Keywords

$(DOC_VERIFY_KEYWORDS)

# Returns

The log-ratio `logpdf(message, x) - reference(x)` at each test point, in order.

# Throws

$(DOC_VERIFY_THROWS)
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
    verify_message_update_rule(node, target::Symbol; m = (;), q = (;), algorithm = default_algorithm(node),
        points = nothing, atol = 1e-6) -> Vector{Float64}

Check the message rule of `node` towards `target` against the node's definition: the message it
returns, and the log scale it declares, against its
[`nodefunction`](@extref MessagePassingRulesBase.nodefunction), integrated numerically.
[`@verify_message_update_rule`](@ref) is the same check written with keywords, and reports
failures at its own line. The rule is recorded as selected for [`check_rule_coverage`](@ref).

$(DOC_VERIFY_REFERENCE)

The rule's log scale is checked when it declares one; an undefined log scale leaves the scale
unchecked.

Limitations: the node must be stochastic and without groups, since only such a node has a
`nodefunction`; `target` is a single interface, not a group member; the message must be a
univariate distribution; the rule runs with an empty
[`RuleContext`](@extref MessagePassingRulesBase.RuleContext), so a rule that reads services
gets `nothing` for them; and a rule that takes messages and marginals together cannot be
verified, since the inputs are one kind or the other. Such rules are checked by their tables.

# Arguments

- `node`: the node, as declared with
  [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node).
- `target`: the interface the message goes to, a `Symbol`.

# Keywords

$(DOC_VERIFY_KEYWORDS)
- `algorithm`: the algorithm value to run under. Default:
  [`default_algorithm`](@extref MessagePassingRulesBase.default_algorithm)`(node)`.

# Returns

The log-ratio `logpdf(message, x) - reference(x)` at each test point, in order.

# Throws

$(DOC_VERIFY_THROWS)
- [`RuleNotFoundError`](@extref MessagePassingRulesBase.RuleNotFoundError) when no rule
  takes the inputs.
- `MethodError` when `node` has no `nodefunction`.

# Examples

```julia
@verify_message_update_rule(node = NormalMeanVariance, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
@verify_message_update_rule(node = NormalMeanVariance, target = :μ, q = (out = NormalMeanVariance(1.0, 2.0), v = InverseGamma(3.0, 4.0)))
```
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

Check the message rule of a node towards one of its interfaces against the node's definition,
integrated numerically. It takes the arguments and keywords of
[`verify_message_update_rule`](@ref), where the checks and their limits are described, all by
name, and returns what it returns.

$(doc_keyword_macro("verify_message_update_rule", "`node` and `target` are"))

# Examples

```julia
@verify_message_update_rule(node = NormalMeanVariance, target = :out, m = (μ = NormalMeanVariance(0.5, 1.5), v = PointMass(2.0)))
```
"""
macro verify_message_update_rule(args...)
    return esc(table_macro_call("verify_message_update_rule", verify_message_update_rule, (:node, :target), args, __source__))
end
