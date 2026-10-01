central_difference(f, θ::Real, step) = (f(θ + step) - f(θ - step)) / (2step)
function central_difference(f, θ::AbstractVector, step)
    return map(eachindex(θ)) do i
        e = zeros(eltype(θ), length(θ))
        e[i] = step
        (f(θ .+ e) - f(θ .- e)) / (2step)
    end
end

automatic_derivative(f, θ::Real) = ForwardDiff.derivative(f, θ)
automatic_derivative(f, θ::AbstractVector) = ForwardDiff.gradient(f, θ)

"""
    test_rule_derivatives(node, target; inputs, at, summary = mean, algorithm = default_algorithm(node),
        step = 1e-6, rtol = 1e-5, atol = 1e-8) -> nothing

Check that derivatives propagate through the message rule of `node` towards `target`: the
derivative of `θ -> summary(rule(inputs(θ)))` by ForwardDiff must agree with a central finite
difference at `θ = at`. A rule that drops the dual part of its inputs fails the comparison. One
that cannot take dual numbers, or throws at a perturbed `θ`, fails as `rule_runs` with the
exception; no rule for the inputs, at `at` or at a dual or perturbed `θ`, fails as
`rule_found` with the near misses. Nothing is thrown for a rule, and the checks after it
still run. [`@test_rule_derivatives`](@ref) is the same check written with keywords, and
reports failures at its own line.

The rule is resolved for the inputs at each `θ`, and the one found at `at` is recorded as
selected for [`check_rule_coverage`](@ref). An in-place rule is checked a second time through
`rule!`, into a buffer its `preallocate` makes from the dual inputs, so both paths are covered.
The rule runs with an empty [`RuleContext`](@extref MessagePassingRulesBase.RuleContext), and
its log scale is not checked.

# Arguments

- `node`: the node, as declared with
  [`@define_factor_node`](@extref MessagePassingRulesBase.@define_factor_node).
- `target`: the interface the message goes to: `:out`, or `(:m, k)` for member `k` of the group
  `m`.

# Keywords

- `inputs`: a function of `θ` returning a `NamedTuple` with the rule's inbound messages `m` and
  marginals `q`: `θ -> (m = (out = Normal(θ, 2.0), σ = PointMass(θ^2)),)`. Required. Joint
  marginals of clusters are not supported.
- `at`: the point to differentiate at, a number or a vector. Required.
- `summary`: a function from the rule's output to a number, whose derivative is compared.
  Default: `mean`.
- `algorithm`: the algorithm value to run under. Default:
  [`default_algorithm`](@extref MessagePassingRulesBase.default_algorithm)`(node)`.
- `step`: the finite-difference step. Default: `1e-6`.
- `rtol`, `atol`: the tolerances of the comparison, element by element for a vector `at`.
  Defaults: `1e-5` and `1e-8`.
- `source`: the line checks are reported against. Default: `LineNumberNode(0, :unknown)`;
  the macro form passes its own line.
"""
function test_rule_derivatives(node, target; inputs::Function, at, summary = mean, algorithm = MessagePassingRulesBase.default_algorithm(node), step = 1.0e-6, rtol = 1.0e-5, atol = 1.0e-8, source = LineNumberNode(0, :unknown))
    resolved_target = MessagePassingRulesBase.as_target(target)
    ctx = MessagePassingRulesBase.RuleContext()
    args_at(θ) = (i = inputs(θ); MessagePassingRulesBase.RuleArgs(m = get(i, :m, NamedTuple()), q = get(i, :q, NamedTuple())))
    function resolved(args)
        spec = MessagePassingRulesBase.find_message_rule(node, resolved_target, algorithm, args)
        spec isa MessagePassingRulesBase.RuleSpec || throw(MessagePassingRulesBase.RuleNotFoundError(spec))
        return spec
    end
    # The algorithm the resolved rule runs with, as the engine passes it: an inherited default
    # rule gets `DefaultAlgorithm()`, not the extension it was reached through.
    run(spec, args, output) = MessagePassingRulesBase.execute_rule(spec, output, MessagePassingRulesBase.rule_algorithm(spec, algorithm), ctx, args, MessagePassingRulesBase.NoAnnotations(), resolved_target)
    run(args, output) = run(resolved(args), args, output)
    prealloc(spec, args) = spec.prealloc(MessagePassingRulesBase.rule_algorithm(spec, algorithm), ctx, args, resolved_target)

    described = "the rule for $node target $resolved_target"
    found, spec = guarded(() -> resolved(args_at(at)), () -> "$described at $at", source)
    found || return nothing
    record_selected_rule!(spec, source)
    paths = Tuple{String, Function}[("rule", θ -> summary(run(args_at(θ), nothing)))]
    if spec.inplace
        push!(paths, ("rule!", θ -> (args = args_at(θ); found = resolved(args); summary(run(found, args, prealloc(found, args))))))
    end
    for (path, f) in paths
        label = "derivative through `$path` of $described at $at"
        differentiated, automatic = guarded(() -> automatic_derivative(f, at), () -> "$label, by ForwardDiff", source)
        differenced, numeric = guarded(() -> central_difference(f, at, step), () -> "$label, by a central difference", source)
        (differentiated && differenced) || continue
        ok = all(isapprox.(automatic, numeric; rtol, atol))
        record_check(ok, :(automatic_derivative ≈ finite_difference), () -> "$label: ForwardDiff gives $automatic, a central difference $numeric", source)
    end
    return nothing
end

"""
    @test_rule_derivatives(node = ..., target = ..., inputs = θ -> (m = (...),), at = ..., ...)

Check that derivatives propagate through the message rule of a node towards one of its
interfaces, by ForwardDiff against a central finite difference. It takes the arguments and
keywords of [`test_rule_derivatives`](@ref), where they are described, all by name.

$(doc_keyword_macro("test_rule_derivatives", "`node` and `target` are"))

# Examples

```julia
@test_rule_derivatives(
    node = NormalMeanVariance, target = :out,
    inputs = θ -> (m = (μ = NormalMeanVariance(θ, 1.0), v = PointMass(θ^2)),),
    at = 1.5, summary = d -> mean(d) + var(d),
)
```
"""
macro test_rule_derivatives(args...)
    return esc(table_macro_call("test_rule_derivatives", test_rule_derivatives, (:node, :target), args, __source__))
end
