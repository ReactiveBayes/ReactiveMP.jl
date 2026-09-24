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
    test_rule_derivatives(node, target; inputs, at, summary = mean, algorithm, step = 1e-6, rtol = 1e-5, atol = 1e-8)

Check that derivatives propagate through a message rule: `ForwardDiff` of
`θ -> summary(rule(inputs(θ)))` must agree with a central finite difference at `θ = at`.
`inputs(θ)` returns a named tuple of `m` and `q`; `θ` is a number or a vector. An in-place
rule is checked a second time through `rule!`, into a buffer preallocated from the dual
inputs, so both paths are covered.
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

    spec = resolved(args_at(at))
    record_selected_rule!(spec, source)
    paths = Tuple{String, Function}[("rule", θ -> summary(run(args_at(θ), nothing)))]
    if spec.inplace
        push!(paths, ("rule!", θ -> (args = args_at(θ); found = resolved(args); summary(run(found, args, prealloc(found, args))))))
    end
    for (path, f) in paths
        automatic = automatic_derivative(f, at)
        numeric = central_difference(f, at, step)
        ok = all(isapprox.(automatic, numeric; rtol, atol))
        record_check(ok, :(automatic_derivative ≈ finite_difference), () -> "derivative through `$path` of the rule for $node target $resolved_target at $at: ForwardDiff gives $automatic, a central difference $numeric", source)
    end
    return nothing
end

"""
    @test_rule_derivatives(node = ..., target = ..., inputs = θ -> (m = (...),), at = ..., ...)

[`test_rule_derivatives`](@ref), written with keywords; failures point at this line.
"""
macro test_rule_derivatives(args...)
    return esc(table_macro_call("test_rule_derivatives", test_rule_derivatives, (:node, :target), args, __source__))
end
