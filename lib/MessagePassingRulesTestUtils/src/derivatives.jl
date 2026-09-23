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
    test_rule_derivatives(node, towards; inputs, at, summary = mean, algorithm, step = 1e-6, rtol = 1e-5, atol = 1e-8)

Check that derivatives propagate through a message rule: `ForwardDiff` of
`θ -> summary(rule(inputs(θ)))` must agree with a central finite difference at `θ = at`.
`inputs(θ)` returns a named tuple of `m` and `q`; `θ` is a number or a vector. An in-place
rule is checked a second time through `rule!`, into a buffer preallocated from the dual
inputs, so both paths are covered.
"""
function test_rule_derivatives(node, towards; inputs::Function, at, summary = mean, algorithm = MessagePassingRulesBase.default_algorithm(node), step = 1.0e-6, rtol = 1.0e-5, atol = 1.0e-8, source = LineNumberNode(0, :unknown))
    target = MessagePassingRulesBase.as_target(towards)
    ctx = MessagePassingRulesBase.RuleContext()
    args_at(θ) = (i = inputs(θ); MessagePassingRulesBase.RuleArgs(m = get(i, :m, NamedTuple()), q = get(i, :q, NamedTuple())))
    function resolved(args)
        spec = MessagePassingRulesBase.find_message_rule(node, target, algorithm, args)
        spec isa MessagePassingRulesBase.RuleSpec || throw(MessagePassingRulesBase.RuleNotFoundError(spec))
        return spec
    end
    run(args, output) = MessagePassingRulesBase.execute_rule(resolved(args), output, algorithm, ctx, args, MessagePassingRulesBase.NoAnnotations(), target)

    spec = resolved(args_at(at))
    record_selected_rule!(spec, source)
    paths = Tuple{String, Function}[("rule", θ -> summary(run(args_at(θ), nothing)))]
    if spec.inplace
        push!(paths, ("rule!", θ -> (args = args_at(θ); summary(run(args, resolved(args).prealloc(algorithm, ctx, args, target))))))
    end
    for (path, f) in paths
        automatic = automatic_derivative(f, at)
        numeric = central_difference(f, at, step)
        ok = all(isapprox.(automatic, numeric; rtol, atol))
        record_check(ok, :(automatic_derivative ≈ finite_difference), () -> "derivative through `$path` of the rule for $node towards $target at $at: ForwardDiff gives $automatic, a central difference $numeric", source)
    end
    return nothing
end

"""
    @test_rule_derivatives(node = ..., towards = ..., inputs = θ -> (m = (...),), at = ..., ...)

[`test_rule_derivatives`](@ref), written with keywords; failures point at this line.
"""
macro test_rule_derivatives(args...)
    return esc(table_macro_call("test_rule_derivatives", test_rule_derivatives, (:node, :towards), args, __source__))
end
