# Opt-in audits of the rules a graph runs, off by default, each naming the rule it objects to:
# impure rules, rules with no in-place form, and scratch read before it is written.

@testmodule DiagnosticsNodes begin
    using MessagePassingRulesBase

    # Towards `out` pure and in place; towards `in` impure, and not in place.
    struct Audited end
    @define_factor_node(node = Audited, type = Stochastic, interfaces = [:out, :in])
    @define_message_update_rule(
        node = Audited, target = :out, inplace = true, args = (m[:in]::Vector{Float64},),
        preallocate = (args) -> similar(args.m[:in]),
        body = (output, args) -> (output .= args.m[:in] .+ 1),
    )
    @define_message_update_rule(node = Audited, target = :in, pure = false, args = (m[:out]::Vector{Float64},), body = (args) -> args.m[:out] .- 1)
    @define_marginal_update_rule(
        node = Audited, target = (:out, :in), pure = false, args = (m[:out]::Vector{Float64}, m[:in]::Vector{Float64}),
        body = (args) -> args.m[:out] .+ args.m[:in],
    )

    # Reads its scratch before writing it: a stale value leaks into the result on reuse.
    struct Stale end
    @define_factor_node(node = Stale, type = Stochastic, interfaces = [:out, :in])
    @define_message_update_rule(
        node = Stale, target = :out, args = (m[:in]::Vector{Float64},),
        scratch = (args) -> (acc = zeros(length(args.m[:in])),),
        body = (scratch, args) -> (scratch.acc .+= args.m[:in]; copy(scratch.acc)),
    )
end

@testitem "diagnostics:off by default" tags = [:engine] setup = [DiagnosticsNodes] begin
    import ReactiveMP: MessageMapping, EngineDiagnostics, FactorNodeActivationOptions, getdata
    import MessagePassingRulesBase: Target, DefaultAlgorithm
    N = DiagnosticsNodes

    @test FactorNodeActivationOptions().diagnostics == EngineDiagnostics()
    @test !EngineDiagnostics().check_everything_pure && !EngineDiagnostics().check_everything_inplace && !EngineDiagnostics().checked_buffers
    # Without them, an impure rule runs.
    towards_in = MessageMapping(N.Audited, Target{:in}(), Val((:out,)), nothing, DefaultAlgorithm(), nothing, N.Audited(), nothing)
    @test getdata(towards_in((Message([2.0], false, false),), nothing)) == [1.0]
end

@testitem "diagnostics:check_everything_pure" tags = [:engine] setup = [DiagnosticsNodes] begin
    import ReactiveMP: MessageMapping, MarginalMapping, EngineDiagnostics, ImpureRuleError, getdata, compute_marginal
    import MessagePassingRulesBase: Target, ClusterTarget, DefaultAlgorithm
    N = DiagnosticsNodes
    diagnostics = EngineDiagnostics(check_everything_pure = true)

    # A pure rule runs; an impure one is an error that names it.
    towards_out = MessageMapping(N.Audited, Target{:out}(), Val((:in,)), nothing, DefaultAlgorithm(), nothing, N.Audited(), nothing, diagnostics)
    @test getdata(towards_out((Message([1.0], false, false),), nothing)) == [2.0]
    towards_in = MessageMapping(N.Audited, Target{:in}(), Val((:out,)), nothing, DefaultAlgorithm(), nothing, N.Audited(), nothing, diagnostics)
    err = try
        towards_in((Message([2.0], false, false),), nothing)
        nothing
    catch e
        e
    end
    @test err isa ImpureRuleError
    text = sprint(showerror, err)
    @test contains(text, "Audited") && contains(text, ":in") && contains(text, "diagnostics_tests.jl") && contains(text, "check_everything_pure")

    # Marginal rules too.
    joint = MarginalMapping(N.Audited, ClusterTarget((:out, :in)), Val((:out, :in)), nothing, DefaultAlgorithm(), N.Audited(), diagnostics)
    @test_throws ImpureRuleError compute_marginal(joint, (Message([1.0], false, false), Message([1.0], false, false)), nothing)
end

@testitem "diagnostics:check_everything_inplace" tags = [:engine] setup = [DiagnosticsNodes] begin
    import ReactiveMP: MessageMapping, EngineDiagnostics, getdata
    import MessagePassingRulesBase: Target, DefaultAlgorithm
    using Test: @test_logs
    N = DiagnosticsNodes
    diagnostics = EngineDiagnostics(check_everything_inplace = true)

    # A coverage report, not an error: a rule with no in-place form is named once, and runs.
    towards_in = MessageMapping(N.Audited, Target{:in}(), Val((:out,)), nothing, DefaultAlgorithm(), nothing, N.Audited(), nothing, diagnostics)
    @test_logs (:warn, r"Audited` towards :in.*no in-place form") getdata(towards_in((Message([2.0], false, false),), nothing))
    @test_logs getdata(towards_in((Message([2.0], false, false),), nothing))
    # One with an in-place form is not.
    towards_out = MessageMapping(N.Audited, Target{:out}(), Val((:in,)), nothing, DefaultAlgorithm(), nothing, N.Audited(), nothing, diagnostics)
    @test_logs getdata(towards_out((Message([1.0], false, false),), nothing))
end

@testitem "diagnostics:checked_buffers" tags = [:engine] setup = [DiagnosticsNodes] begin
    import ReactiveMP: MessageMapping, EngineDiagnostics, getdata, poison!
    import MessagePassingRulesBase: Target, DefaultAlgorithm
    N = DiagnosticsNodes

    # Unchecked, a rule reading its scratch before writing it carries a stale value along.
    unchecked = MessageMapping(N.Stale, Target{:out}(), Val((:in,)), nothing, DefaultAlgorithm(), nothing, N.Stale(), nothing)
    getdata(unchecked((Message([1.0], false, false),), nothing))
    @test getdata(unchecked((Message([1.0], false, false),), nothing)) == [2.0]

    # Checked, the scratch is poisoned before each reuse, so the stale read shows as NaN.
    checked = MessageMapping(N.Stale, Target{:out}(), Val((:in,)), nothing, DefaultAlgorithm(), nothing, N.Stale(), nothing, EngineDiagnostics(checked_buffers = true))
    @test getdata(checked((Message([1.0], false, false),), nothing)) == [1.0]
    @test all(isnan, getdata(checked((Message([1.0], false, false),), nothing)))

    # Poisoning reaches floating-point arrays inside tuples and named tuples, and leaves the rest.
    scratch = (a = [1.0, 2.0], b = ([3.0f0], [1, 2]), c = 5)
    poison!(scratch)
    @test all(isnan, scratch.a) && all(isnan, scratch.b[1]) && scratch.b[2] == [1, 2]
end

@testitem "diagnostics:through activation" tags = [:engine] setup = [DiagnosticsNodes] begin
    import ReactiveMP: activate!, factornode, getinterfaces, get_stream_of_inbound_messages, get_stream_of_marginals, set_initial_message!,
        FactorNodeActivationOptions, RandomVariableActivationOptions, MessageProductContext, EngineDiagnostics, ImpureRuleError
    using Rocket
    N = DiagnosticsNodes

    # The option reaches the node's streams: under q(out, in), the message towards `in` runs the
    # impure rule once `out` sends one, and the audit stops it.
    function run(diagnostics)
        out, in = randomvar(), randomvar()
        node = factornode(N.Audited, [(:out, out), (:in, in)], ((:out, :in),))
        product = MessageProductContext()
        foreach(v -> activate!(v, RandomVariableActivationOptions(nothing, product, product)), (out, in))
        activate!(node, FactorNodeActivationOptions(; diagnostics))
        received = []
        subscription = subscribe!(get_stream_of_marginals(in), (q) -> push!(received, ReactiveMP.getdata(q)))
        set_initial_message!(get_stream_of_inbound_messages(first(getinterfaces(node))), [2.0])
        unsubscribe!(subscription)
        return received
    end
    @test run(EngineDiagnostics()) == [[1.0]]
    @test_throws ImpureRuleError run(EngineDiagnostics(check_everything_pure = true))
end
