# Routing through the definition macros is devirtualized: a rule call resolves at compile time
# and allocates nothing. The test reports the figures rather than only asserting them. Each
# call is measured inside a function, never at global scope, where dynamic dispatch would
# allocate and skew the number.

@testmodule MacroGate begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, Target, IndexedTarget, DefaultAlgorithm, AbstractAlgorithm, DefaultAlgorithmExtension

    # A second rule set, standing alone; and an extension, which inherits the default rules.
    struct Alternative <: AbstractAlgorithm end
    struct Extension <: DefaultAlgorithmExtension end

    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :v, :p...])
    @define_message_update_rule(node = Gauss, target = :out, args = (m[:μ]::Float64, m[:v]::Float64), body = (args) -> args.m[:μ] + args.m[:v])
    @define_message_update_rule(node = Gauss, target = :out, algorithm = Alternative, args = (m[:μ]::Float64, m[:v]::Float64), body = (args) -> args.m[:μ] * args.m[:v])
    @define_message_update_rule(node = Gauss, target = (:p, k), args = (q[:p][k]::Float64,), body = (args) -> 2 * args.q[:p][k])
    @define_message_update_rule(node = Gauss, target = :μ, args = (m[:out]::Float64,), body = (args) -> fill(args.m[:out], 4))

    const POINT = RuleArgs(m = (μ = 1.0, v = 2.0))
    const MEMBER = RuleArgs(q = (p = (nothing, 5.0),))
    const OUT = RuleArgs(m = (out = 1.0,))

    one_rule() = getresult(message_passing_rule(Gauss, Target(:out), DefaultAlgorithm(), POINT))
    indexed() = getresult(message_passing_rule(Gauss, IndexedTarget(:p, 2), DefaultAlgorithm(), MEMBER))
    # A call site whose algorithm is only known at run time can reach two rules.
    two_rules(flag::Bool) = getresult(message_passing_rule(Gauss, Target(:out), flag ? DefaultAlgorithm() : Alternative(), POINT))
    # An extension with no rule of its own reaches the default's through the fallback.
    inherited() = getresult(message_passing_rule(Gauss, Target(:out), Extension(), POINT))
    # Negative control: the body allocates, so the gate must see it.
    allocating() = getresult(message_passing_rule(Gauss, Target(:μ), DefaultAlgorithm(), OUT))

    measure(f, args...) = (f(args...); @allocated f(args...))
end

@testitem "gate:routing-macros" tags = [:base, :alloc] setup = [MacroGate] begin
    using JET
    G = MacroGate
    one = G.measure(G.one_rule)
    indexed = G.measure(G.indexed)
    two = G.measure(G.two_rules, time() > 0)
    control = G.measure(G.allocating)
    inherited = G.measure(G.inherited)
    two_type = only(Base.return_types(G.two_rules, (Bool,)))
    @info "routing through the macros" VERSION one indexed two two_inferred = two_type inherited control

    @test one == 0
    @test indexed == 0
    # The fallback from an extension to the default rules is resolved at compile time.
    @test inherited == 0
    @test (@inferred G.inherited()) === 3.0
    @test control > 0
    @test (@inferred G.one_rule()) === 3.0
    @test (@inferred G.indexed()) === 10.0
    @test G.two_rules(true) == 3.0 && G.two_rules(false) == 2.0
    JET.@test_opt G.one_rule()
    JET.@test_opt G.indexed()
    JET.@test_opt G.inherited()
end
