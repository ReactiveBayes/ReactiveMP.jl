# The Phase 0 devirtualization gate, re-run through the definition macros. It reports the
# figures rather than only asserting them; see DISCUSSION.md §3.15 for the spike's numbers
# and for the mistakes that make this measurement lie.

@testmodule MacroGate begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, Target, IndexedTarget, BP, VMP

    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :v, :p...])
    @define_message_update_rule(node = Gauss, towards = :out, args = (m[:μ]::Float64, m[:v]::Float64), body = (args) -> args.m[:μ] + args.m[:v])
    @define_message_update_rule(node = Gauss, towards = :out, algorithm = VMP, args = (m[:μ]::Float64, m[:v]::Float64), body = (args) -> args.m[:μ] * args.m[:v])
    @define_message_update_rule(node = Gauss, towards = (:p, k), args = (q[:p][k]::Float64,), body = (args) -> 2 * args.q[:p][k])
    @define_message_update_rule(node = Gauss, towards = :μ, args = (m[:out]::Float64,), body = (args) -> fill(args.m[:out], 4))

    const POINT = RuleArgs(m = (μ = 1.0, v = 2.0))
    const MEMBER = RuleArgs(q = (p = (nothing, 5.0),))
    const OUT = RuleArgs(m = (out = 1.0,))

    one_rule() = message_passing_rule(Gauss, Target(:out), BP(), POINT)
    indexed() = message_passing_rule(Gauss, IndexedTarget(:p, 2), BP(), MEMBER)
    # A call site whose algorithm is only known at run time can reach two rules.
    two_rules(flag::Bool) = message_passing_rule(Gauss, Target(:out), flag ? BP() : VMP(), POINT)
    # Negative control: the body allocates, so the gate must see it.
    allocating() = message_passing_rule(Gauss, Target(:μ), BP(), OUT)

    measure(f, args...) = (f(args...); @allocated f(args...))
end

@testitem "gate:routing-macros" tags = [:base, :alloc] setup = [MacroGate] begin
    using JET
    G = MacroGate
    one = G.measure(G.one_rule)
    indexed = G.measure(G.indexed)
    two = G.measure(G.two_rules, time() > 0)
    control = G.measure(G.allocating)
    two_type = only(Base.return_types(G.two_rules, (Bool,)))
    @info "routing through the macros" VERSION one indexed two two_inferred = two_type control

    @test one == 0
    @test indexed == 0
    @test control > 0
    @test (@inferred G.one_rule()) === 3.0
    @test (@inferred G.indexed()) === 10.0
    @test G.two_rules(true) == 3.0 && G.two_rules(false) == 2.0
    JET.@test_opt G.one_rule()
    JET.@test_opt G.indexed()
end
