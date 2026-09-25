# One default algorithm, and custom algorithms of two kinds: a direct subtype of
# `AbstractAlgorithm` stands alone, a `DefaultAlgorithmExtension` inherits the default rules
# and dependencies for whatever it does not define itself.

@testmodule AlgorithmRules begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: AbstractAlgorithm, DefaultAlgorithmExtension

    struct Gauss end
    @define_factor_node(
        node = Gauss, type = Stochastic, interfaces = [:out, :μ, :v],
        dependencies = [:out => (m[:μ], m[:v])],
    )
    # Default rules. The `:out` rule returns the algorithm it received, to show which one.
    @define_message_update_rule(node = Gauss, target = :out, args = (m[:μ]::Float64, m[:v]::Float64), body = (algo, args) -> (algo, args.m[:μ] + args.m[:v]))
    @define_message_update_rule(node = Gauss, target = :μ, args = (m[:out]::Float64, m[:v]::Float64), body = (args) -> :default)

    # Overrides one rule, with broader inputs than the default's: under subtype dispatch this
    # would be ambiguous with it, and must simply win instead.
    struct Tweaked <: DefaultAlgorithmExtension end
    @define_message_update_rule(node = Gauss, target = :μ, algorithm = Tweaked, args = (m[:out]::Real, m[:v]::Real), body = (args) -> :tweaked)

    # Inherits everything; and one that declares dependencies of its own.
    struct Plain <: DefaultAlgorithmExtension end
    struct OwnDependencies <: DefaultAlgorithmExtension end
    @define_dependencies(node = Gauss, algorithm = OwnDependencies, dependencies = [:out => (m[:μ],)])

    struct Alone <: AbstractAlgorithm end
end

@testitem "algorithm:default" tags = [:base] setup = [AlgorithmRules] begin
    using MessagePassingRulesBase: default_algorithm, RuleArgs, Target
    A = AlgorithmRules

    @test default_algorithm(A.Gauss) === DefaultAlgorithm()
    args = RuleArgs(m = (μ = 1.0, v = 2.0))
    @test getresult(message_passing_rule(A.Gauss, Target(:out), DefaultAlgorithm(), args)) == (DefaultAlgorithm(), 3.0)
end

@testitem "algorithm:extension-inherits" tags = [:base] setup = [AlgorithmRules] begin
    using MessagePassingRulesBase: RuleArgs, Target, RuleSpec, find_message_rule, rule_algorithm
    A = AlgorithmRules
    point = RuleArgs(m = (μ = 1.0, v = 2.0))
    out_point = RuleArgs(m = (out = 1.0, v = 2.0))

    # No rule of its own: the default's, run with `DefaultAlgorithm()` in its `algo` slot.
    @test getresult(message_passing_rule(A.Gauss, Target(:out), A.Tweaked(), point)) == (DefaultAlgorithm(), 3.0)
    inherited = find_message_rule(A.Gauss, Target(:out), A.Tweaked(), point)
    @test inherited isa RuleSpec && inherited.algorithm === DefaultAlgorithm
    @test rule_algorithm(inherited, A.Tweaked()) === DefaultAlgorithm()

    # Its own rule wins, even though the default's is more specific in the inputs.
    @test getresult(message_passing_rule(A.Gauss, Target(:μ), A.Tweaked(), out_point)) === :tweaked
    own = find_message_rule(A.Gauss, Target(:μ), A.Tweaked(), out_point)
    @test rule_algorithm(own, A.Tweaked()) === A.Tweaked()
    @test getresult(message_passing_rule(A.Gauss, Target(:μ), DefaultAlgorithm(), out_point)) === :default
    @test getresult(message_passing_rule(A.Gauss, Target(:μ), A.Plain(), out_point)) === :default
end

@testitem "algorithm:standalone-inherits-nothing" tags = [:base] setup = [AlgorithmRules] begin
    using MessagePassingRulesBase: RuleArgs, Target, RuleNotFound, find_message_rule
    A = AlgorithmRules
    nf = find_message_rule(A.Gauss, Target(:out), A.Alone(), RuleArgs(m = (μ = 1.0, v = 2.0)))
    @test nf isa RuleNotFound
    @test nf.algorithm === A.Alone()
end

@testitem "algorithm:not-found-names-the-extension" tags = [:base] setup = [AlgorithmRules] begin
    using MessagePassingRulesBase: RuleArgs, Target, RuleNotFound, RuleNotFoundError, find_message_rule
    A = AlgorithmRules
    args = RuleArgs(m = (μ = "one", v = 2.0))
    nf = find_message_rule(A.Gauss, Target(:out), A.Plain(), args)
    @test nf isa RuleNotFound && nf.algorithm === A.Plain()
    text = sprint(showerror, RuleNotFoundError(nf))
    @test contains(text, "under $(A.Plain())")
    # The default rule it would have inherited is a near miss, with the algorithm accepted.
    @test contains(text, "type mismatch")
    @test contains(text, "✓ algorithm $(DefaultAlgorithm)")
end

@testitem "algorithm:dependencies" tags = [:base] setup = [AlgorithmRules] begin
    using MessagePassingRulesBase: dependencies_spec
    A = AlgorithmRules
    default = dependencies_spec(A.Gauss, DefaultAlgorithm())
    @test default !== nothing
    @test dependencies_spec(A.Gauss, A.Plain()) === default
    @test dependencies_spec(A.Gauss, A.OwnDependencies()) !== default
    @test dependencies_spec(A.Gauss, A.OwnDependencies()).algorithm === A.OwnDependencies
    @test dependencies_spec(A.Gauss, A.Alone()) === nothing
end

@testitem "algorithm:queries" tags = [:base] setup = [AlgorithmRules] begin
    using MessagePassingRulesBase: list_rules, check_rules, check_rule_ambiguities
    A = AlgorithmRules
    @test length(list_rules(A.Gauss; algorithm = DefaultAlgorithm())) == 2
    # An extension can select its own rule and the default's.
    @test length(list_rules(A.Gauss; algorithm = A.Tweaked())) == 3
    @test length(list_rules(A.Gauss; algorithm = A.Plain())) == 2
    @test isempty(list_rules(A.Gauss; algorithm = A.Alone()))
    # An extension's rule is checked against the dependencies it inherits, and overriding a
    # default rule is not an ambiguity.
    @test isempty(check_rules(A))
    @test isempty(check_rule_ambiguities(A))
end
