@testmodule BrokenRules begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: DefaultAlgorithm, AbstractAlgorithm

    struct MixtureVMP <: AbstractAlgorithm end
    struct Standalone <: AbstractAlgorithm end

    struct Gauss end
    @define_factor_node(node = Gauss, type = Stochastic, interfaces = [:out, :μ, :τ, :p...])

    @define_message_update_rule(node = Gauss, target = :out, args = (m[:μ]::Real, m[:τ]::Real), body = (args) -> 1)
    @define_message_update_rule(node = Gauss, target = :nope, args = (m[:μ]::Real,), body = (args) -> 1)
    @define_message_update_rule(node = Gauss, target = :p, args = (m[:μ]::Real,), body = (args) -> 1)
    @define_message_update_rule(node = Gauss, target = :μ, args = (q[:τ, :out]::Any,), body = (args) -> 1)
    @define_message_update_rule(node = Gauss, target = :τ, args = (m[:p]::Any, m[:μ...]::Any), body = (args) -> 1)
    @define_marginal_update_rule(node = Gauss, target = (:μ, :out), args = (m[:out]::Any,), body = (args) -> 1)

    struct Undeclared end
    @define_message_update_rule(node = Undeclared, target = :out, algorithm = DefaultAlgorithm, args = (), body = () -> 1)

    # Consistent with dependencies, and then not.
    struct Mix end
    @define_factor_node(
        node = Mix, type = Stochastic, interfaces = [:out, :m..., :p...], algorithm = MixtureVMP,
        dependencies = [(:m, k) => (q[:out], q[:p][k]), :out => (q[:m...], q[:p...])],
    )
    @define_message_update_rule(node = Mix, target = (:m, k), args = (q[:out]::Any, q[:p][k]::Any), body = (args) -> 1)
    @define_message_update_rule(node = Mix, target = :out, args = (q[:m...]::Any,), body = (args) -> 1)

    # A declaration that extends the default scheme with `q(a)`: a rule for `a` must read it.
    struct Extended <: AbstractAlgorithm end
    struct Transition end
    @define_factor_node(node = Transition, type = Stochastic, interfaces = [:y, :x, :a])
    @define_dependencies(node = Transition, algorithm = Extended, dependencies = [:y => (default,), :x => (default,), :a => (default, q[:a])])
    @define_message_update_rule(node = Transition, target = :a, algorithm = Extended, args = (q[:y]::Any, q[:x]::Any), body = (args) -> 1)

    # Two rules some call matches equally well.
    struct Amb end
    @define_factor_node(node = Amb, type = Stochastic, interfaces = [:out, :a, :b])
    @define_message_update_rule(node = Amb, target = :out, args = (m[:a]::Float64, m[:b]::Real), body = (args) -> 1)
    @define_message_update_rule(node = Amb, target = :out, args = (m[:a]::Real, m[:b]::Float64), body = (args) -> 2)
    # Disjoint in one slot, so no call matches both: not ambiguous, however the others overlap.
    @define_message_update_rule(node = Amb, target = :b, args = (m[:out]::Float64, m[:a]::Real), body = (args) -> 1)
    @define_message_update_rule(node = Amb, target = :b, args = (m[:out]::String, m[:a]::Float64), body = (args) -> 2)
    # Nested, not ambiguous: one is more specific.
    @define_message_update_rule(node = Amb, target = :a, args = (m[:out]::Real,), body = (args) -> 1)
    @define_message_update_rule(node = Amb, target = :a, args = (m[:out]::Float64,), body = (args) -> 2)
end

@testitem "diagnostics:check_rules" tags = [:base] setup = [BrokenRules] begin
    using MessagePassingRulesBase: check_rules
    B = BrokenRules
    messages = [issue.message for issue in check_rules(B)]
    has(text) = any(m -> contains(m, text), messages)

    @test has("has no interface `nope`")
    @test has("`p` is a group; its targets are written `(:p, k)`")
    @test has("`q[:τ, :out]`: a cluster lists existing interfaces in interface order")
    @test has("`m[:p]`: `p` is a group")
    @test has("`m[:μ...]`: `μ` is not a group")
    @test has("`target = (:μ, :out)`: a cluster lists existing interfaces in interface order")
    @test has("has no declaration")
    @test has("consumes (q[:m...]) but the dependencies of")
    @test has("consumes (q[:x], q[:y]) but the dependencies of Extended add (q[:a]) to the default scheme's inputs")
    @test length(messages) == 9
end

@testitem "diagnostics:clean" tags = [:base] setup = [RepresentativeRules, DependencyNodes, ToyNodes] begin
    using MessagePassingRulesBase: check_rules, check_rule_ambiguities
    @test isempty(check_rules(RepresentativeRules))
    @test isempty(check_rules(DependencyNodes))
    @test isempty(check_rule_ambiguities(RepresentativeRules))
end

@testitem "diagnostics:ambiguities" tags = [:base] setup = [BrokenRules] begin
    using MessagePassingRulesBase: check_rule_ambiguities, Target
    pairs = check_rule_ambiguities(BrokenRules)
    @test length(pairs) == 1
    a, b = only(pairs)
    @test a.node === BrokenRules.Amb && b.node === BrokenRules.Amb
    @test a.target === Target{:out}
end

@testitem "diagnostics:rule-not-found" tags = [:base] setup = [BrokenRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, Target, DefaultAlgorithm, RuleNotFoundError
    B = BrokenRules
    text(f) = try
        f()
        ""
    catch err
        err isa RuleNotFoundError ? sprint(showerror, err) : rethrow()
    end

    # The shape fits, a type does not.
    mismatch = text(() -> getresult(message_passing_rule(B.Gauss, Target(:out), DefaultAlgorithm(), RuleArgs(m = (μ = 1.0, τ = "x")))))
    @test contains(mismatch, "type mismatch")
    @test contains(mismatch, "✓ m[:μ]::Real  got Float64")
    @test contains(mismatch, "✗ m[:τ]::Real  got String")
    @test contains(mismatch, "✓ algorithm")

    # Wrong inputs altogether.
    shape = text(() -> getresult(message_passing_rule(B.Gauss, Target(:out), DefaultAlgorithm(), RuleArgs(m = (μ = 1.0,), q = (τ = 1.0,)))))
    @test contains(shape, "no rule of this shape")
    @test contains(shape, "✗ m[:τ]::Real  not provided")
    @test contains(shape, "✗ q[:τ]::Float64  provided but not consumed")

    # Right inputs, wrong algorithm.
    algorithm = text(() -> getresult(message_passing_rule(B.Gauss, Target(:out), B.Standalone(), RuleArgs(m = (μ = 1.0, τ = 2.0)))))
    @test contains(algorithm, "no rule of this shape")
    @test contains(algorithm, "✗ algorithm")

    none = text(() -> getresult(message_passing_rule(B.Gauss, Target(:nothing_here), DefaultAlgorithm(), RuleArgs())))
    @test contains(none, "no rule exists for this node and target")
end
