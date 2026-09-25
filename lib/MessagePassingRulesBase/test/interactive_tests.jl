@testitem "interactive:call_message_update_rule" tags = [:base] setup = [RepresentativeRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: AnnotationStore, getannotation, RuleContext
    S = RepresentativeRules
    P, N = S.Point, S.Normal

    @test (@call_message_update_rule(node = S.NMV, target = :out, m = (μ = P(1.0), v = P(2.0)))) == N(1.0, 2.0)
    # Function form, identical.
    @test call_message_update_rule(S.NMV, :out; m = (μ = P(1.0), v = P(2.0))) == N(1.0, 2.0)
    # Indexed target and the node's default algorithm (its own, `MixtureVMP`, here).
    @test (@call_message_update_rule(node = S.NormalMixture, target = (:m, 2), q = (out = N(0.5, 1.0), switch = S.Categorical([0.5, 0.5]), p = (nothing, P(20.0))))) == N(0.5, 20.0)
    # A cluster given by its members.
    @test (@call_message_update_rule(node = S.NMV, target = :v, clusters = ((:out, :μ) => (1.0, 4.0),))) == P(9.0)
    # Annotations are collected when asked for.
    ann = AnnotationStore()
    @call_message_update_rule(node = S.NMV, target = :μ, m = (out = P(3.0), v = P(1.0)), ann = ann)
    @test getannotation(ann, :logscale) == 0.0

    @test (@call_marginal_update_rule(node = S.NMV, target = (:out, :μ), m = (out = P(1.0), μ = P(2.0)), q = (v = P(1.0),))) == (1.0, 2.0)
    @test (@call_average_energy(node = S.NMV, q = (out = N(0.0, 1.0), μ = N(1.0, 2.0), v = P(2.0)))) == 2.0
end

@testitem "interactive:queries" tags = [:base] setup = [RepresentativeRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: list_rules, rule_coverage, DefaultAlgorithm, RuleSpec
    S = RepresentativeRules

    @test length(list_rules(S.NMV)) == 5
    @test length(list_rules(S.NMV, :out)) == 1
    @test length(list_rules(S.DeltaFn, :in)) == 2
    @test length(list_rules(S.DeltaFn, :in; algorithm = S.ToyDelta(nothing))) == 1
    # A standalone algorithm with no rules for this node selects none.
    @test isempty(list_rules(S.NMV, :out; algorithm = S.MixtureVMP()))

    spec = @which_message_update_rule(node = S.NMV, target = :out, m = (μ = S.Point(1.0), v = S.Point(2.0)))
    @test spec isa RuleSpec
    @test spec === only(list_rules(S.NMV, :out))
    @test which_message_update_rule(S.NMV, :out; m = (μ = S.Point(1.0), v = S.Point(2.0))) === spec
    marginal = @which_marginal_update_rule(node = S.NMV, target = (:out, :μ), m = (out = S.Point(1.0), μ = S.Point(2.0)), q = (v = S.Point(1.0),))
    @test marginal.kind === :marginal
    energy = @which_average_energy(node = S.NMV, q = (out = S.Normal(0.0, 1.0), μ = S.Normal(1.0, 2.0), v = S.Point(2.0)))
    @test energy.kind === :average_energy

    coverage = rule_coverage(S.NMV)
    @test coverage.algorithms == [DefaultAlgorithm]
    @test coverage.counts[("→ out", DefaultAlgorithm)] == 1
    @test coverage.counts[("q(out, μ)", DefaultAlgorithm)] == 1
    @test coverage.counts[("average energy", DefaultAlgorithm)] == 1
    @test !haskey(coverage.counts, ("→ v", S.MixtureVMP))
end

@testitem "interactive:display" tags = [:base] setup = [RepresentativeRules, DependencyNodes] begin
    using MessagePassingRulesBase: list_rules, nodespec, dependencies_spec, rule_coverage, Target, IndexedTarget, ClusterTarget
    S = RepresentativeRules
    plain(x) = sprint(show, MIME"text/plain"(), x)
    html(x) = sprint(show, MIME"text/html"(), x)

    @test sprint(show, Target(:out)) == ":out"
    @test sprint(show, IndexedTarget(:m, 2)) == "(:m, 2)"
    @test sprint(show, ClusterTarget((:y, :x))) == "(:y, :x)"

    rule = only(list_rules(S.NMV, :out))
    text = plain(rule)
    @test contains(text, "message rule")
    @test contains(text, "towards :out")
    @test contains(text, "m[:μ]::")
    @test contains(text, "Normal(mean(args.m[:μ])")
    @test contains(text, "rule_macro_tests.jl")
    @test !contains(sprint(show, rule), "\n")                  # compact form is one line
    @test contains(html(rule), "<table")

    node = plain(nodespec(S.NormalMixture))
    @test contains(node, "stochastic")
    @test contains(node, "m...")
    @test contains(node, "MixtureVMP")
    @test contains(html(nodespec(S.NormalMixture)), "<table")

    deps = plain(dependencies_spec(DependencyNodes.NormalMixture, DependencyNodes.MixtureVMP()))
    @test contains(deps, "(:m, k)")
    @test contains(deps, "q[:p][k]")
    @test contains(html(dependencies_spec(DependencyNodes.NormalMixture, DependencyNodes.MixtureVMP())), "<table")

    coverage = plain(rule_coverage(S.NMV))
    @test contains(coverage, "→ out")
    @test contains(coverage, "✓")
    @test contains(html(rule_coverage(S.NMV)), "<table")
end

@testitem "interactive:visualize" tags = [:base] setup = [RepresentativeRules] begin
    using MessagePassingRulesBase: visualize_spec, nodespec
    err = try
        visualize_spec(nodespec(RepresentativeRules.NMV))
        nothing
    catch e
        e
    end
    @test err isa MethodError
    @test contains(sprint(showerror, err), "visualisation backend")
end
