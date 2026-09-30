@testitem "interactive:call_message_update_rule" tags = [:base] setup = [RepresentativeRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: AnnotationStore, getannotation, RuleContext, DefaultAlgorithm, getalgorithm, getcontext, getscratch, getarguments, gettarget
    S = RepresentativeRules
    P, N = S.Point, S.Normal

    @test (getresult(@call_message_update_rule(node = S.NMV, target = :out, m = (μ = P(1.0), v = P(2.0))))) == N(1.0, 2.0)
    # Function form, identical.
    @test getresult(call_message_update_rule(S.NMV, :out; m = (μ = P(1.0), v = P(2.0)))) == N(1.0, 2.0)
    # Indexed target and the node's default algorithm (its own, `MixtureVMP`, here).
    @test (getresult(@call_message_update_rule(node = S.NormalMixture, target = (:m, 2), q = (out = N(0.5, 1.0), switch = S.Categorical([0.5, 0.5]), p = (nothing, P(20.0)))))) == N(0.5, 20.0)
    # A cluster given by its members.
    @test (getresult(@call_message_update_rule(node = S.NMV, target = :v, clusters = ((:out, :μ) => (1.0, 4.0),)))) == P(9.0)
    # The result carries the rule's log scale and what produced it.
    result = @call_message_update_rule(node = S.NMV, target = :μ, m = (out = P(3.0), v = P(1.0)))
    @test getresult(result) == N(3.0, 1.0)
    @test getlogscale(result) === 0.0
    @test getrule(result) === which_message_update_rule(S.NMV, :μ; m = (out = P(3.0), v = P(1.0)))
    @test getalgorithm(result) === DefaultAlgorithm() && gettarget(result) === MessagePassingRulesBase.Target(:μ)
    @test getarguments(result).m[:out] == P(3.0) && getarguments(result).logscale === nothing
    @test getcontext(result) isa RuleContext && getscratch(result) === nothing
    @test getannotations(result) === MessagePassingRulesBase.NoAnnotations()
    # Incoming log scales are given like the messages.
    switch = call_message_update_rule(S.Mixture, :switch; m = (out = N(0.0, 1.0), inputs = (N(1.0, 1.0), N(2.0, 1.0))), logscale = (out = 0.0, inputs = (-1.0, -4.0)))
    @test getresult(switch).p == [-1.0, -4.0] && getlogscale(switch) == -5.0
    # Annotations are collected when asked for.
    ann = AnnotationStore()
    @test getannotations(call_message_update_rule(S.NMV, :μ; m = (out = P(3.0), v = P(1.0)), ann = ann)) === ann
    # A marginal and an energy have no log scale.
    @test getlogscale(call_average_energy(S.NMV; q = (out = N(0.0, 1.0), μ = N(1.0, 2.0), v = P(2.0)))) === nothing

    @test (getresult(@call_marginal_update_rule(node = S.NMV, target = (:out, :μ), m = (out = P(1.0), μ = P(2.0)), q = (v = P(1.0),)))) == (1.0, 2.0)
    @test (getresult(@call_average_energy(node = S.NMV, q = (out = N(0.0, 1.0), μ = N(1.0, 2.0), v = P(2.0))))) == 2.0
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

@testitem "interactive:a function node named by its type" tags = [:base] begin
    using MessagePassingRulesBase

    double(x) = 2x
    @define_factor_node(node = double, type = Deterministic, interfaces = [:out, :in])
    @define_message_update_rule(node = double, target = :out, args = (m[:in]::Real,), logscale = 0, body = (args) -> 2 * args.m[:in])
    @define_marginal_update_rule(node = double, target = (:out, :in), args = (m[:out]::Real, m[:in]::Real), body = (args) -> (args.m[:out], args.m[:in]))
    @define_average_energy(node = double, args = (q[:out]::Real, q[:in]::Real), body = (args) -> 0.0)

    @test getresult(@call_message_update_rule(node = double, target = :out, m = (in = 1.0,))) == 2.0

    # The type of the function in place of the function says what to write instead, whether the
    # algorithm is the default or given.
    message = "pass the function itself, `node = double`"
    for call in (
            () -> call_message_update_rule(typeof(double), :out; m = (in = 1.0,)),
            () -> call_message_update_rule(typeof(double), :out; m = (in = 1.0,), algorithm = DefaultAlgorithm()),
            () -> @call_message_update_rule(node = typeof(double), target = :out, m = (in = 1.0,)),
            () -> call_marginal_update_rule(typeof(double), (:out, :in); m = (out = 2.0, in = 1.0)),
            () -> call_average_energy(typeof(double); q = (out = 2.0, in = 1.0)),
            () -> which_message_update_rule(typeof(double), :out; m = (in = 1.0,)),
            () -> which_marginal_update_rule(typeof(double), (:out, :in); m = (out = 2.0, in = 1.0)),
            () -> which_average_energy(typeof(double); q = (out = 2.0, in = 1.0)),
        )
        err = try
            call()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test contains(sprint(showerror, err), message)
    end
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

@testitem "interactive:coverage of wildcard and parametric rules" tags = [:base] setup = [DefaultArgsNodes] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: rule_coverage, registries

    # A marginal rule over any cluster has a row of its own.
    coverage = rule_coverage(DefaultArgsNodes.Tensor)
    @test "q(any cluster)" in coverage.rows
    @test contains(sprint(show, MIME"text/plain"(), coverage), "q(any cluster)")
    @test contains(sprint(show, only(filter(s -> s.kind === :marginal && s.algorithm === DefaultAlgorithm, MessagePassingRulesBase.list_rules(DefaultArgsNodes.Tensor)))), "towards any cluster")

    # A parametric algorithm: rules on the type itself and on one of its variants are two columns,
    # labelled apart, and the node's own instance adds no empty column of its own.
    struct Variant{T} <: AbstractAlgorithm end
    Variant() = Variant{Nothing}()
    struct Parametric end
    @define_factor_node(node = Parametric, type = Stochastic, interfaces = [:out, :in], algorithm = Variant)
    @define_message_update_rule(node = Parametric, target = :out, algorithm = Variant, args = (m[:in]::Float64,), body = (args) -> args.m[:in])
    @define_message_update_rule(node = Parametric, target = :in, algorithm = Variant{Int}, args = (m[:out]::Float64,), body = (args) -> args.m[:out])
    coverage = rule_coverage(Parametric)
    @test Set(coverage.algorithms) == Set([Variant, Variant{Int}])
    html = sprint(show, MIME"text/html"(), coverage)
    @test contains(html, "<th>Variant</th>") && contains(html, "<th>Variant{Int64}</th>")

    # Shown from the module that defines them, names are not qualified.
    @test !contains(sprint(show, MIME"text/html"(), coverage; context = :module => @__MODULE__), string(nameof(@__MODULE__)))
    spec = only(MessagePassingRulesBase.list_rules(Parametric, :out))
    @test !contains(sprint(show, MIME"text/plain"(), spec; context = :module => @__MODULE__), string(nameof(@__MODULE__)) * ".")

    # Collecting the registries reads no deprecated binding, which would warn.
    Base.@deprecate_binding OldName Parametric false
    @test_logs min_level = Base.CoreLogging.Warn registries()
end

@testitem "interactive:tooling" tags = [:base] setup = [RepresentativeRules] begin
    using MessagePassingRulesBase
    import MessagePassingRulesBase as B
    S = RepresentativeRules
    P = S.Point

    @test B.as_target(:out) === B.Target(:out) && B.as_target((:m, 2)) === B.IndexedTarget(:m, 2)
    @test B.as_target(B.Target(:out)) === B.Target(:out)
    @test B.as_cluster((:out, :μ)) === B.ClusterTarget((:out, :μ)) && B.as_cluster(B.ClusterTarget((:out, :μ))) === B.ClusterTarget((:out, :μ))
    args = B.interactive_args((μ = P(1.0), v = P(2.0)), (;), ())
    @test args isa B.RuleArgs && args.m[:μ] == P(1.0) && args.logscale === nothing

    # An observer sees every rule an interactive call selects; registering it twice counts once.
    seen = []
    observer = spec -> push!(seen, spec)
    B.add_selection_observer!(observer)
    B.add_selection_observer!(observer)
    call_message_update_rule(S.NMV, :out; m = (μ = P(1.0), v = P(2.0)))
    @test length(seen) == 1 && only(seen) === which_message_update_rule(S.NMV, :out; m = (μ = P(1.0), v = P(2.0)))
    filter!(!=(observer), B.INTERACTIVE_SELECTION_OBSERVERS)
end

@testitem "interactive:coverage labels a bounded variant with its bounds" tags = [:base] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: rule_coverage, AbstractAlgorithm

    struct Bounded{M} <: AbstractAlgorithm end
    struct FirstMethod end
    struct SecondMethod end
    struct Variants end
    @define_factor_node(node = Variants, type = Stochastic, interfaces = [:out, :in])
    @define_message_update_rule(node = Variants, target = :out, algorithm = Bounded{<:FirstMethod}, args = (m[:in]::Float64,), body = (args) -> args.m[:in])
    @define_message_update_rule(node = Variants, target = :out, algorithm = Bounded{<:SecondMethod}, args = (m[:in]::Float64,), body = (args) -> args.m[:in])
    text = sprint(show, MIME"text/plain"(), rule_coverage(Variants))
    @test contains(text, "Bounded{<:FirstMethod}") && contains(text, "Bounded{<:SecondMethod}")
end
