# Representative rules, written through the definition macros, with toy distributions in place
# of ExponentialFamily.

@testmodule RepresentativeRules begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: DefaultAlgorithm, AbstractAlgorithm, annotate!
    import BayesBase: mean, var

    struct Point
        x::Float64
    end
    mean(p::Point) = p.x
    struct Normal
        μ::Float64
        v::Float64
    end
    mean(n::Normal) = n.μ
    var(n::Normal) = n.v
    struct Categorical
        p::Vector{Float64}
    end

    # A delta node's own algorithm, carrying its known inverse or `nothing`.
    struct ToyDelta{I} <: AbstractAlgorithm
        inverse::I
    end
    # The mixture's own algorithm: its rules ignore the factorisation.
    struct MixtureVMP <: AbstractAlgorithm end

    struct NMV end
    @define_factor_node(node = NMV, type = Stochastic, interfaces = [:out, :μ, :v])

    # 1. The trivial case; `algorithm` omitted, so the node's default applies.
    @define_message_update_rule(
        node = NMV,
        target = :out,
        args = (m[:μ]::Point, m[:v]::Point),
        body = (args) -> Normal(mean(args.m[:μ]), mean(args.m[:v])),
    )

    # 2. Writing a log scale through `ann`.
    @define_message_update_rule(
        node = NMV,
        target = :μ,
        args = (m[:out]::Point, m[:v]::Point),
        body = (args, ann) -> begin
            annotate!(ann, :logscale, 0.0)
            Normal(mean(args.m[:out]), mean(args.m[:v]))
        end,
    )

    # 3. A function node.
    @define_factor_node(node = +, type = Deterministic, interfaces = [:out, :in1, :in2])
    @define_message_update_rule(
        node = +,
        target = :in2,
        algorithm = DefaultAlgorithm,
        args = (m[:out]::Point, m[:in1]::Point),
        body = (args) -> Point(mean(args.m[:out]) - mean(args.m[:in1])),
    )

    # 4. THE CANARY: indexed target, a group, its own algorithm, and the index bound by writing `k`.
    struct NormalMixture end
    @define_factor_node(
        node = NormalMixture, type = Stochastic, interfaces = [:out, :switch, :m..., :p...], algorithm = MixtureVMP,
    )
    @define_message_update_rule(
        node = NormalMixture,
        target = (:m, k),
        args = (q[:out]::Normal, q[:switch]::Categorical, q[:p][k]::Point),
        body = (args) -> Normal(mean(args.q[:out]), mean(args.q[:p][k])),
    )

    # 5. A group consumed whole.
    struct Mixture end
    @define_factor_node(node = Mixture, type = Stochastic, interfaces = [:out, :switch, :inputs...])
    @define_message_update_rule(
        node = Mixture,
        target = :out,
        args = (m[:switch]::Categorical, m[:inputs...]::Normal),
        body = (args) -> sum(map(mean, args.m[:inputs]) .* args.m[:switch].p),
    )

    # 6. The context service that replaces the engine leak.
    @define_message_update_rule(
        node = Mixture,
        target = :switch,
        ctx = (:product,),
        args = (m[:out]::Normal, m[:inputs...]::Normal),
        body = (ctx, args) -> Categorical(collect(map(input -> last(ctx.product(args.m[:out], input)), args.m[:inputs]))),
    )

    # 7 and 8. Same node, target and algorithm type, different parameters: dispatch on the
    # algorithm's type parameter, and the parameter read through `algo`.
    struct DeltaFn end
    @define_factor_node(node = DeltaFn, type = Deterministic, interfaces = [:out, :in...])
    @define_message_update_rule(
        node = DeltaFn,
        target = (:in, k),
        algorithm = ToyDelta{Nothing},
        args = (m[:in][k]::Normal, q[:in...]::Any),
        body = (args) -> Normal(mean(args.m[:in][k]), var(args.m[:in][k]) + k),
    )
    @define_message_update_rule(
        node = DeltaFn,
        target = (:in, k),
        algorithm = ToyDelta{<:Function},
        args = (m[:out]::Point,),
        body = (algo, args) -> Point(algo.inverse(mean(args.m[:out]))),
    )

    # 9. A marginal rule over a structural cluster, and a joint as an input.
    @define_marginal_update_rule(
        node = NMV,
        target = (:out, :μ),
        args = (m[:out]::Point, m[:μ]::Point, q[:v]::Point),
        body = (args) -> (mean(args.m[:out]), mean(args.m[:μ])),
    )
    @define_message_update_rule(
        node = NMV,
        target = :v,
        args = (q[:out, :μ]::Tuple),
        body = (args) -> Point(abs2(args.q[:out, :μ][1] - args.q[:out, :μ][2])),
    )

    # 10. In place: `preallocate` declares the buffer, the typed `output` receives it, and
    # the body may also read the inbound message on its own edge.
    struct Vec end
    @define_factor_node(node = Vec, type = Stochastic, interfaces = [:out, :μ])
    @define_message_update_rule(
        node = Vec,
        target = :out,
        inplace = true,
        args = (m[:μ]::Vector{Float64}, m[:out]::Vector{Float64}),
        preallocate = (args) -> similar(args.m[:μ]),
        body = (output::Vector{Float64}, args) -> (output .= args.m[:μ] .+ args.m[:out]; output),
    )

    # In place towards a group member: `k` is bound in `preallocate` as well.
    struct Stack end
    @define_factor_node(node = Stack, type = Stochastic, interfaces = [:out, :x...])
    @define_message_update_rule(
        node = Stack,
        target = (:x, k),
        inplace = true,
        args = (m[:out]::Vector{Float64},),
        preallocate = (args) -> zeros(k),
        body = (output::Vector{Float64}, args) -> (output .= args.m[:out][1:k]; output),
    )

    # 11. Average energy.
    @define_average_energy(
        node = NMV,
        args = (q[:out]::Normal, q[:μ]::Normal, q[:v]::Point),
        body = (args) -> (var(args.q[:out]) + var(args.q[:μ]) + abs2(mean(args.q[:out]) - mean(args.q[:μ]))) / mean(args.q[:v]),
    )

    # An impure rule under a pure algorithm.
    struct Counter end
    @define_factor_node(node = Counter, type = Stochastic, interfaces = [:out])
    const COUNT = Ref(0)
    @define_message_update_rule(
        node = Counter, target = :out, pure = false, args = (), body = () -> (COUNT[] += 1),
    )
end

@testitem "rules:representative" tags = [:base] setup = [RepresentativeRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleArgs, Marginals, Target, IndexedTarget, ClusterTarget, DefaultAlgorithm,
        RuleContext, RuleAnnotations, AnnotationStore, getannotation
    S = RepresentativeRules
    P, N, C = S.Point, S.Normal, S.Categorical

    @test message_passing_rule(S.NMV, Target(:out), DefaultAlgorithm(), RuleArgs(m = (μ = P(1.0), v = P(2.0)))) == N(1.0, 2.0)

    ann = RuleAnnotations(out = AnnotationStore())
    @test message_passing_rule(S.NMV, Target(:μ), DefaultAlgorithm(), RuleArgs(m = (out = P(3.0), v = P(1.0))), RuleContext(), ann) == N(3.0, 1.0)
    @test getannotation(ann, :logscale) == 0.0

    @test message_passing_rule(+, Target(:in2), DefaultAlgorithm(), RuleArgs(m = (out = P(5.0), in1 = P(2.0)))) == P(3.0)

    # Only member 2 is selected; the rest of the group is `nothing`.
    canary = RuleArgs(q = (out = N(0.5, 1.0), switch = C([0.5, 0.5]), p = (nothing, P(20.0))))
    @test message_passing_rule(S.NormalMixture, IndexedTarget(:m, 2), S.MixtureVMP(), canary) == N(0.5, 20.0)

    mix = RuleArgs(m = (switch = C([0.25, 0.75]), inputs = (N(1.0, 1.0), N(3.0, 1.0))))
    @test message_passing_rule(S.Mixture, Target(:out), DefaultAlgorithm(), mix) == 0.25 * 1.0 + 0.75 * 3.0

    product = (left, right) -> (nothing, -abs2(S.mean(left) - S.mean(right)))
    switch = RuleArgs(m = (out = N(0.0, 1.0), inputs = (N(1.0, 1.0), N(2.0, 1.0))))
    @test message_passing_rule(S.Mixture, Target(:switch), DefaultAlgorithm(), switch, RuleContext(product = product)).p == [-1.0, -4.0]

    lin = RuleArgs(m = (in = (nothing, nothing, N(1.0, 2.0)),), q = (in = (1, 2, 3),))
    @test message_passing_rule(S.DeltaFn, IndexedTarget(:in, 3), S.ToyDelta(nothing), lin) == N(1.0, 5.0)
    inv = RuleArgs(m = (out = P(4.0),))
    @test message_passing_rule(S.DeltaFn, IndexedTarget(:in, 1), S.ToyDelta(sqrt), inv) == P(2.0)

    @test message_passing_marginalrule(S.NMV, ClusterTarget((:out, :μ)), DefaultAlgorithm(), RuleArgs(m = (out = P(1.0), μ = P(2.0)), q = (v = P(1.0),))) == (1.0, 2.0)
    joint = RuleArgs(q = Marginals(NamedTuple(), Val(((:out, :μ),)), ((1.0, 4.0),)))
    @test message_passing_rule(S.NMV, Target(:v), DefaultAlgorithm(), joint) == P(9.0)

    vec_args = RuleArgs(m = (μ = [1.0, 2.0], out = [10.0, 20.0]))
    buffer = zeros(2)
    @test message_passing_rule!(buffer, S.Vec, Target(:out), DefaultAlgorithm(), vec_args) === buffer
    @test buffer == [11.0, 22.0]
    @test message_passing_rule(S.Vec, Target(:out), DefaultAlgorithm(), vec_args) == [11.0, 22.0]

    @test message_passing_rule(S.Stack, IndexedTarget(:x, 2), DefaultAlgorithm(), RuleArgs(m = (out = [7.0, 8.0, 9.0],))) == [7.0, 8.0]

    energy = RuleArgs(q = (out = N(0.0, 1.0), μ = N(1.0, 2.0), v = P(2.0)))
    @test message_passing_average_energy(S.NMV, DefaultAlgorithm(), energy) == (1.0 + 2.0 + 1.0) / 2.0
end

@testitem "rules:specs" tags = [:base] setup = [RepresentativeRules] begin
    using MessagePassingRulesBase: find_message_rule, registered_rules, RuleArgs, Target, DefaultAlgorithm, RuleSpec
    S = RepresentativeRules
    ours = filter(spec -> parentmodule(spec.body) === S || spec.node in (S.NMV, S.NormalMixture, S.Mixture, S.DeltaFn, S.Vec, S.Counter, S.Stack, +), registered_rules())
    @test length(ours) == 14

    spec = find_message_rule(S.NMV, Target(:out), DefaultAlgorithm(), RuleArgs(m = (μ = S.Point(1.0), v = S.Point(2.0))))
    @test spec isa RuleSpec
    @test spec.kind === :message
    @test spec.algorithm === DefaultAlgorithm
    @test endswith(String(spec.file), "rule_macro_tests.jl")
    @test spec.line > 0
    @test contains(spec.source, "Normal(mean(args.m[:μ])")

    switch = only(filter(s -> s.node === S.Mixture && s.target === MessagePassingRulesBase.Target{:switch}, ours))
    @test switch.services === (:product,)

    counter = only(filter(s -> s.node === S.Counter, ours))
    @test !counter.pure
    @test spec.pure
    @test only(filter(s -> s.node === S.Vec, ours)).inplace
end

@testitem "rules:malformed" tags = [:base] begin
    using MessagePassingRulesBase

    function expansion_error(ex)
        err = try
            macroexpand(@__MODULE__, ex)
            nothing
        catch e
            e isa LoadError ? e.error : e
        end
        return err === nothing ? "" : sprint(showerror, err)
    end
    rule(kw...) = expansion_error(Expr(:macrocall, Symbol("@define_message_update_rule"), LineNumberNode(1), kw...))
    kw(k, v) = Expr(:(=), k, v)
    base = (kw(:node, :X), kw(:target, QuoteNode(:out)))

    @test contains(rule(kw(:target, QuoteNode(:out)), kw(:args, :(())), kw(:body, :(() -> 1))), "`node` is required")
    @test contains(rule(base..., kw(:args, :(())), kw(:body, :((args, foo) -> 1))), "unknown body slot `foo`")
    @test contains(rule(base..., kw(:args, :(())), kw(:body, :((args, algo) -> 1))), "canonical order")
    @test contains(rule(base..., kw(:args, :(())), kw(:body, :((output, args) -> 1))), "the `output` slot requires `inplace = true`")
    @test contains(rule(base..., kw(:inplace, true), kw(:args, :(())), kw(:body, :((output, args) -> 1))), "`preallocate` is required")
    @test contains(rule(base..., kw(:inplace, true), kw(:preallocate, :((args) -> 1)), kw(:args, :(())), kw(:body, :((args) -> 1))), "must take `output` first")
    @test contains(rule(base..., kw(:args, :((x[:μ],))), kw(:body, :((args) -> 1))), "`m[...]` or `q[...]`")
    @test contains(rule(base..., kw(:args, :((q[:p[:k]],))), kw(:body, :((args) -> 1))), "indexes a Symbol")
    @test contains(rule(base..., kw(:args, :((m[:a, :b],))), kw(:body, :((args) -> 1))), "only marginals")
    @test contains(rule(base..., kw(:args, :((m[:a], m[:a]))), kw(:body, :((args) -> 1))), "given twice")
    @test contains(rule(base..., kw(:args, :(())), kw(:body, :(1))), "must be a lambda")
    # Any service name is allowed: the context is whatever the caller supplies.
    @test rule(base..., kw(:args, :(())), kw(:body, :(() -> 1)), kw(:ctx, :((:gpu,)))) == ""
    @test contains(rule(kw(:node, :X), kw(:target, :out), kw(:args, :(())), kw(:body, :(() -> 1))), "`target` must be")
end
