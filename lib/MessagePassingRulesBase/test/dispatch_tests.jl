@testmodule HandRules begin
    # Rules written the way the definition macros will lower them, so the dispatch core is
    # tested before any macro exists.
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleSpec, RuleArgs, Messages, Marginals, Target, IndexedTarget,
        ClusterTarget, DefaultAlgorithm, AbstractAlgorithm, target_index
    import MessagePassingRulesBase: find_message_rule, find_marginal_rule, find_average_energy, ispure

    struct Gauss end
    struct Exploding end
    struct Missing end
    struct Mix end

    # A mixture's own algorithm, and one that stands alone with no rules at all.
    struct MixtureVMP <: AbstractAlgorithm end
    struct Standalone <: AbstractAlgorithm end

    struct Stateful <: AbstractAlgorithm end
    ispure(::Type{Stateful}) = false

    # Keys in canonical (sorted) order, as the definition macro will generate them.
    const PointArgs = RuleArgs{<:Messages{(:v, :μ), <:Tuple{Real, Real}}, <:Marginals{(), Tuple{}, (), Tuple{}}}

    const GAUSS_OUT = RuleSpec(
        kind = :message, node = Gauss, target = Target{:out}, algorithm = DefaultAlgorithm, signature = PointArgs,
        body = (output, scratch, algo, ctx, args, ann, target) -> args.m[:μ] + args.m[:v],
    )
    find_message_rule(::Type{Gauss}, ::Target{:out}, ::DefaultAlgorithm, ::PointArgs) = GAUSS_OUT

    const GAUSS_OUT_STATEFUL = RuleSpec(
        kind = :message, node = Gauss, target = Target{:out}, algorithm = Stateful, signature = PointArgs,
        body = (output, scratch, algo, ctx, args, ann, target) -> -1.0,
    )
    find_message_rule(::Type{Gauss}, ::Target{:out}, ::Stateful, ::PointArgs) = GAUSS_OUT_STATEFUL

    const EXPLODES = RuleSpec(
        kind = :message, node = Exploding, target = Target{:out}, algorithm = DefaultAlgorithm, signature = RuleArgs,
        body = (output, scratch, algo, ctx, args, ann, target) -> error("broken, but found"),
    )
    find_message_rule(::Type{Exploding}, ::Target{:out}, ::DefaultAlgorithm, ::RuleArgs) = EXPLODES

    const MIX_M = RuleSpec(
        kind = :message, node = Mix, target = IndexedTarget{:m}, algorithm = MixtureVMP, signature = RuleArgs,
        body = (output, scratch, algo, ctx, args, ann, target) -> args.q[:p][target_index(target)],
    )
    find_message_rule(::Type{Mix}, ::IndexedTarget{:m}, ::MixtureVMP, ::RuleArgs) = MIX_M

    const GAUSS_JOINT = RuleSpec(
        kind = :marginal, node = Gauss, target = ClusterTarget{(:out, :μ)}, algorithm = DefaultAlgorithm, signature = RuleArgs,
        body = (output, scratch, algo, ctx, args, ann, target) -> (args.m[:out], args.m[:μ]),
    )
    find_marginal_rule(::Type{Gauss}, ::ClusterTarget{(:out, :μ)}, ::DefaultAlgorithm, ::RuleArgs) = GAUSS_JOINT

    const GAUSS_ENERGY = RuleSpec(
        kind = :average_energy, node = Gauss, target = Nothing, algorithm = DefaultAlgorithm, signature = RuleArgs,
        body = (output, scratch, algo, ctx, args, ann, target) -> 42.0,
    )
    find_average_energy(::Type{Gauss}, ::DefaultAlgorithm, ::RuleArgs) = GAUSS_ENERGY

    const INPLACE = RuleSpec(
        kind = :message, node = Gauss, target = Target{:v}, algorithm = DefaultAlgorithm, signature = RuleArgs,
        inplace = true,
        prealloc = (algo, ctx, args, target) -> similar(args.m[:x]),
        body = (output, scratch, algo, ctx, args, ann, target) -> (output .= 2 .* args.m[:x]; output),
    )
    find_message_rule(::Type{Gauss}, ::Target{:v}, ::DefaultAlgorithm, ::RuleArgs) = INPLACE

    const ANNOTATES = RuleSpec(
        kind = :message, node = Gauss, target = Target{:μ}, algorithm = DefaultAlgorithm, signature = RuleArgs,
        body = (output, scratch, algo, ctx, args, ann, target) -> begin
            MessagePassingRulesBase.annotate!(ann, :note, 1.5)
            ctx.node
        end,
    )
    find_message_rule(::Type{Gauss}, ::Target{:μ}, ::DefaultAlgorithm, ::RuleArgs) = ANNOTATES

    const POINT = RuleArgs(m = (μ = 1.0, v = 2.0))

    measure(args) = (getresult(message_passing_rule(Gauss, Target(:out), DefaultAlgorithm(), args)); @allocated getresult(message_passing_rule(Gauss, Target(:out), DefaultAlgorithm(), args)))
end

@testitem "dispatch:resolution" tags = [:base] setup = [HandRules] begin
    using MessagePassingRulesBase: find_message_rule, RuleSpec, RuleNotFound, Target, DefaultAlgorithm, RuleArgs
    H = HandRules

    @test find_message_rule(H.Gauss, Target(:out), DefaultAlgorithm(), H.POINT) === H.GAUSS_OUT
    # The algorithm is part of the signature: same node and edge, different rule.
    @test find_message_rule(H.Gauss, Target(:out), H.Stateful(), H.POINT) === H.GAUSS_OUT_STATEFUL

    # Resolution is total: a missing rule is a value, never an exception.
    nf = find_message_rule(H.Missing, Target(:out), DefaultAlgorithm(), H.POINT)
    @test nf isa RuleNotFound
    @test nf.node === H.Missing
    # An algorithm that stands alone inherits nothing from the default.
    nf2 = find_message_rule(H.Gauss, Target(:out), H.Standalone(), H.POINT)
    @test nf2 isa RuleNotFound

    # The resolved type is the one concrete `RuleSpec` at a call site reaching one rule.
    @test (@inferred find_message_rule(H.Gauss, Target(:out), DefaultAlgorithm(), H.POINT)) isa RuleSpec
end

@testitem "dispatch:execution" tags = [:base] setup = [HandRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: Target, IndexedTarget, ClusterTarget, DefaultAlgorithm, RuleArgs, Marginals,
        RuleContext, RuleAnnotations, AnnotationStore, getannotation, RuleNotFoundError
    H = HandRules

    @test getresult(message_passing_rule(H.Gauss, Target(:out), DefaultAlgorithm(), H.POINT)) == 3.0

    mix_args = RuleArgs(q = (p = (10.0, 20.0, 30.0),))
    @test getresult(message_passing_rule(H.Mix, IndexedTarget(:m, 2), H.MixtureVMP(), mix_args)) == 20.0

    joint_args = RuleArgs(m = (out = 1.0, μ = 2.0))
    @test getresult(message_passing_marginalrule(H.Gauss, ClusterTarget((:out, :μ)), DefaultAlgorithm(), joint_args)) == (1.0, 2.0)
    @test getresult(message_passing_average_energy(H.Gauss, DefaultAlgorithm(), RuleArgs())) == 42.0

    # `ctx` carries the node; `ann` is where the rule writes.
    ann = RuleAnnotations(out = AnnotationStore())
    node = :some_node_instance
    @test getresult(message_passing_rule(H.Gauss, Target(:μ), DefaultAlgorithm(), RuleArgs(), RuleContext(node = node), ann)) === node
    @test getannotation(ann, :note) == 1.5

    # The same not-found path for all three.
    @test_throws RuleNotFoundError getresult(message_passing_rule(H.Missing, Target(:out), DefaultAlgorithm(), H.POINT))
    @test_throws RuleNotFoundError getresult(message_passing_marginalrule(H.Missing, ClusterTarget((:a, :b)), DefaultAlgorithm(), RuleArgs()))
    @test_throws RuleNotFoundError getresult(message_passing_average_energy(H.Missing, DefaultAlgorithm(), RuleArgs()))
end

@testitem "dispatch:fallback-contract" tags = [:base] setup = [HandRules] begin
    # A missing rule and a broken rule are distinguishable, and a fallback runs on the
    # first only. The engine's fallback sits on the `RuleNotFound` branch, decided before
    # any body runs, so no exception from a selected rule can reach it.
    using MessagePassingRulesBase: find_message_rule, execute_rule, RuleNotFound, Target, DefaultAlgorithm, RuleContext, NoAnnotations
    H = HandRules

    calls = Ref(0)
    function with_fallback(node, args)
        spec = find_message_rule(node, Target(:out), DefaultAlgorithm(), args)
        if spec isa RuleNotFound
            calls[] += 1
            return :fallback
        end
        return execute_rule(spec, nothing, DefaultAlgorithm(), RuleContext(), args, NoAnnotations(), Target(:out))
    end

    @test with_fallback(H.Missing, H.POINT) === :fallback
    @test calls[] == 1
    @test_throws ErrorException("broken, but found") with_fallback(H.Exploding, H.POINT)
    @test calls[] == 1
end

@testitem "dispatch:inplace" tags = [:base] setup = [HandRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: Target, DefaultAlgorithm, RuleArgs
    H = HandRules

    args = RuleArgs(m = (x = [1.0, 2.0],))
    allocated = getresult(message_passing_rule(H.Gauss, Target(:v), DefaultAlgorithm(), args))
    buffer = zeros(2)
    inplace = getresult(message_passing_rule!(buffer, H.Gauss, Target(:v), DefaultAlgorithm(), args))
    @test inplace === buffer
    @test allocated == inplace == [2.0, 4.0]

    # `rule!` on a rule with no in-place form is an error, not a silent allocation.
    @test_throws ArgumentError getresult(message_passing_rule!(zeros(1), H.Gauss, Target(:out), DefaultAlgorithm(), H.POINT))
end

@testitem "dispatch:purity" tags = [:base] setup = [HandRules] begin
    using MessagePassingRulesBase: ispure, DefaultAlgorithm, RuleSpec, Target, RuleArgs
    H = HandRules

    @test ispure(DefaultAlgorithm) && ispure(H.MixtureVMP)
    @test !ispure(H.Stateful)
    # A rule inherits its algorithm's purity unless it overrides it.
    @test H.GAUSS_OUT.pure
    @test !H.GAUSS_OUT_STATEFUL.pure
    body = (o, s, a, c, r, n, t) -> nothing
    @test !RuleSpec(kind = :message, node = H.Gauss, target = Target{:out}, algorithm = DefaultAlgorithm, signature = RuleArgs, body = body, pure = false).pure
end

@testitem "gate:routing" tags = [:base, :alloc] setup = [HandRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: Target, DefaultAlgorithm
    H = HandRules
    # Resolve and execute at a call site that can reach exactly one rule.
    @test (@inferred getresult(message_passing_rule(H.Gauss, Target(:out), DefaultAlgorithm(), H.POINT))) === 3.0
    @test H.measure(H.POINT) == 0
end

@testitem "context:matrix-correction" tags = [:base] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleContext, RuleArgs, Target, find_message_rule, missing_services, DEFAULT_CONTEXT_SERVICES

    # A strategy for correcting a matrix, such as MatrixCorrectionTools' ones, reached as
    # `ctx.matrix_correction`. `nothing` means not set, and the rule applies its own default
    # (here doubling), so it is never a missing service.
    struct Corrected end
    @define_factor_node(node = Corrected, type = Stochastic, interfaces = [:out, :in])
    @define_message_update_rule(
        node = Corrected, target = :out, ctx = (:matrix_correction,),
        args = (m[:in]::Float64,),
        body = (ctx, args) -> matrix_correction(ctx, x -> 2x)(args.m[:in]),
    )

    @test :matrix_correction in DEFAULT_CONTEXT_SERVICES
    @test RuleContext().matrix_correction === nothing
    args = RuleArgs(m = (in = 2.0,))
    @test getresult(message_passing_rule(Corrected, Target(:out), DefaultAlgorithm(), args)) == 4.0
    @test getresult(message_passing_rule(Corrected, Target(:out), DefaultAlgorithm(), args, RuleContext(matrix_correction = x -> 10x))) == 20.0
    @test matrix_correction(RuleContext(), nothing) === nothing
    @test matrix_correction(RuleContext(matrix_correction = :set), :default) === :set

    spec = find_message_rule(Corrected, Target(:out), DefaultAlgorithm(), args)
    @test spec.services === (:matrix_correction,)
    # Supplied as `nothing`, it is set to "the rule's own default"; left out, it is missing.
    @test isempty(missing_services(spec, RuleContext(matrix_correction = nothing)))
    @test missing_services(spec, RuleContext()) == (:matrix_correction,)
end

@testitem "context:any service" tags = [:base] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleContext, RuleArgs, Target, find_message_rule, missing_services

    # The context holds whatever the caller supplies, read as `ctx.name`, and a rule may declare
    # a service of its own; one not supplied reads as `nothing`.
    struct Scaled end
    @define_factor_node(node = Scaled, type = Stochastic, interfaces = [:out, :in])
    @define_message_update_rule(node = Scaled, target = :out, ctx = (:scale,), args = (m[:in]::Float64,), body = (ctx, args) -> ctx.scale * args.m[:in])

    ctx = RuleContext(scale = 3.0)
    @test ctx.scale === 3.0 && ctx.rng === nothing
    @test ctx isa RuleContext{@NamedTuple{scale::Float64}} && ismutable(ctx)
    args = RuleArgs(m = (in = 2.0,))
    @test getresult(message_passing_rule(Scaled, Target(:out), DefaultAlgorithm(), args, ctx)) == 6.0
    spec = find_message_rule(Scaled, Target(:out), DefaultAlgorithm(), args)
    @test missing_services(spec, ctx) == () && missing_services(spec, RuleContext()) == (:scale,)
    # What an engine checks when it resolves a rule: an error naming the rule and the service,
    # and nothing to allocate when every service is supplied.
    @test MessagePassingRulesBase.check_services(spec, ctx) === nothing
    checks(spec, ctx, n) = (
        for _ in 1:n
            MessagePassingRulesBase.check_services(spec, ctx)
        end; nothing
    )
    allocations(spec, ctx) = @allocated checks(spec, ctx, 100)
    allocations(spec, ctx)
    @test allocations(spec, ctx) == 0
    @test_throws ArgumentError MessagePassingRulesBase.check_services(spec, RuleContext())
    report = try
        MessagePassingRulesBase.check_services(spec, RuleContext(rng = nothing))
    catch error
        sprint(showerror, error)
    end
    @test contains(report, "Scaled") && contains(report, ":scale") && contains(report, "context")
    # An entry that is `nothing` counts as supplied.
    @test MessagePassingRulesBase.check_services(spec, RuleContext(scale = nothing)) === nothing
    # The contract of the calls by hand: they run with the context given and check nothing, so a
    # declared service it lacks reads as `nothing` inside the rule.
    @define_message_update_rule(node = Scaled, target = :in, ctx = (:scale,), args = (m[:out]::Float64,), body = (ctx, args) -> ctx.scale)
    @test getresult(call_message_update_rule(Scaled, :in; m = (out = 1.0,))) === nothing
    @test getresult(@call_message_update_rule(node = Scaled, target = :in, m = (out = 1.0,))) === nothing
    @test getresult(message_passing_rule(Scaled, Target(:in), DefaultAlgorithm(), RuleArgs(m = (out = 1.0,)))) === nothing
    # Merging overrides and adds, as an engine layers a model's services over its defaults.
    merged = merge(RuleContext(rng = :default, scale = 1.0), (scale = 2.0, extra = 1))
    @test merged.rng === :default && merged.scale === 2.0 && merged.extra === 1
    @test propertynames(merged) == (:rng, :scale, :extra)
    @test (@inferred (c -> c.scale)(ctx)) === 3.0
end
