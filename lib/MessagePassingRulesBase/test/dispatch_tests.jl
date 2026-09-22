@testmodule HandRules begin
    # Rules written the way the definition macros will lower them, so the dispatch core is
    # tested before any macro exists.
    using MessagePassingRulesBase
    using MessagePassingRulesBase: RuleSpec, RuleArgs, Messages, Marginals, Target, IndexedTarget,
        ClusterTarget, BP, VMP, AbstractAlgorithm, target_index
    import MessagePassingRulesBase: find_message_rule, find_marginal_rule, find_average_energy, ispure

    struct Gauss end
    struct Exploding end
    struct Missing end
    struct Mix end

    struct Stateful <: AbstractAlgorithm end
    ispure(::Type{Stateful}) = false

    # Keys in canonical (sorted) order, as the definition macro will generate them.
    const PointArgs = RuleArgs{<:Messages{(:v, :μ), <:Tuple{Real, Real}}, <:Marginals{(), Tuple{}, (), Tuple{}}}

    const GAUSS_OUT = RuleSpec(
        kind = :message, node = Gauss, target = Target{:out}, algorithm = BP, signature = PointArgs,
        body = (output, algo, ctx, args, ann, target) -> args.m[:μ] + args.m[:v],
    )
    find_message_rule(::Type{Gauss}, ::Target{:out}, ::BP, ::PointArgs) = GAUSS_OUT

    const GAUSS_OUT_STATEFUL = RuleSpec(
        kind = :message, node = Gauss, target = Target{:out}, algorithm = Stateful, signature = PointArgs,
        body = (output, algo, ctx, args, ann, target) -> -1.0,
    )
    find_message_rule(::Type{Gauss}, ::Target{:out}, ::Stateful, ::PointArgs) = GAUSS_OUT_STATEFUL

    const EXPLODES = RuleSpec(
        kind = :message, node = Exploding, target = Target{:out}, algorithm = BP, signature = RuleArgs,
        body = (output, algo, ctx, args, ann, target) -> error("broken, but found"),
    )
    find_message_rule(::Type{Exploding}, ::Target{:out}, ::BP, ::RuleArgs) = EXPLODES

    const MIX_M = RuleSpec(
        kind = :message, node = Mix, target = IndexedTarget{:m}, algorithm = VMP, signature = RuleArgs,
        body = (output, algo, ctx, args, ann, target) -> args.q[:p][target_index(target)],
    )
    find_message_rule(::Type{Mix}, ::IndexedTarget{:m}, ::VMP, ::RuleArgs) = MIX_M

    const GAUSS_JOINT = RuleSpec(
        kind = :marginal, node = Gauss, target = ClusterTarget{(:out, :μ)}, algorithm = BP, signature = RuleArgs,
        body = (output, algo, ctx, args, ann, target) -> (args.m[:out], args.m[:μ]),
    )
    find_marginal_rule(::Type{Gauss}, ::ClusterTarget{(:out, :μ)}, ::BP, ::RuleArgs) = GAUSS_JOINT

    const GAUSS_ENERGY = RuleSpec(
        kind = :average_energy, node = Gauss, target = Nothing, algorithm = BP, signature = RuleArgs,
        body = (output, algo, ctx, args, ann, target) -> 42.0,
    )
    find_average_energy(::Type{Gauss}, ::BP, ::RuleArgs) = GAUSS_ENERGY

    const INPLACE = RuleSpec(
        kind = :message, node = Gauss, target = Target{:v}, algorithm = BP, signature = RuleArgs,
        inplace = true,
        prealloc = (algo, ctx, args, target) -> similar(args.m[:x]),
        body = (output, algo, ctx, args, ann, target) -> (output .= 2 .* args.m[:x]; output),
    )
    find_message_rule(::Type{Gauss}, ::Target{:v}, ::BP, ::RuleArgs) = INPLACE

    const ANNOTATES = RuleSpec(
        kind = :message, node = Gauss, target = Target{:μ}, algorithm = BP, signature = RuleArgs,
        body = (output, algo, ctx, args, ann, target) -> begin
            MessagePassingRulesBase.annotate!(ann, :logscale, 1.5)
            ctx.node
        end,
    )
    find_message_rule(::Type{Gauss}, ::Target{:μ}, ::BP, ::RuleArgs) = ANNOTATES

    const POINT = RuleArgs(m = (μ = 1.0, v = 2.0))

    measure(args) = (message_passing_rule(Gauss, Target(:out), BP(), args); @allocated message_passing_rule(Gauss, Target(:out), BP(), args))
end

@testitem "dispatch:resolution" tags = [:base] setup = [HandRules] begin
    using MessagePassingRulesBase: find_message_rule, RuleSpec, RuleNotFound, Target, BP, VMP, RuleArgs
    H = HandRules

    @test find_message_rule(H.Gauss, Target(:out), BP(), H.POINT) === H.GAUSS_OUT
    # The algorithm is part of the signature: same node and edge, different rule.
    @test find_message_rule(H.Gauss, Target(:out), H.Stateful(), H.POINT) === H.GAUSS_OUT_STATEFUL

    # Resolution is total: a missing rule is a value, never an exception.
    nf = find_message_rule(H.Missing, Target(:out), BP(), H.POINT)
    @test nf isa RuleNotFound
    @test nf.node === H.Missing
    nf2 = find_message_rule(H.Gauss, Target(:out), VMP(), H.POINT)
    @test nf2 isa RuleNotFound

    # The resolved type is the one concrete `RuleSpec` at a call site reaching one rule.
    @test (@inferred find_message_rule(H.Gauss, Target(:out), BP(), H.POINT)) isa RuleSpec
end

@testitem "dispatch:execution" tags = [:base] setup = [HandRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: Target, IndexedTarget, ClusterTarget, BP, VMP, RuleArgs, Marginals,
        RuleContext, RuleAnnotations, AnnotationStore, getannotation, RuleNotFoundError
    H = HandRules

    @test message_passing_rule(H.Gauss, Target(:out), BP(), H.POINT) == 3.0

    mix_args = RuleArgs(q = (p = (10.0, 20.0, 30.0),))
    @test message_passing_rule(H.Mix, IndexedTarget(:m, 2), VMP(), mix_args) == 20.0

    joint_args = RuleArgs(m = (out = 1.0, μ = 2.0))
    @test message_passing_marginalrule(H.Gauss, ClusterTarget((:out, :μ)), BP(), joint_args) == (1.0, 2.0)
    @test message_passing_average_energy(H.Gauss, BP(), RuleArgs()) == 42.0

    # `ctx` carries the node; `ann` is where the rule writes.
    ann = RuleAnnotations(out = AnnotationStore())
    node = :some_node_instance
    @test message_passing_rule(H.Gauss, Target(:μ), BP(), RuleArgs(), RuleContext(node = node), ann) === node
    @test getannotation(ann, :logscale) == 1.5

    # The same not-found path for all three.
    @test_throws RuleNotFoundError message_passing_rule(H.Missing, Target(:out), BP(), H.POINT)
    @test_throws RuleNotFoundError message_passing_marginalrule(H.Missing, ClusterTarget((:a, :b)), BP(), RuleArgs())
    @test_throws RuleNotFoundError message_passing_average_energy(H.Missing, BP(), RuleArgs())
end

@testitem "dispatch:fallback-contract" tags = [:base] setup = [HandRules] begin
    # A missing rule and a broken rule are distinguishable, and a fallback runs on the
    # first only. The engine's fallback sits on the `RuleNotFound` branch, decided before
    # any body runs, so no exception from a selected rule can reach it.
    using MessagePassingRulesBase: find_message_rule, execute_rule, RuleNotFound, Target, BP, RuleContext, NoAnnotations
    H = HandRules

    calls = Ref(0)
    function with_fallback(node, args)
        spec = find_message_rule(node, Target(:out), BP(), args)
        if spec isa RuleNotFound
            calls[] += 1
            return :fallback
        end
        return execute_rule(spec, nothing, BP(), RuleContext(), args, NoAnnotations(), Target(:out))
    end

    @test with_fallback(H.Missing, H.POINT) === :fallback
    @test calls[] == 1
    @test_throws ErrorException("broken, but found") with_fallback(H.Exploding, H.POINT)
    @test calls[] == 1
end

@testitem "dispatch:inplace" tags = [:base] setup = [HandRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: Target, BP, RuleArgs
    H = HandRules

    args = RuleArgs(m = (x = [1.0, 2.0],))
    allocated = message_passing_rule(H.Gauss, Target(:v), BP(), args)
    buffer = zeros(2)
    inplace = message_passing_rule!(buffer, H.Gauss, Target(:v), BP(), args)
    @test inplace === buffer
    @test allocated == inplace == [2.0, 4.0]

    # `rule!` on a rule with no in-place form is an error, not a silent allocation.
    @test_throws ArgumentError message_passing_rule!(zeros(1), H.Gauss, Target(:out), BP(), H.POINT)
end

@testitem "dispatch:purity" tags = [:base] setup = [HandRules] begin
    using MessagePassingRulesBase: ispure, BP, VMP, RuleSpec, Target, RuleArgs
    H = HandRules

    @test ispure(BP) && ispure(VMP)
    @test !ispure(H.Stateful)
    # A rule inherits its algorithm's purity unless it overrides it.
    @test H.GAUSS_OUT.pure
    @test !H.GAUSS_OUT_STATEFUL.pure
    body = (o, a, c, r, n, t) -> nothing
    @test !RuleSpec(kind = :message, node = H.Gauss, target = Target{:out}, algorithm = BP, signature = RuleArgs, body = body, pure = false).pure
end

@testitem "gate:routing" tags = [:base, :alloc] setup = [HandRules] begin
    using MessagePassingRulesBase
    using MessagePassingRulesBase: Target, BP
    H = HandRules
    # Resolve and execute at a call site that can reach exactly one rule.
    @test (@inferred message_passing_rule(H.Gauss, Target(:out), BP(), H.POINT)) === 3.0
    @test H.measure(H.POINT) == 0
end
