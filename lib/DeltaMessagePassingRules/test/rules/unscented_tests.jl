# The Unscented rules against v6's own tables (ReactiveMP 6.5.0's `test/rules/delta/unscented/`), which
# check no type promotion either. A rule reaches the function through `getnodefn(ctx.node, …)`,
# which the engine's node implements; here a test node stands in for it.

@testmodule DeltaTestNode begin
    using MessagePassingRulesBase: MessagePassingRulesBase, Target, RuleContext

    struct FunctionNode{F}
        f::F
    end
    MessagePassingRulesBase.getnodefn(node::FunctionNode, ::Target{:out}) = node.f

    context(f) = RuleContext(node = FunctionNode(f))

    g(x) = x .^ 2 .- 5.0
    g_inv(y) = sqrt.(y .+ 5.0)
    h(x, y) = x .^ 2 .- y
    h_inv_x(z, y) = sqrt.(z .+ y)
    h_inv_z(x, y) = x .^ 2 .- y
end

@testitem "rules:Delta:unscented:out" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily
    using .DeltaTestNode: context, g, h
    plain = DeltaApproximation(method = Unscented())
    alpha = DeltaApproximation(method = Unscented(; alpha = 1.0))

    @test_message_update_rule(
        node = DeltaFn{typeof(g)}, target = :out, algorithm = plain, check_type_promotion = false,
        cases = [
            (m = (in = (NormalMeanVariance(2.0, 3.0),),), ctx = context(g)) => NormalMeanVariance(2.0000000001164153, 66.00000000093132),
            (m = (in = (MvNormalMeanCovariance([2.0], [3.0]),),), ctx = context(g)) => MvNormalMeanCovariance([2.0000000001164153], [66.00000000093132]),
            (m = (in = (NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0)),), ctx = context(h)) => NormalMeanVariance(1.9999999997671694, 67.00000899657607),
        ],
    )
    @test_message_update_rule(
        node = DeltaFn{typeof(g)}, target = :out, algorithm = alpha, check_type_promotion = false,
        cases = [
            (m = (in = (NormalMeanVariance(2.0, 3.0),),), ctx = context(g)) => NormalMeanVariance(2.0, 66.0),
            (m = (in = (MvNormalMeanCovariance([2.0], [3.0]),),), ctx = context(g)) => MvNormalMeanCovariance([2.0], [66.0]),
        ],
    )
end

@testitem "rules:Delta:unscented:in" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily
    using .DeltaTestNode: g_inv, h_inv_x, h_inv_z

    # A known inverse: the other members arrive in the group, the target's own as `nothing`.
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = DeltaApproximation(method = Unscented(), inverse = g_inv), check_type_promotion = false,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (nothing,)),) => NormalMeanVariance(2.6255032138433307, 0.10796282966583703),
            (m = (out = MvNormalMeanCovariance([2.0], [3.0;;]), in = (nothing,)),) => MvNormalMeanCovariance([2.6255032138433307], [0.10796282966583703;;]),
        ],
    )
    inverses = DeltaApproximation(method = Unscented(), inverse = (h_inv_x, h_inv_z))
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = inverses, check_type_promotion = false,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (nothing, NormalMeanVariance(5.0, 1.0))),) => NormalMeanVariance(2.6187538476660848, 0.14431487274498522),
            (m = (out = MvNormalMeanCovariance([2.0], [3.0;;]), in = (nothing, MvNormalMeanCovariance([5.0], [1.0;;]))),) => MvNormalMeanCovariance([2.6187538476660848], [0.14431487274498522;;]),
        ],
    )
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 2), algorithm = inverses, check_type_promotion = false,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 1.0), in = (NormalMeanVariance(3.0, 1.0), nothing)),) => NormalMeanVariance(2.0000000002328306, 19.00000100088073),
        ],
    )

    # No inverse: the input's share of the joint over the inputs, divided by its own message.
    plain = DeltaApproximation(method = Unscented())
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = plain, check_type_promotion = false, atol = 1.0e-3,
        cases = [
            (m = (in = (NormalMeanVariance(5.0, 10.0), nothing),), clusters = ((:in,) => JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((), ())),)) => NormalWeightedMeanPrecision(0.5, 0.9),
            (m = (in = (MvNormalMeanCovariance([5.0], [10.0;;]), nothing),), clusters = ((:in,) => JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((1,), (1,))),)) => MvNormalWeightedMeanPrecision([0.5], [0.9;;]),
        ],
    )
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 2), algorithm = plain, check_type_promotion = false,
        cases = [
            (m = (in = (nothing, NormalMeanVariance(0.0, 10.0), nothing),), clusters = ((:in,) => JointNormal(MvNormalMeanCovariance(ones(3), [1.0 0 0; 0 1.0 0; 0 0 1.0]), ((), (), ())),)) => NormalWeightedMeanPrecision(1.0, 0.9),
        ],
    )
end

@testitem "rules:Delta:unscented:marginals" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily
    using .DeltaTestNode: context, g, h
    plain = DeltaApproximation(method = Unscented())

    @test_marginal_update_rule(
        node = DeltaFn, target = (:in,), algorithm = plain, check_type_promotion = false, atol = 1.0e-10,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (NormalMeanVariance(2.0, 1.0),)), ctx = context(g)) => JointNormal(NormalMeanVariance(2.3809523807887425, 0.23809523822182999), ((),)),
        ],
    )
    @test_marginal_update_rule(
        node = DeltaFn, target = (:in,), algorithm = plain, check_type_promotion = false,
        cases = [
            (m = (out = MvNormalMeanCovariance([2.0], [3.0]), in = (MvNormalMeanCovariance([2.0], [1.0;;]),)), ctx = context(g)) => JointNormal(MvNormalMeanCovariance([2.3809523807887425], [0.23809523822182999;;]), ((1,),)),
        ],
    )
    # ForneyLab: test_delta_unscented, MDeltaUTInGX 1
    @test_marginal_update_rule(
        node = DeltaFn, target = (:in,), algorithm = plain, check_type_promotion = false, atol = 1.0e-4,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (NormalMeanVariance(2.0, 1.0), NormalMeanVariance(5.0, 1.0))), ctx = context(h)) =>
                JointNormal(MvNormalMeanCovariance([2.3636363470614055, 4.9090909132334355], [0.2727273058237252 0.1818181735464949; 0.18181817354649488 0.9545454566127697]), ((), ())),
        ],
    )
end

@testitem "delta:algorithm" tags = [:rules] begin
    using DeltaMessagePassingRules, MessagePassingRulesApproximations
    using MessagePassingRulesBase: dependencies_spec, free_energy_partition, target_dependencies, IndexedTarget

    # No constructor skips the guard, and its error names the methods the node takes.
    @test_throws ArgumentError DeltaApproximation(method = :nonsense)
    @test_throws ArgumentError DeltaApproximation(:nonsense, nothing)
    message = try
        DeltaApproximation(:nonsense, nothing)
    catch error
        sprint(showerror, error)
    end
    @test contains(message, "`nonsense` is not an approximation method of the Delta node")
    @test contains(message, "Unscented()") && contains(message, "Linearization()")
    @test contains(message, "ExponentialFamilyProjection")
    @test DeltaApproximation(method = Unscented()).inverse === nothing
    @test DeltaApproximation(Linearization(), identity).inverse === identity

    # The inverse decides what a message towards an input consumes; free energy counts the
    # joint over the inputs either way.
    unknown = dependencies_spec(DeltaFn, DeltaApproximation(method = Unscented()))
    known = dependencies_spec(DeltaFn, DeltaApproximation(method = Unscented(), inverse = identity))
    @test map(d -> d.container, target_dependencies(unknown, IndexedTarget(:in, 1))) == (:m, :q)
    @test map(d -> d.container, target_dependencies(known, IndexedTarget(:in, 1))) == (:m, :m)
    @test free_energy_partition(unknown) == free_energy_partition(known) == ((:out,), (:in,))
end
