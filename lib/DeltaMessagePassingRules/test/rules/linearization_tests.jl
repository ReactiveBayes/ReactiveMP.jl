# The Linearization rules against v6's own tables (ReactiveMP 6.5.0's
# `test/rules/delta/linearization/`), with their type-promotion checks. The functions are v6's,
# with integer constants, so that a Float32 input stays Float32.

@testitem "rules:Delta:linearization:out" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily
    using .DeltaTestNode: context
    linearized = DeltaApproximation(method = Linearization())

    g(x) = x .^ 2 .- 5
    t, v = 2, 5
    g_closure(x) = x .^ t .- v          # a function of the enclosing scope
    g_sum(x) = sum(x)                   # v6's dot(x, ones(length(x)))
    h(x, y) = x .^ 2 .- y

    @test_message_update_rule(
        node = DeltaFn, target = :out, algorithm = linearized,
        cases = [
            # ForneyLab: test_delta_extended, SPDeltaEOutNG 1 and 2
            (m = (in = (NormalMeanVariance(2.0, 3.0),),), ctx = context(g)) => NormalMeanVariance(-1.0, 48.0),
            (m = (in = (MvNormalMeanCovariance([2.0], [3.0;;]),),), ctx = context(g)) => MvNormalMeanCovariance([-1.0], [48.0;;]),
            (m = (in = (NormalMeanVariance(2.0, 3.0),),), ctx = context(g_closure)) => NormalMeanVariance(-1.0, 48.0),
            (m = (in = (MvNormalMeanCovariance([2.0], [3.0;;]),),), ctx = context(g_closure)) => MvNormalMeanCovariance([-1.0], [48.0;;]),
            (m = (in = (MvNormalMeanCovariance(ones(2), [1.0 0.0; 0.0 1.0]),),), ctx = context(g_sum)) => NormalMeanVariance(2.0, 2.0),
            # ForneyLab: test_delta_extended, SPDeltaEOutNGX 1 and 2
            (m = (in = (NormalMeanVariance(2.0, 3.0), NormalMeanVariance(5.0, 1.0)),), ctx = context(h)) => NormalMeanVariance(-1.0, 49.0),
            (m = (in = (MvNormalMeanCovariance([2.0], [3.0;;]), MvNormalMeanCovariance([5.0], [1.0;;])),), ctx = context(h)) => MvNormalMeanCovariance([-1.0], [49.0;;]),
        ],
    )
end

@testitem "rules:Delta:linearization:in" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily

    g_inv(y) = sqrt.(y .+ 5)
    h_inv_x(z, y) = sqrt.(z .+ y)
    h_inv_z(x, y) = x .^ 2 .- y

    # A known inverse: the target's own member arrives as `nothing`.
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = DeltaApproximation(method = Linearization(), inverse = g_inv), atol = 1.0e-5,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (nothing,)),) => NormalMeanVariance(2.6457513110645907, 0.10714285714285711),
            (m = (out = MvNormalMeanCovariance([2.0], [3.0;;]), in = (nothing,)),) => MvNormalMeanCovariance([2.6457513110645907], [0.10714285714285711;;]),
        ],
    )
    inverses = DeltaApproximation(method = Linearization(), inverse = (h_inv_x, h_inv_z))
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = inverses,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (nothing, NormalMeanVariance(5.0, 1.0))),) => NormalMeanVariance(2.6457513110645907, 0.14285714285714282),
            (m = (out = MvNormalMeanCovariance([2.0], [3.0;;]), in = (nothing, MvNormalMeanCovariance([5.0], [1.0;;]))),) => MvNormalMeanCovariance([2.6457513110645907], [0.14285714285714282;;]),
        ],
    )
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 2), algorithm = inverses, atol = 1.0e-5,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 1.0), in = (NormalMeanVariance(5.0, 1.0), nothing)),) => NormalMeanVariance(-1.0, 17.0),
            (m = (out = MvNormalMeanCovariance([2.0], [1.0;;]), in = (MvNormalMeanCovariance([5.0], [1.0;;]), nothing)),) => MvNormalMeanCovariance([-1.0], [17.0;;]),
        ],
    )

    # No inverse: the input's share of the joint over the inputs, divided by its own message.
    linearized = DeltaApproximation(method = Linearization())
    I2, I3 = [1.0 0.0; 0.0 1.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 1), algorithm = linearized, atol = 1.0e-3,
        cases = [
            (m = (in = (NormalMeanVariance(5.0, 10.0), nothing),), clusters = ((:in,) => JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((), ())),)) => NormalWeightedMeanPrecision(0.5, 0.9),
            (m = (in = (MvNormalMeanCovariance([5.0], [10.0;;]), nothing),), clusters = ((:in,) => JointNormal(MvNormalMeanCovariance(ones(2), [1.0 0.1; 0.1 1.0]), ((1,), (1,))),)) => MvNormalWeightedMeanPrecision([0.5], [0.9;;]),
        ],
    )
    @test_message_update_rule(
        node = DeltaFn, target = (:in, 2), algorithm = linearized,
        cases = [
            (m = (in = (nothing, NormalMeanVariance(0.0, 10.0), nothing),), clusters = ((:in,) => JointNormal(MvNormalMeanCovariance(ones(3), I3), ((), (), ())),)) => NormalWeightedMeanPrecision(1.0, 0.9),
            (m = (in = (nothing, MvNormalMeanCovariance(zeros(2), 10 * I2), nothing),), clusters = ((:in,) => JointNormal(MvNormalMeanCovariance(ones(3), I3), ((1,), (2,), ())),)) => MvNormalWeightedMeanPrecision(ones(2), 0.9 * I2),
        ],
    )
end

@testitem "rules:Delta:linearization:marginals" tags = [:rules] setup = [DeltaTestNode] begin
    using DeltaMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesApproximations, ExponentialFamily
    using .DeltaTestNode: context
    linearized = DeltaApproximation(method = Linearization())

    g(x) = x .^ 2 .- 5
    h(x, y) = x .^ 2 .- y
    joint_h = MvNormalMeanCovariance([2.6, 4.85], [0.20000000000000007 0.19999999999999998; 0.19999999999999998 0.95])

    @test_marginal_update_rule(
        node = DeltaFn, target = (:in,), algorithm = linearized,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (NormalMeanVariance(2.0, 1.0),)), ctx = context(g)) => JointNormal(NormalMeanVariance(2.6315789473684212, 0.1578947368421053), ((),)),
            (m = (out = MvNormalMeanCovariance([2.0], [3.0;;]), in = (MvNormalMeanCovariance([2.0], [1.0;;]),)), ctx = context(g)) => JointNormal(MvNormalMeanCovariance([2.6315789473684212], [0.1578947368421053;;]), ((1,),)),
            (m = (out = NormalMeanVariance(2.0, 3.0), in = (NormalMeanVariance(2.0, 1.0), NormalMeanVariance(5.0, 1.0))), ctx = context(h)) => JointNormal(joint_h, ((), ())),
            # ForneyLab: test_delta_extended, MDeltaEInGX 2. v6's comment doubts the sizes of the
            # left block; they are the inputs' own, one each.
            (m = (out = MvNormalMeanCovariance([2.0], [3.0;;]), in = (MvNormalMeanCovariance([2.0], [1.0;;]), MvNormalMeanCovariance([5.0], [1.0;;]))), ctx = context(h)) => JointNormal(joint_h, ((1,), (1,))),
        ],
    )
end
