# `+`: tables of cases; the joint of Gaussian inputs is one MvNormalWeightedMeanPrecision.

@testitem "rules:+:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = +, target = :out,
        cases = [
            (m = (in = (PointMass(1.0), PointMass(1.0)),),) => PointMass(2.0),
            (m = (in = (PointMass([2.0]), PointMass([-2.0])),),) => PointMass([0.0]),
            (m = (in = (PointMass([1.0 2.0; 3.0 4.0]), PointMass([-2.0 -1.0; -4.0 -3.0])),),) => PointMass([-1.0 1.0; -1.0 1.0]),
            (m = (in = (NormalMeanVariance(1.0, 2.0), PointMass(1.0)),),) => NormalMeanVariance(2.0, 2.0),
            (m = (in = (NormalMeanVariance(-1.0, 3.0), PointMass(-3.0)),),) => NormalMeanVariance(-4.0, 3.0),
            (m = (in = (NormalMeanPrecision(4.0, 7.0), PointMass(1.0)),),) => NormalMeanPrecision(5.0, 7.0),
            (m = (in = (NormalMeanPrecision(-1.0, 2.0), PointMass(-3.0)),),) => NormalMeanPrecision(-4.0, 2.0),
            (m = (in = (NormalWeightedMeanPrecision(4.0, 2.0), PointMass(1.0)),),) => NormalMeanVariance(3.0, 1 / 2),
            (m = (in = (NormalWeightedMeanPrecision(8.0, 4.0), PointMass(-3.0)),),) => NormalMeanVariance(-1.0, 1 / 4),
            (m = (in = (PointMass(1.0), NormalMeanVariance(1.0, 2.0)),),) => NormalMeanVariance(2.0, 2.0),
            (m = (in = (PointMass(-3.0), NormalMeanVariance(-1.0, 3.0)),),) => NormalMeanVariance(-4.0, 3.0),
            (m = (in = (PointMass(1.0), NormalMeanPrecision(4.0, 7.0)),),) => NormalMeanPrecision(5.0, 7.0),
            (m = (in = (PointMass(-3.0), NormalMeanPrecision(-1.0, 2.0)),),) => NormalMeanPrecision(-4.0, 2.0),
            (m = (in = (PointMass(1.0), NormalWeightedMeanPrecision(4.0, 2.0)),),) => NormalMeanVariance(3.0, 1 / 2),
            (m = (in = (PointMass(-3.0), NormalWeightedMeanPrecision(8.0, 4.0)),),) => NormalMeanVariance(-1.0, 1 / 4),
            (m = (in = (MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), PointMass([1.0, 1.0])),),) => MvNormalMeanCovariance([2.0, 4.0], [3.0 2.0; 2.0 4.0]),
            (m = (in = (MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), PointMass([1.0, 1.0])),),) => MvNormalMeanCovariance([-3.0, 4.0], [3.0 2.0; 2.0 4.0]),
            (m = (in = (MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), PointMass([1.0, 1.0])),),) => MvNormalMeanPrecision([-3.0, 4.0], [3.0 2.0; 2.0 4.0]),
            (m = (in = (MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), PointMass([-2.0, 1.0])),),) => MvNormalMeanPrecision([-6.0, 4.0], [3.0 2.0; 2.0 4.0]),
            (m = (in = (MvNormalWeightedMeanPrecision([1.0, 3.0], [1.0 0.0; 0.0 1.0]), PointMass([2.0, 7.0])),),) => MvNormalWeightedMeanPrecision([3.0, 10.0], [1.0 0.0; 0.0 1.0]),
            (m = (in = (MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.0; 0.0 2.0]), PointMass([1.0, 3.0])),),) => MvNormalWeightedMeanPrecision([4.0, 10.0], [2.0 0.0; 0.0 2.0]),
            (m = (in = (PointMass([1.0, 1.0]), MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),),) => MvNormalMeanCovariance([2.0, 4.0], [3.0 2.0; 2.0 4.0]),
            (m = (in = (PointMass([1.0, 1.0]), MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0])),),) => MvNormalMeanCovariance([-3.0, 4.0], [3.0 2.0; 2.0 4.0]),
            (m = (in = (PointMass([1.0, 1.0]), MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0])),),) => MvNormalMeanPrecision([-3.0, 4.0], [3.0 2.0; 2.0 4.0]),
            (m = (in = (PointMass([-2.0, 1.0]), MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0])),),) => MvNormalMeanPrecision([-6.0, 4.0], [3.0 2.0; 2.0 4.0]),
            (m = (in = (PointMass([2.0, 7.0]), MvNormalWeightedMeanPrecision([1.0, 3.0], [1.0 0.0; 0.0 1.0])),),) => MvNormalWeightedMeanPrecision([3.0, 10.0], [1.0 0.0; 0.0 1.0]),
            (m = (in = (PointMass([1.0, 3.0]), MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.0; 0.0 2.0])),),) => MvNormalWeightedMeanPrecision([4.0, 10.0], [2.0 0.0; 0.0 2.0]),
            (m = (in = (NormalMeanVariance(1.0, 2.0), NormalMeanVariance(3.0, 4.0)),),) => NormalMeanVariance(4.0, 6.0),
            (m = (in = (NormalMeanVariance(-1.0, 2.0), NormalMeanVariance(-2.0, 3.0)),),) => NormalMeanVariance(-3.0, 5.0),
            (m = (in = (NormalMeanPrecision(2.0, 2.0), NormalMeanPrecision(-1.0, 3.0)),),) => NormalMeanVariance(1.0, (2.0 + 3.0) / (2.0 * 3.0)),
            (m = (in = (NormalMeanPrecision(-1.0, 2.0), NormalMeanPrecision(-1.0, 3.0)),),) => NormalMeanVariance(-2.0, (2.0 + 3.0) / (2.0 * 3.0)),
            (m = (in = (NormalMeanPrecision(2.0, 2.0), NormalMeanVariance(-1.0, 3.0)),),) => NormalMeanVariance(1.0, 3.5),
            (m = (in = (NormalWeightedMeanPrecision(8.0, 4.0), NormalMeanVariance(-3.0, 1.0)),),) => NormalMeanVariance(-1.0, 5 / 4),
            (m = (in = (NormalMeanVariance(-4.0, 2.0), NormalWeightedMeanPrecision(6.0, 3.0)),),) => NormalMeanVariance(-2.0, 7 / 3),
            (m = (in = (MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0])),),) => MvNormalMeanCovariance([2.0, 6.0], [6.0 4.0; 4.0 8.0]),
            (m = (in = (MvNormalMeanCovariance([-1.0, 3.0], [3.0 2.0; 2.0 4.0]), MvNormalMeanCovariance([0.0, 3.0], [3.0 2.0; 2.0 4.0])),),) => MvNormalMeanCovariance([-1.0, 6.0], [6.0 4.0; 4.0 8.0]),
            (m = (in = (MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]), MvNormalMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0])),),) => MvNormalMeanCovariance([2.0, -7.0], [1.0 0.0; 0.0 4 / 3]),
            (m = (in = (MvNormalMeanCovariance([1.0, -1.0], [3.0 1.0; 1.0 4.0]), MvNormalMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0])),),) => MvNormalMeanCovariance([2.0, 3.0], [36 / 10 4 / 5; 4 / 5 44 / 10]),
            (m = (in = (MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]), MvNormalWeightedMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0])),),) => MvNormalMeanCovariance([3 / 2, -7 / 3], [1.0 0.0; 0.0 4 / 3]),
            (m = (in = (MvNormalMeanCovariance([1.0, -1.0], [3.0 1.0; 1.0 4.0]), MvNormalWeightedMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0])),),) => MvNormalMeanCovariance([4 / 5, 2 / 5], [36 / 10 4 / 5; 4 / 5 44 / 10]),
            (m = (in = (MvNormalWeightedMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0]), MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0])),),) => MvNormalMeanCovariance([3 / 2, -7 / 3], [1.0 0.0; 0.0 4 / 3]),
            (m = (in = (MvNormalWeightedMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0]), MvNormalMeanCovariance([1.0, -1.0], [3.0 1.0; 1.0 4.0])),),) => MvNormalMeanCovariance([4 / 5, 2 / 5], [36 / 10 4 / 5; 4 / 5 44 / 10]),
            (m = (in = (MvNormalWeightedMeanPrecision([1.0, 4.0], [1.0 0.0; 0.0 1.0]), MvNormalWeightedMeanPrecision([1.0, 1.0], [1.0 0.0; 0.0 1.0])),),) => MvNormalMeanCovariance([2.0, 5.0], [2.0 0.0; 0.0 2.0]),
            (m = (in = (MvNormalWeightedMeanPrecision([1.0, 4.0], [1.0 0.0; 0.0 1.0]), MvNormalWeightedMeanPrecision([1.0, -1.0], [2.0 0.0; 0.0 2.0])),),) => MvNormalMeanCovariance([1.5, 3.5], [1.5 0.0; 0.0 1.5]),
        ],
    )
end

@testitem "rules:+:in" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    @test_message_update_rule(
        node = +, target = (:in, 1),
        cases = [
            (m = (out = PointMass(1.0), in = (nothing, PointMass(-1.0))),) => PointMass(2.0),
            (m = (out = PointMass([1.0]), in = (nothing, PointMass([-2.0]))),) => PointMass([3.0]),
            (m = (out = PointMass([1.0 2.0; 3.0 4.0]), in = (nothing, PointMass([-2.0 -1.0; -4.0 -1.0]))),) => PointMass([3.0 3.0; 7.0 5.0]),
            (m = (out = NormalMeanVariance(1.0, 2.0), in = (nothing, PointMass(2.0))),) => NormalMeanVariance(-1.0, 2.0),
            (m = (out = NormalMeanVariance(-1.0, 3.0), in = (nothing, PointMass(-3.0))),) => NormalMeanVariance(2.0, 3.0),
            (m = (out = NormalMeanPrecision(4.0, 7.0), in = (nothing, PointMass(1.0))),) => NormalMeanPrecision(3.0, 7.0),
            (m = (out = NormalMeanPrecision(-1.0, 2.0), in = (nothing, PointMass(-3.0))),) => NormalMeanPrecision(2.0, 2.0),
            (m = (out = NormalWeightedMeanPrecision(4.0, 2.0), in = (nothing, PointMass(1.0))),) => NormalMeanVariance(1.0, 1 / 2),
            (m = (out = NormalWeightedMeanPrecision(8.0, 4.0), in = (nothing, PointMass(-3.0))),) => NormalMeanVariance(5.0, 1 / 4),
            (m = (out = PointMass(1.0), in = (nothing, NormalMeanVariance(1.0, 2.0))),) => NormalMeanVariance(0.0, 2.0),
            (m = (out = PointMass(-3.0), in = (nothing, NormalMeanVariance(-1.0, 3.0))),) => NormalMeanVariance(-2.0, 3.0),
            (m = (out = PointMass(1.0), in = (nothing, NormalMeanPrecision(4.0, 7.0))),) => NormalMeanPrecision(-3.0, 7.0),
            (m = (out = PointMass(-3.0), in = (nothing, NormalMeanPrecision(-1.0, 2.0))),) => NormalMeanPrecision(-2.0, 2.0),
            (m = (out = PointMass(1.0), in = (nothing, NormalWeightedMeanPrecision(4.0, 2.0))),) => NormalMeanVariance(-1.0, 1 / 2),
            (m = (out = PointMass(-3.0), in = (nothing, NormalWeightedMeanPrecision(8.0, 4.0))),) => NormalMeanVariance(-5.0, 1 / 4),
            (m = (out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (nothing, PointMass([1.0, 1.0]))),) => MvNormalMeanCovariance([0.0, 2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (nothing, PointMass([1.0, 1.0]))),) => MvNormalMeanCovariance([-5.0, 2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (nothing, PointMass([3.0, 2.0]))),) => MvNormalMeanPrecision([-7.0, 1.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (nothing, PointMass([-2.0, 3.0]))),) => MvNormalMeanPrecision([-2.0, 0.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 3.0], [1.0 0.0; 0.0 1.0]), in = (nothing, PointMass([2.0, 7.0]))),) => MvNormalWeightedMeanPrecision([-1.0, -4.0], [1.0 0.0; 0.0 1.0]),
            (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.0; 0.0 2.0]), in = (nothing, PointMass([1.0, 3.0]))),) => MvNormalWeightedMeanPrecision([0.0, -2.0], [2.0 0.0; 0.0 2.0]),
            (m = (out = PointMass([1.0, 1.0]), in = (nothing, MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => MvNormalMeanCovariance([0.0, -2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = PointMass([1.0, 1.0]), in = (nothing, MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => MvNormalMeanCovariance([5.0, -2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = PointMass([1.0, 1.0]), in = (nothing, MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => MvNormalMeanPrecision([5.0, -2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = PointMass([-2.0, 1.0]), in = (nothing, MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => MvNormalMeanPrecision([2.0, -2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = PointMass([2.0, 7.0]), in = (nothing, MvNormalWeightedMeanPrecision([1.0, 3.0], [1.0 0.0; 0.0 1.0]))),) => MvNormalWeightedMeanPrecision([1.0, 4.0], [1.0 0.0; 0.0 1.0]),
            (m = (out = PointMass([1.0, 3.0]), in = (nothing, MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.0; 0.0 2.0]))),) => MvNormalWeightedMeanPrecision([0.0, 2.0], [2.0 0.0; 0.0 2.0]),
            (m = (out = NormalMeanVariance(1.0, 2.0), in = (nothing, NormalMeanVariance(3.0, 4.0))),) => NormalMeanVariance(-2.0, 6.0),
            (m = (out = NormalMeanVariance(-1.0, 2.0), in = (nothing, NormalMeanVariance(-2.0, 3.0))),) => NormalMeanVariance(1.0, 5.0),
            (m = (out = NormalMeanPrecision(2.0, 2.0), in = (nothing, NormalMeanPrecision(-1.0, 3.0))),) => NormalMeanVariance(3.0, (2.0 + 3.0) / (2.0 * 3.0)),
            (m = (out = NormalMeanPrecision(-1.0, 2.0), in = (nothing, NormalMeanPrecision(-1.0, 3.0))),) => NormalMeanVariance(0.0, (2.0 + 3.0) / (2.0 * 3.0)),
            (m = (out = NormalMeanPrecision(2.0, 2.0), in = (nothing, NormalMeanVariance(-1.0, 3.0))),) => NormalMeanVariance(3.0, 3.5),
            (m = (out = NormalWeightedMeanPrecision(8.0, 4.0), in = (nothing, NormalMeanVariance(-3.0, 1.0))),) => NormalMeanVariance(5.0, 5 / 4),
            (m = (out = NormalMeanVariance(-4.0, 2.0), in = (nothing, NormalWeightedMeanPrecision(6.0, 3.0))),) => NormalMeanVariance(-6.0, 7 / 3),
            (m = (out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (nothing, MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => MvNormalMeanCovariance([0.0, 0.0], [6.0 4.0; 4.0 8.0]),
            (m = (out = MvNormalMeanCovariance([-1.0, 1.0], [3.0 2.0; 2.0 4.0]), in = (nothing, MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => MvNormalMeanCovariance([-2.0, -2.0], [6.0 4.0; 4.0 8.0]),
            (m = (out = MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]), in = (nothing, MvNormalMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0]))),) => MvNormalMeanCovariance([0.0, 7.0], [1.0 0.0; 0.0 4 / 3]),
            (m = (out = MvNormalMeanCovariance([1.0, -1.0], [3.0 1.0; 1.0 4.0]), in = (nothing, MvNormalMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0]))),) => MvNormalMeanCovariance([0.0, -5.0], [36 / 10 4 / 5; 4 / 5 44 / 10]),
            (m = (out = MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]), in = (nothing, MvNormalWeightedMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0]))),) => MvNormalMeanCovariance([1 / 2, 7 / 3], [1.0 0.0; 0.0 4 / 3]),
            (m = (out = MvNormalMeanCovariance([1.0, 1.0], [3.0 1.0; 1.0 4.0]), in = (nothing, MvNormalWeightedMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0]))),) => MvNormalMeanCovariance([6 / 5, -2 / 5], [36 / 10 4 / 5; 4 / 5 44 / 10]),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0]), in = (nothing, MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]))),) => MvNormalMeanCovariance([-1 / 2, -7 / 3], [1.0 0.0; 0.0 4 / 3]),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0]), in = (nothing, MvNormalMeanCovariance([1.0, 1.0], [3.0 1.0; 1.0 4.0]))),) => MvNormalMeanCovariance([-6 / 5, 2 / 5], [36 / 10 4 / 5; 4 / 5 44 / 10]),
        ],
    )
    @test_message_update_rule(
        node = +, target = (:in, 2),
        cases = [
            (m = (out = PointMass(1.0), in = (PointMass(-1.0), nothing)),) => PointMass(2.0),
            (m = (out = PointMass([1.0]), in = (PointMass([-2.0]), nothing)),) => PointMass([3.0]),
            (m = (out = PointMass([1.0 2.0; 3.0 4.0]), in = (PointMass([-2.0 -1.0; -4.0 -1.0]), nothing)),) => PointMass([3.0 3.0; 7.0 5.0]),
            (m = (out = NormalMeanVariance(1.0, 2.0), in = (PointMass(2.0), nothing)),) => NormalMeanVariance(-1.0, 2.0),
            (m = (out = NormalMeanVariance(-1.0, 3.0), in = (PointMass(-3.0), nothing)),) => NormalMeanVariance(2.0, 3.0),
            (m = (out = NormalMeanPrecision(4.0, 7.0), in = (PointMass(1.0), nothing)),) => NormalMeanPrecision(3.0, 7.0),
            (m = (out = NormalMeanPrecision(-1.0, 2.0), in = (PointMass(-3.0), nothing)),) => NormalMeanPrecision(2.0, 2.0),
            (m = (out = NormalWeightedMeanPrecision(4.0, 2.0), in = (PointMass(1.0), nothing)),) => NormalMeanVariance(1.0, 1 / 2),
            (m = (out = NormalWeightedMeanPrecision(8.0, 4.0), in = (PointMass(-3.0), nothing)),) => NormalMeanVariance(5.0, 1 / 4),
            (m = (out = PointMass(1.0), in = (NormalMeanVariance(1.0, 2.0), nothing)),) => NormalMeanVariance(0.0, 2.0),
            (m = (out = PointMass(-3.0), in = (NormalMeanVariance(-1.0, 3.0), nothing)),) => NormalMeanVariance(-2.0, 3.0),
            (m = (out = PointMass(1.0), in = (NormalMeanPrecision(4.0, 7.0), nothing)),) => NormalMeanPrecision(-3.0, 7.0),
            (m = (out = PointMass(-3.0), in = (NormalMeanPrecision(-1.0, 2.0), nothing)),) => NormalMeanPrecision(-2.0, 2.0),
            (m = (out = PointMass(1.0), in = (NormalWeightedMeanPrecision(4.0, 2.0), nothing)),) => NormalMeanVariance(-1.0, 1 / 2),
            (m = (out = PointMass(-3.0), in = (NormalWeightedMeanPrecision(8.0, 4.0), nothing)),) => NormalMeanVariance(-5.0, 1 / 4),
            (m = (out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (PointMass([1.0, 1.0]), nothing)),) => MvNormalMeanCovariance([0.0, 2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (PointMass([1.0, 1.0]), nothing)),) => MvNormalMeanCovariance([-5.0, 2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (PointMass([3.0, 2.0]), nothing)),) => MvNormalMeanPrecision([-7.0, 1.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (PointMass([-2.0, 3.0]), nothing)),) => MvNormalMeanPrecision([-2.0, 0.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 3.0], [1.0 0.0; 0.0 1.0]), in = (PointMass([2.0, 7.0]), nothing)),) => MvNormalWeightedMeanPrecision([-1.0, -4.0], [1.0 0.0; 0.0 1.0]),
            (m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.0; 0.0 2.0]), in = (PointMass([1.0, 3.0]), nothing)),) => MvNormalWeightedMeanPrecision([0.0, -2.0], [2.0 0.0; 0.0 2.0]),
            (m = (out = PointMass([1.0, 1.0]), in = (MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), nothing)),) => MvNormalMeanCovariance([0.0, -2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = PointMass([1.0, 1.0]), in = (MvNormalMeanCovariance([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), nothing)),) => MvNormalMeanCovariance([5.0, -2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = PointMass([1.0, 1.0]), in = (MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), nothing)),) => MvNormalMeanPrecision([5.0, -2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = PointMass([-2.0, 1.0]), in = (MvNormalMeanPrecision([-4.0, 3.0], [3.0 2.0; 2.0 4.0]), nothing)),) => MvNormalMeanPrecision([2.0, -2.0], [3.0 2.0; 2.0 4.0]),
            (m = (out = PointMass([2.0, 7.0]), in = (MvNormalWeightedMeanPrecision([1.0, 3.0], [1.0 0.0; 0.0 1.0]), nothing)),) => MvNormalWeightedMeanPrecision([1.0, 4.0], [1.0 0.0; 0.0 1.0]),
            (m = (out = PointMass([1.0, 3.0]), in = (MvNormalWeightedMeanPrecision([2.0, 4.0], [2.0 0.0; 0.0 2.0]), nothing)),) => MvNormalWeightedMeanPrecision([0.0, 2.0], [2.0 0.0; 0.0 2.0]),
            (m = (out = NormalMeanVariance(1.0, 2.0), in = (NormalMeanVariance(3.0, 4.0), nothing)),) => NormalMeanVariance(-2.0, 6.0),
            (m = (out = NormalMeanVariance(-1.0, 2.0), in = (NormalMeanVariance(-2.0, 3.0), nothing)),) => NormalMeanVariance(1.0, 5.0),
            (m = (out = NormalMeanPrecision(2.0, 2.0), in = (NormalMeanPrecision(-1.0, 3.0), nothing)),) => NormalMeanVariance(3.0, (2.0 + 3.0) / (2.0 * 3.0)),
            (m = (out = NormalMeanPrecision(-1.0, 2.0), in = (NormalMeanPrecision(-1.0, 3.0), nothing)),) => NormalMeanVariance(0.0, (2.0 + 3.0) / (2.0 * 3.0)),
            (m = (out = NormalMeanPrecision(2.0, 2.0), in = (NormalMeanVariance(-1.0, 3.0), nothing)),) => NormalMeanVariance(3.0, 3.5),
            (m = (out = NormalWeightedMeanPrecision(8.0, 4.0), in = (NormalMeanVariance(-3.0, 1.0), nothing)),) => NormalMeanVariance(5.0, 5 / 4),
            (m = (out = NormalMeanVariance(-4.0, 2.0), in = (NormalWeightedMeanPrecision(6.0, 3.0), nothing)),) => NormalMeanVariance(-6.0, 7 / 3),
            (m = (out = MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), in = (MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), nothing)),) => MvNormalMeanCovariance([0.0, 0.0], [6.0 4.0; 4.0 8.0]),
            (m = (out = MvNormalMeanCovariance([-1.0, 1.0], [3.0 2.0; 2.0 4.0]), in = (MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), nothing)),) => MvNormalMeanCovariance([-2.0, -2.0], [6.0 4.0; 4.0 8.0]),
            (m = (out = MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]), in = (MvNormalMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0]), nothing)),) => MvNormalMeanCovariance([0.0, 7.0], [1.0 0.0; 0.0 4 / 3]),
            (m = (out = MvNormalMeanCovariance([1.0, -1.0], [3.0 1.0; 1.0 4.0]), in = (MvNormalMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0]), nothing)),) => MvNormalMeanCovariance([0.0, -5.0], [36 / 10 4 / 5; 4 / 5 44 / 10]),
            (m = (out = MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]), in = (MvNormalWeightedMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0]), nothing)),) => MvNormalMeanCovariance([1 / 2, 7 / 3], [1.0 0.0; 0.0 4 / 3]),
            (m = (out = MvNormalMeanCovariance([1.0, 1.0], [3.0 1.0; 1.0 4.0]), in = (MvNormalWeightedMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0]), nothing)),) => MvNormalMeanCovariance([6 / 5, -2 / 5], [36 / 10 4 / 5; 4 / 5 44 / 10]),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, -7.0], [2.0 0.0; 0.0 3.0]), in = (MvNormalMeanPrecision([1.0, 0.0], [2.0 0.0; 0.0 1.0]), nothing)),) => MvNormalMeanCovariance([-1 / 2, -7 / 3], [1.0 0.0; 0.0 4 / 3]),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 4.0], [2.0 1.0; 1.0 3.0]), in = (MvNormalMeanCovariance([1.0, 1.0], [3.0 1.0; 1.0 4.0]), nothing)),) => MvNormalMeanCovariance([-6 / 5, 2 / 5], [36 / 10 4 / 5; 4 / 5 44 / 10]),
        ],
    )
end

@testitem "rules:+:marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    @test_marginal_update_rule(
        node = +, target = (:in,),
        cases = [
            (m = (out = NormalMeanVariance(3.0, 4.0), in = (NormalMeanVariance(2.0, 2.0), PointMass(2.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(5 / 4, 3 / 4), ((:in, 2),) => PointMass(2.0)),
            (m = (out = NormalMeanPrecision(3, 4), in = (NormalMeanPrecision(1, 4), PointMass(1.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(12.0, 8.0), ((:in, 2),) => PointMass(1.0)),
            (m = (out = NormalWeightedMeanPrecision(3.0, 4.0), in = (NormalWeightedMeanPrecision(1.0, 4.0), PointMass(1.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(0.0, 8.0), ((:in, 2),) => PointMass(1.0)),
            (m = (out = NormalMeanPrecision(2.0, 4.0), in = (NormalMeanVariance(1.0, 2.0), PointMass(1.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(9 / 2, 9 / 2), ((:in, 2),) => PointMass(1.0)),
            (m = (out = NormalMeanVariance(2.0, 4.0), in = (NormalMeanPrecision(1.0, 2.0), PointMass(1.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(9 / 4, 9 / 4), ((:in, 2),) => PointMass(1.0)),
            (m = (out = NormalMeanPrecision(3.0, 4.0), in = (NormalWeightedMeanPrecision(1.0, 2.0), PointMass(2.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(5.0, 6.0), ((:in, 2),) => PointMass(2.0)),
            (m = (out = NormalWeightedMeanPrecision(2.0, 2.0), in = (NormalMeanPrecision(1.0, 2.0), PointMass(-1.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(6.0, 4.0), ((:in, 2),) => PointMass(-1.0)),
            (m = (out = NormalMeanVariance(3.0, 3.0), in = (NormalWeightedMeanPrecision(2.0, 1.0), PointMass(2.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(7 / 3, 4 / 3), ((:in, 2),) => PointMass(2.0)),
            (m = (out = NormalWeightedMeanPrecision(2.0, 4.0), in = (NormalMeanVariance(2.0, 2.0), PointMass(1.0))),) => FactorizedCluster(((:in, 1),) => NormalWeightedMeanPrecision(-1.0, 9 / 2), ((:in, 2),) => PointMass(1.0)),
            (m = (out = NormalMeanVariance(3.0, 4.0), in = (PointMass(2.0), NormalMeanVariance(2.0, 2.0))),) => FactorizedCluster(((:in, 1),) => PointMass(2.0), ((:in, 2),) => NormalWeightedMeanPrecision(5 / 4, 3 / 4)),
            (m = (out = NormalMeanPrecision(3, 4), in = (PointMass(1.0), NormalMeanPrecision(1, 4))),) => FactorizedCluster(((:in, 1),) => PointMass(1.0), ((:in, 2),) => NormalWeightedMeanPrecision(12.0, 8.0)),
            (m = (out = NormalWeightedMeanPrecision(3.0, 4.0), in = (PointMass(1.0), NormalWeightedMeanPrecision(1.0, 4.0))),) => FactorizedCluster(((:in, 1),) => PointMass(1.0), ((:in, 2),) => NormalWeightedMeanPrecision(0.0, 8.0)),
            (m = (out = NormalMeanPrecision(2.0, 4.0), in = (PointMass(1.0), NormalMeanVariance(1.0, 2.0))),) => FactorizedCluster(((:in, 1),) => PointMass(1.0), ((:in, 2),) => NormalWeightedMeanPrecision(9 / 2, 9 / 2)),
            (m = (out = NormalMeanVariance(2.0, 4.0), in = (PointMass(1.0), NormalMeanPrecision(1.0, 2.0))),) => FactorizedCluster(((:in, 1),) => PointMass(1.0), ((:in, 2),) => NormalWeightedMeanPrecision(9 / 4, 9 / 4)),
            (m = (out = NormalMeanPrecision(3.0, 4.0), in = (PointMass(2.0), NormalWeightedMeanPrecision(1.0, 2.0))),) => FactorizedCluster(((:in, 1),) => PointMass(2.0), ((:in, 2),) => NormalWeightedMeanPrecision(5.0, 6.0)),
            (m = (out = NormalWeightedMeanPrecision(2.0, 2.0), in = (PointMass(-1.0), NormalMeanPrecision(1.0, 2.0))),) => FactorizedCluster(((:in, 1),) => PointMass(-1.0), ((:in, 2),) => NormalWeightedMeanPrecision(6.0, 4.0)),
            (m = (out = NormalMeanVariance(3.0, 3.0), in = (PointMass(2.0), NormalWeightedMeanPrecision(2.0, 1.0))),) => FactorizedCluster(((:in, 1),) => PointMass(2.0), ((:in, 2),) => NormalWeightedMeanPrecision(7 / 3, 4 / 3)),
            (m = (out = NormalWeightedMeanPrecision(2.0, 4.0), in = (PointMass(1.0), NormalMeanVariance(2.0, 2.0))),) => FactorizedCluster(((:in, 1),) => PointMass(1.0), ((:in, 2),) => NormalWeightedMeanPrecision(-1.0, 9 / 2)),
            (m = (out = MvNormalMeanCovariance([2.0, 2.0], [2.0 0.0; 0.0 2.0]), in = (MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]), PointMass([1.0, 1.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([1 / 4, 11 / 8], [1.0 -1 / 4; -1 / 4 7 / 8]), ((:in, 2),) => PointMass([1.0, 1.0])),
            (m = (out = MvNormalMeanCovariance([3.0, 2.0], [2.0 0.0; 0.0 2.0]), in = (MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]), PointMass([1.0, 1.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([4.0, 7 / 2], [5 / 2 1.0; 1.0 5 / 2]), ((:in, 2),) => PointMass([1.0, 1.0])),
            (m = (out = MvNormalMeanCovariance([3.0, 2.0], [4.0 2.0; 2.0 4.0]), in = (MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 0.0; 0.0 2.0]), PointMass([2.0, 1.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([7 / 6, 13 / 6], [7 / 3 -1 / 6; -1 / 6 7 / 3]), ((:in, 2),) => PointMass([2.0, 1.0])),
            (m = (out = MvNormalMeanPrecision([2.0, 2.0], [1.0 0.0; 0.0 1.0]), in = (MvNormalMeanCovariance([2.0, 3.0], [3.0 2.0; 2.0 4.0]), PointMass([1.0, -1.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([5 / 4, 29 / 8], [3 / 2 -1 / 4; -1 / 4 11 / 8]), ((:in, 2),) => PointMass([1.0, -1.0])),
            (m = (out = MvNormalMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 3.0]), in = (MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]), PointMass([2.0, 2.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([-1.0, -1.0], [5.0 2.0; 2.0 5.0]), ((:in, 2),) => PointMass([2.0, 2.0])),
            (m = (out = MvNormalMeanPrecision([3.0, 2.0], [4.0 2.0; 2.0 4.0]), in = (MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0]), PointMass([-2.0, 1.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([23.0, 16.0], [7.0 3.0; 3.0 5.0]), ((:in, 2),) => PointMass([-2.0, 1.0])),
            (m = (out = MvNormalWeightedMeanPrecision([2.0, 2.0], [1.0 0.0; 0.0 1.0]), in = (MvNormalMeanCovariance([2.0, 3.0], [3.0 2.0; 2.0 4.0]), PointMass([1.0, -1.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([5 / 4, 29 / 8], [3 / 2 -1 / 4; -1 / 4 11 / 8]), ((:in, 2),) => PointMass([1.0, -1.0])),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 2.0]), in = (MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]), PointMass([2.0, 2.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([-4.0, -2.0], [5.0 2.0; 2.0 4.0]), ((:in, 2),) => PointMass([2.0, 2.0])),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], [4.0 1.0; 1.0 4.0]), in = (MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 0.0; 0.0 2.0]), PointMass([-1.0, 1.0]))),) => FactorizedCluster(((:in, 1),) => MvNormalWeightedMeanPrecision([5.0, 0.0], [7.0 1.0; 1.0 6.0]), ((:in, 2),) => PointMass([-1.0, 1.0])),
            (m = (out = MvNormalMeanCovariance([2.0, 2.0], [2.0 0.0; 0.0 2.0]), in = (PointMass([1.0, 1.0]), MvNormalMeanCovariance([1.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([1.0, 1.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([1 / 4, 11 / 8], [1.0 -1 / 4; -1 / 4 7 / 8])),
            (m = (out = MvNormalMeanCovariance([3.0, 2.0], [2.0 0.0; 0.0 2.0]), in = (PointMass([1.0, 1.0]), MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([1.0, 1.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([4.0, 7 / 2], [5 / 2 1.0; 1.0 5 / 2])),
            (m = (out = MvNormalMeanCovariance([3.0, 2.0], [4.0 2.0; 2.0 4.0]), in = (PointMass([2.0, 1.0]), MvNormalWeightedMeanPrecision([1.0, 2.0], [2.0 0.0; 0.0 2.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([2.0, 1.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([7 / 6, 13 / 6], [7 / 3 -1 / 6; -1 / 6 7 / 3])),
            (m = (out = MvNormalMeanPrecision([2.0, 2.0], [1.0 0.0; 0.0 1.0]), in = (PointMass([1.0, -1.0]), MvNormalMeanCovariance([2.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([1.0, -1.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([5 / 4, 29 / 8], [3 / 2 -1 / 4; -1 / 4 11 / 8])),
            (m = (out = MvNormalMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 3.0]), in = (PointMass([2.0, 2.0]), MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([2.0, 2.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([-1.0, -1.0], [5.0 2.0; 2.0 5.0])),
            (m = (out = MvNormalMeanPrecision([3.0, 2.0], [4.0 2.0; 2.0 4.0]), in = (PointMass([-2.0, 1.0]), MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([-2.0, 1.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([23.0, 16.0], [7.0 3.0; 3.0 5.0])),
            (m = (out = MvNormalWeightedMeanPrecision([2.0, 2.0], [1.0 0.0; 0.0 1.0]), in = (PointMass([1.0, -1.0]), MvNormalMeanCovariance([2.0, 3.0], [3.0 2.0; 2.0 4.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([1.0, -1.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([5 / 4, 29 / 8], [3 / 2 -1 / 4; -1 / 4 11 / 8])),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 2.0]), in = (PointMass([2.0, 2.0]), MvNormalMeanPrecision([1.0, 1.0], [2.0 1.0; 1.0 2.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([2.0, 2.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([-4.0, -2.0], [5.0 2.0; 2.0 4.0])),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], [4.0 1.0; 1.0 4.0]), in = (PointMass([-1.0, 1.0]), MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 0.0; 0.0 2.0]))),) => FactorizedCluster(((:in, 1),) => PointMass([-1.0, 1.0]), ((:in, 2),) => MvNormalWeightedMeanPrecision([5.0, 0.0], [7.0 1.0; 1.0 6.0])),
            (m = (out = NormalMeanVariance(1.0, 2.0), in = (NormalMeanVariance(3.0, 4.0), NormalMeanVariance(5.0, 6.0))),) => MvNormalWeightedMeanPrecision([5 / 4, 4 / 3], [3 / 4 1 / 2; 1 / 2 2 / 3]),
            (m = (out = NormalMeanPrecision(1.0, 2.0), in = (NormalMeanPrecision(3.0, 4.0), NormalMeanPrecision(5.0, 6.0))),) => MvNormalWeightedMeanPrecision([14.0, 32.0], [6.0 2.0; 2.0 8.0]),
            (m = (out = NormalWeightedMeanPrecision(1.0, 2.0), in = (NormalWeightedMeanPrecision(3.0, 4.0), NormalWeightedMeanPrecision(5.0, 6.0))),) => MvNormalWeightedMeanPrecision([4.0, 6.0], [6.0 2.0; 2.0 8.0]),
            (m = (out = NormalMeanVariance(1.0, 2.0), in = (NormalMeanPrecision(3.0, 4.0), NormalWeightedMeanPrecision(5.0, 6.0))),) => MvNormalWeightedMeanPrecision([25 / 2, 11 / 2], [9 / 2 1 / 2; 1 / 2 13 / 2]),
            (m = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 1.0; 1.0 2.0]), in = (MvNormalMeanCovariance([2.0, 3.0], [3.0 1.0; 1.0 2.0]), MvNormalMeanCovariance([1.0, 2.0], [3.0 1.0; 1.0 2.0]))),) => MvNormalWeightedMeanPrecision([1 / 5, 12 / 5, 0.0, 2.0], [[0.8 -0.4 0.4 -0.2; -0.4 1.2 -0.2 0.6]; [0.4 -0.2 0.8 -0.4; -0.2 0.6 -0.4 1.2]]),
            (m = (out = MvNormalMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 2.0]), in = (MvNormalMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 2.0]), MvNormalMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 2.0]))),) => MvNormalWeightedMeanPrecision([10.0, 10.0, 10.0, 10.0], [6.0 2.0 3.0 1.0; 2.0 4.0 1.0 2.0; 3.0 1.0 6.0 2.0; 1.0 2.0 2.0 4.0]),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0]), in = (MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0]), MvNormalWeightedMeanPrecision([1.0, 2.0], [3.0 1.0; 1.0 1.0]))),) => MvNormalWeightedMeanPrecision([2.0, 4.0, 2.0, 4.0], [6.0 2.0 3.0 1.0; 2.0 2.0 1.0 1.0; 3.0 1.0 6.0 2.0; 1.0 1.0 2.0 2.0]),
            (m = (out = MvNormalMeanCovariance([1.0, 1.0], [3.0 1.0; 1.0 2.0]), in = (MvNormalMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 2.0]), MvNormalWeightedMeanPrecision([1.0, 1.0], [3.0 1.0; 1.0 2.0]))),) => MvNormalWeightedMeanPrecision([21 / 5, 17 / 5, 6 / 5, 7 / 5], [17 / 5 4 / 5 2 / 5 -1 / 5; 4 / 5 13 / 5 -1 / 5 3 / 5; 2 / 5 -1 / 5 17 / 5 4 / 5; -1 / 5 3 / 5 4 / 5 13 / 5]),
        ],
    )
end

@testitem "rules:+:weighted-mean inputs" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    I2 = [1.0 0.0; 0.0 1.0]
    # in[1] = out - in[2] for two weighted-mean messages: E[out] = (1, 2), E[in[2]] = (1, 1), so
    # E[in[1]] = E[out] - E[in[2]], not E[in[2]] - E[out].
    @test_message_update_rule(
        node = +, target = (:in, 1),
        cases = [(m = (out = MvNormalWeightedMeanPrecision([2.0, 4.0], 2 * I2), in = (nothing, MvNormalWeightedMeanPrecision([1.0, 1.0], I2))),) => MvNormalMeanCovariance([0.0, 1.0], 1.5 * I2)],
    )
    # Two Gammas of one scale convolve into their shapes' sum.
    @test_message_update_rule(
        node = +, target = :out,
        cases = [(m = (in = (Gamma(1.0, 2.0), Gamma(2.5, 2.0)),),) => Gamma(3.5, 2.0)],
    )
end

@testitem "rules:+:many inputs" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions

    # Towards `out`, the sum of every input; towards an input, `out` less the others.
    @test_message_update_rule(
        node = +, target = :out,
        cases = [
            (m = (in = (NormalMeanVariance(1.0, 1.0), PointMass(2.0), NormalMeanVariance(-1.0, 3.0)),),) => NormalMeanVariance(2.0, 4.0),
            (m = (in = (PointMass(1.0), PointMass(2.0), PointMass(3.0), PointMass(4.0)),),) => PointMass(10.0),
            (m = (in = (MvNormalMeanCovariance([1.0, 0.0], [2.0 0.0; 0.0 1.0]), PointMass([1.0, 1.0]), MvNormalMeanCovariance([0.0, 1.0], [1.0 0.5; 0.5 1.0])),),) =>
                MvNormalMeanCovariance([2.0, 2.0], [3.0 0.5; 0.5 2.0]),
        ],
    )
    @test_message_update_rule(
        node = +, target = (:in, 2),
        cases = [
            (m = (out = NormalMeanVariance(5.0, 1.0), in = (NormalMeanVariance(1.0, 1.0), nothing, NormalMeanVariance(-1.0, 3.0))),) => NormalMeanVariance(5.0, 5.0),
            (m = (out = PointMass(5.0), in = (NormalMeanVariance(1.0, 1.0), nothing, PointMass(2.0))),) => NormalMeanVariance(2.0, 1.0),
            (m = (out = PointMass(5.0), in = (PointMass(1.0), nothing, PointMass(2.0))),) => PointMass(2.0),
        ],
    )
    @test_message_update_rule(
        node = +, target = (:in, 3),
        cases = [(m = (out = NormalMeanVariance(0.0, 1.0), in = (PointMass(1.0), NormalMeanVariance(1.0, 2.0), nothing)),) => NormalMeanVariance(-2.0, 3.0)],
    )

    # With a normal `out`: the Gaussian inputs jointly, a point mass on its own. The joint over
    # three Gaussian inputs adds `out`'s precision to every entry.
    @test_marginal_update_rule(
        node = +, target = (:in,),
        cases = [
            (m = (out = NormalMeanPrecision(1.0, 2.0), in = (NormalMeanPrecision(1.0, 1.0), NormalMeanPrecision(0.0, 2.0), NormalMeanPrecision(-1.0, 3.0))),) =>
                MvNormalWeightedMeanPrecision([3.0, 2.0, -1.0], [3.0 2.0 2.0; 2.0 4.0 2.0; 2.0 2.0 5.0]),
            # `out`'s factor of in[1] + in[3] + 1: its weighted mean shifted by -W·1.
            (m = (out = NormalMeanPrecision(1.0, 2.0), in = (NormalMeanPrecision(1.0, 1.0), PointMass(1.0), NormalMeanPrecision(-1.0, 3.0))),) =>
                FactorizedCluster(((:in, 1), (:in, 3)) => MvNormalWeightedMeanPrecision([1.0, -3.0], [3.0 2.0; 2.0 5.0]), ((:in, 2),) => PointMass(1.0)),
            (m = (out = NormalMeanPrecision(1.0, 2.0), in = (PointMass(1.0), PointMass(2.0))),) => FactorizedCluster(((:in, 1),) => PointMass(1.0), ((:in, 2),) => PointMass(2.0)),
        ],
    )
end

@testitem "rules:+:inputs given their sum" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using StandardMessagePassingRules: InputsGivenSum
    using LinearAlgebra: det

    # With `out` known, every Gaussian input but the last is free, and the last is the rest of
    # the sum: N(x₁ | 2, 2) N(1 - x₁ | 0, 1) for x₁ + 2 + x₃ = 3.
    @test_marginal_update_rule(
        node = +, target = (:in,),
        cases = [
            (m = (out = PointMass(3.0), in = (NormalMeanVariance(2.0, 2.0), PointMass(2.0), NormalMeanVariance(0.0, 1.0))),) =>
                InputsGivenSum(NormalWeightedMeanPrecision(2.0, 1.5), (PointMass(2.0),), PointMass(3.0)),
            (m = (out = PointMass(3.0), in = (NormalMeanVariance(2.0, 2.0), PointMass(2.0))),) => InputsGivenSum(nothing, (PointMass(2.0),), PointMass(3.0)),
            (m = (out = PointMass(3.0), in = (PointMass(1.0), PointMass(2.0))),) => InputsGivenSum(nothing, (PointMass(1.0), PointMass(2.0)), PointMass(3.0)),
        ],
    )

    # The entropy: the free members' joint on the plane, with precision diag(1 ./ v[1:end-1]) plus
    # 1/v[end] in every entry, whose log-determinant is log Σv - Σ log v, and one point-mass
    # entropy for each known input and for the sum.
    v = [2.0, 0.5, 3.0]
    joint = getresult(call_marginal_update_rule(+, (:in,); m = (out = PointMass(4.0), in = (NormalMeanVariance(1.0, v[1]), NormalMeanVariance(0.0, v[2]), PointMass(1.0), NormalMeanVariance(2.0, v[3])))))
    free_entropy = (2 * (1 + log(2π)) - (log(sum(v)) - sum(log, v))) / 2
    @test entropy(joint) == entropy(joint.free) + 2 * entropy(PointMass(1.0))
    @test BayesBase.value(entropy(joint)) ≈ free_entropy
    @test BayesBase.value(entropy(joint)) ≈ entropy(joint.free)
    @test det(precision(joint.free)) ≈ sum(v) / prod(v)

    # Multivariate inputs: the same plane in blocks.
    Σ₁, Σ₂ = [2.0 0.5; 0.5 1.0], [1.0 0.0; 0.0 3.0]
    mv = getresult(call_marginal_update_rule(+, (:in,); m = (out = PointMass([1.0, 2.0]), in = (MvNormalMeanCovariance([0.0, 0.0], Σ₁), MvNormalMeanCovariance([1.0, 1.0], Σ₂)))))
    @test precision(mv.free) ≈ inv(Σ₁) + inv(Σ₂)
    @test weightedmean(mv.free) ≈ inv(Σ₂) * ([1.0, 2.0] - [1.0, 1.0])
end
