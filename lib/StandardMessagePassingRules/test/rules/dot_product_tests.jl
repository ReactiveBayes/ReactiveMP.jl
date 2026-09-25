# `dot`: tables of cases, with and without a `ctx.matrix_correction`; the default correction
# for an unset one; and the SoftDot hint for two Gaussian inputs.

@testitem "rules:dot:out" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using MessagePassingRulesBase: RuleContext
    using MatrixCorrectionTools: NoCorrection, ReplaceZeroDiagonalEntries, AddToDiagonalEntries
    using BayesBase: tiny

    @test_message_update_rule(
        node = dot, target = :out,
        cases = [
            (m = (in1 = PointMass(1.0), in2 = NormalMeanVariance(2.0, 2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(2.0, 2.0),
            (m = (in1 = PointMass(-1.0), in2 = NormalMeanPrecision(3.0, 1.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(-3.0, 1.0),
            (m = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(2.0, 0.5)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(8.0, 8.0),
            (m = (in1 = PointMass(1.0), in2 = NormalMeanVariance(2.0, 2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(2.0, 2.0),
            (m = (in1 = PointMass(-1.0), in2 = NormalMeanPrecision(3.0, 1.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(-3.0, 1.0),
            (m = (in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(2.0, 0.5)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(8.0, 8.0),
            (m = (in1 = NormalMeanVariance(2.0, 2.0), in2 = PointMass(4.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(8.0, 32.0),
            (m = (in1 = NormalMeanPrecision(2.0, inv(3.0)), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(4.0, 12.0),
            (m = (in1 = NormalWeightedMeanPrecision(2.0, 0.5), in2 = PointMass(1.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(4.0, 2.0),
            (m = (in1 = NormalMeanVariance(2.0, 2.0), in2 = PointMass(4.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(8.0, 32.0),
            (m = (in1 = NormalMeanPrecision(2.0, inv(3.0)), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(4.0, 12.0),
            (m = (in1 = NormalWeightedMeanPrecision(2.0, 0.5), in2 = PointMass(1.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(4.0, 2.0),
            (m = (in1 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0]), in2 = PointMass([4.0, 1.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(-3, 28),
            (m = (in1 = MvNormalMeanPrecision([2.0, 1.0], [2.0 -0.5; -0.5 5.0]), in2 = PointMass([2.0, 2.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(6.0, 128 / 39),
            (m = (in1 = MvNormalWeightedMeanPrecision([3.0, 2.0], [10.0 1.0; 1.0 20.0]), in2 = PointMass([-1.0, 3.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(-7 / 199, 116 / 199),
            (m = (in1 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0]), in2 = PointMass([4.0, 1.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(-3, 28),
            (m = (in1 = MvNormalMeanPrecision([2.0, 1.0], [2.0 -0.5; -0.5 5.0]), in2 = PointMass([2.0, 2.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(6.0, 128 / 39),
            (m = (in1 = MvNormalWeightedMeanPrecision([3.0, 2.0], [10.0 1.0; 1.0 20.0]), in2 = PointMass([-1.0, 3.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(-7 / 199, 116 / 199),
            (m = (in1 = PointMass([4.0, 1.0]), in2 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(-3, 28),
            (m = (in1 = PointMass([2.0, 2.0]), in2 = MvNormalMeanPrecision([2.0, 1.0], [2.0 -0.5; -0.5 5.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(6.0, 128 / 39),
            (m = (in1 = PointMass([-1.0, 3.0]), in2 = MvNormalWeightedMeanPrecision([3.0, 2.0], [10.0 1.0; 1.0 20.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalMeanVariance(-7 / 199, 116 / 199),
            (m = (in1 = PointMass([4.0, 1.0]), in2 = MvNormalMeanCovariance([-1.0, 1.0], [2.0 -1.0; -1.0 4.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(-3, 28),
            (m = (in1 = PointMass([2.0, 2.0]), in2 = MvNormalMeanPrecision([2.0, 1.0], [2.0 -0.5; -0.5 5.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(6.0, 128 / 39),
            (m = (in1 = PointMass([-1.0, 3.0]), in2 = MvNormalWeightedMeanPrecision([3.0, 2.0], [10.0 1.0; 1.0 20.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalMeanVariance(-7 / 199, 116 / 199),
        ],
    )
    @test_throws r"Please use SoftDot instead" getresult(call_message_update_rule(dot, :out; m = (in1 = NormalMeanVariance(1.0, 1.0), in2 = NormalMeanVariance(1.0, 1.0))))
end

@testitem "rules:dot:in1-in2" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using MessagePassingRulesBase: RuleContext
    using MatrixCorrectionTools: NoCorrection, ReplaceZeroDiagonalEntries, AddToDiagonalEntries
    using BayesBase: tiny

    @test_message_update_rule(
        node = dot, target = :in1,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 2.0), in2 = PointMass(-1.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(-1.0, 0.5),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in2 = PointMass(-2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(-2.0, 4.0),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in2 = PointMass(-1.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(-2.0, 1.0),
            (m = (out = NormalMeanVariance(2.0, 2.0), in2 = PointMass(-1.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(-1.0, 0.5),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in2 = PointMass(-2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(-2.0, 4.0),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in2 = PointMass(-1.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(-2.0, 1.0),
            (m = (out = NormalMeanVariance(2.0, 2.0), in2 = PointMass(0.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(0.0, 0.0),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in2 = PointMass(0.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(0.0, 0.0),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in2 = PointMass(0.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(0.0, 0.0),
            (m = (out = NormalMeanVariance(2.0, 2.0), in2 = PointMass(0.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(0.0, tiny),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in2 = PointMass(0.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(0.0, tiny),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in2 = PointMass(0.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(0.0, tiny),
            (m = (out = NormalMeanVariance(2.0, 1.0), in2 = PointMass([-1.0, 2.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => MvNormalWeightedMeanPrecision([-2.0, 4.0], [1.0 -2.0; -2.0 4.0]),
            (m = (out = NormalMeanPrecision(1.0, inv(2.0)), in2 = PointMass([1.0, 1.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => MvNormalWeightedMeanPrecision([0.5, 0.5], [0.5 0.5; 0.5 0.5]),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in2 = PointMass([-2.0, 3.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => MvNormalWeightedMeanPrecision([-4.0, 6.0], [4.0 -6.0; -6.0 9.0]),
            (m = (out = NormalMeanVariance(2.0, 1.0), in2 = PointMass([-1.0, 2.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([-2.0, 4.0], [1.0 -2.0; -2.0 4.0]),
            (m = (out = NormalMeanPrecision(1.0, inv(2.0)), in2 = PointMass([1.0, 1.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([0.5, 0.5], [0.5 0.5; 0.5 0.5]),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in2 = PointMass([-2.0, 3.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([-4.0, 6.0], [4.0 -6.0; -6.0 9.0]),
            (m = (out = NormalMeanVariance(2.0, 1.0), in2 = PointMass([-1.0, 2.0])), ctx = RuleContext(matrix_correction = AddToDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([-2.0, 4.0], [1.0 + tiny -2.0; -2.0 4.0 + tiny]),
            (m = (out = NormalMeanPrecision(1.0, inv(2.0)), in2 = PointMass([1.0, 1.0])), ctx = RuleContext(matrix_correction = AddToDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([0.5, 0.5], [0.5 + tiny 0.5; 0.5 0.5 + tiny]),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in2 = PointMass([-2.0, 3.0])), ctx = RuleContext(matrix_correction = AddToDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([-4.0, 6.0], [4.0 + tiny -6.0; -6.0 9.0 + tiny]),
        ],
    )
    @test_message_update_rule(
        node = dot, target = :in2,
        cases = [
            (m = (out = NormalMeanVariance(2.0, 2.0), in1 = PointMass(-1.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(-1.0, 0.5),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in1 = PointMass(-2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(-2.0, 4.0),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in1 = PointMass(-1.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(-2.0, 1.0),
            (m = (out = NormalMeanVariance(2.0, 2.0), in1 = PointMass(-1.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(-1.0, 0.5),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in1 = PointMass(-2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(-2.0, 4.0),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in1 = PointMass(-1.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(-2.0, 1.0),
            (m = (out = NormalMeanVariance(2.0, 2.0), in1 = PointMass(0.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(0.0, 0.0),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in1 = PointMass(0.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(0.0, 0.0),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in1 = PointMass(0.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => NormalWeightedMeanPrecision(0.0, 0.0),
            (m = (out = NormalMeanVariance(2.0, 2.0), in1 = PointMass(0.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(0.0, tiny),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in1 = PointMass(0.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(0.0, tiny),
            (m = (out = NormalWeightedMeanPrecision(2.0, 1.0), in1 = PointMass(0.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => NormalWeightedMeanPrecision(0.0, tiny),
            (m = (out = NormalMeanVariance(2.0, 2.0), in1 = PointMass([-1.0, 1.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => MvNormalWeightedMeanPrecision([-1.0, 1.0], [0.5 -0.5; -0.5 0.5]),
            (m = (out = NormalMeanPrecision(1.0, inv(2.0)), in1 = PointMass([2.0, 1.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => MvNormalWeightedMeanPrecision([1.0, 0.5], [2.0 1.0; 1.0 0.5]),
            (m = (out = NormalWeightedMeanPrecision(1.0, 1.0), in1 = PointMass([-1.0, 3.0])), ctx = RuleContext(matrix_correction = NoCorrection())) => MvNormalWeightedMeanPrecision([-1.0, 3.0], [1.0 -3.0; -3.0 9.0]),
            (m = (out = NormalMeanVariance(2.0, 2.0), in1 = PointMass([-1.0, 1.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([-1.0, 1.0], [0.5 -0.5; -0.5 0.5]),
            (m = (out = NormalMeanPrecision(1.0, inv(2.0)), in1 = PointMass([2.0, 1.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([1.0, 0.5], [2.0 1.0; 1.0 0.5]),
            (m = (out = NormalWeightedMeanPrecision(1.0, 1.0), in1 = PointMass([-1.0, 3.0])), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([-1.0, 3.0], [1.0 -3.0; -3.0 9.0]),
            (m = (out = NormalMeanVariance(2.0, 2.0), in1 = PointMass([-1.0, 1.0])), ctx = RuleContext(matrix_correction = AddToDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([-1.0, 1.0], [0.5 + tiny -0.5; -0.5 0.5 + tiny]),
            (m = (out = NormalMeanPrecision(1.0, inv(2.0)), in1 = PointMass([2.0, 1.0])), ctx = RuleContext(matrix_correction = AddToDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([1.0, 0.5], [2.0 + tiny 1.0; 1.0 0.5 + tiny]),
            (m = (out = NormalWeightedMeanPrecision(1.0, 1.0), in1 = PointMass([-1.0, 3.0])), ctx = RuleContext(matrix_correction = AddToDiagonalEntries(tiny))) => MvNormalWeightedMeanPrecision([-1.0, 3.0], [1.0 + tiny -3.0; -3.0 9.0 + tiny]),
        ],
    )
    # An unset correction is dot's default, ReplaceZeroDiagonalEntries(tiny): the rank-one
    # precision a w aᵀ for a = (1, 0) has a zero on its diagonal, which NoCorrection keeps.
    m = (out = NormalMeanPrecision(2.0, 3.0), in1 = PointMass([1.0, 0.0]))
    @test_message_update_rule(
        node = dot, target = :in2,
        cases = [
            (m = m,) => MvNormalWeightedMeanPrecision([6.0, 0.0], [3.0 0.0; 0.0 tiny]),
            (m = m, ctx = RuleContext(matrix_correction = NoCorrection())) => MvNormalWeightedMeanPrecision([6.0, 0.0], [3.0 0.0; 0.0 0.0]),
        ],
        check_type_promotion = false,
    )
    @test_throws r"Please use SoftDot instead" getresult(call_message_update_rule(dot, :in2; m = (out = NormalMeanVariance(1.0, 1.0), in1 = NormalMeanVariance(1.0, 1.0))))
    @test_throws r"Please use SoftDot instead" getresult(call_message_update_rule(dot, :in1; m = (out = NormalMeanVariance(1.0, 1.0), in2 = NormalMeanVariance(1.0, 1.0))))
end

@testitem "rules:dot:marginals" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using MessagePassingRulesBase: RuleContext
    using MatrixCorrectionTools: NoCorrection, ReplaceZeroDiagonalEntries, AddToDiagonalEntries
    using BayesBase: tiny

    @test_marginal_update_rule(
        node = dot, target = (:in1, :in2),
        cases = [
            (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass(1.0), in2 = NormalMeanVariance(2.0, 2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(1.0), (:in2,) => NormalWeightedMeanPrecision(1.5, 1.0)),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(2.5, 4.5)),
            (m = (out = NormalWeightedMeanPrecision(0.5, 0.5), in1 = PointMass(-2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(-2.0), (:in2,) => NormalWeightedMeanPrecision(-0.5, 2.5)),
            (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass(1.0), in2 = NormalMeanVariance(2.0, 2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(1.0), (:in2,) => NormalWeightedMeanPrecision(1.5, 1.0)),
            (m = (out = NormalMeanPrecision(1.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(2.5, 4.5)),
            (m = (out = NormalWeightedMeanPrecision(0.5, 0.5), in1 = PointMass(-2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(-2.0), (:in2,) => NormalWeightedMeanPrecision(-0.5, 2.5)),
            (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass(1.0), in2 = NormalMeanVariance(2.0, 2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(1.0), (:in2,) => NormalWeightedMeanPrecision(1.5, 1.0)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanPrecision(1.0, 0.5)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(2.5, 4.5)),
            (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass(-2.0), in2 = NormalWeightedMeanPrecision(0.5, 0.5)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(-2.0), (:in2,) => NormalWeightedMeanPrecision(-0.5, 2.5)),
            (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass(1.0), in2 = NormalMeanVariance(2.0, 2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(1.0), (:in2,) => NormalWeightedMeanPrecision(1.5, 1.0)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanPrecision(1.0, 0.5)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(2.5, 4.5)),
            (m = (out = NormalMeanVariance(1.0, 2.0), in1 = PointMass(-2.0), in2 = NormalWeightedMeanPrecision(0.5, 0.5)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(-2.0), (:in2,) => NormalWeightedMeanPrecision(-0.5, 2.5)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = NormalMeanVariance(1.0, 2.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(2.5, 4.5), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanPrecision(3.0, 1.0), in1 = NormalMeanVariance(1.0, 2.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(6.5, 4.5), (:in2,) => PointMass(2.0)),
            (m = (out = NormalWeightedMeanPrecision(4.0, 1.0), in1 = NormalMeanVariance(1.0, 3.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(25 / 3, 13 / 3), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = NormalMeanVariance(1.0, 2.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(2.5, 4.5), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanPrecision(3.0, 1.0), in1 = NormalMeanVariance(1.0, 2.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(6.5, 4.5), (:in2,) => PointMass(2.0)),
            (m = (out = NormalWeightedMeanPrecision(4.0, 1.0), in1 = NormalMeanVariance(1.0, 3.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(25 / 3, 13 / 3), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = NormalMeanVariance(1.0, 2.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(2.5, 4.5), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(3.0, 1.0), in1 = NormalMeanPrecision(1.0, 0.5), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(6.5, 4.5), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(4.0, 1.0), in1 = NormalWeightedMeanPrecision(1.0 / 3.0, 1.0 / 3.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(25 / 3, 13 / 3), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = NormalMeanVariance(1.0, 2.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(2.5, 4.5), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(3.0, 1.0), in1 = NormalMeanPrecision(1.0, 0.5), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(6.5, 4.5), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(4.0, 1.0), in1 = NormalWeightedMeanPrecision(1.0 / 3.0, 1.0 / 3.0), in2 = PointMass(2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => NormalWeightedMeanPrecision(25 / 3, 13 / 3), (:in2,) => PointMass(2.0)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(2.5, 4.5)),
            (m = (out = NormalMeanPrecision(3.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(6.5, 4.5)),
            (m = (out = NormalWeightedMeanPrecision(4.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 3.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(25 / 3, 13 / 3)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(2.5, 4.5)),
            (m = (out = NormalMeanPrecision(3.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(6.5, 4.5)),
            (m = (out = NormalWeightedMeanPrecision(4.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 3.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(25 / 3, 13 / 3)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(2.5, 4.5)),
            (m = (out = NormalMeanVariance(3.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanPrecision(1.0, 0.5)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(6.5, 4.5)),
            (m = (out = NormalMeanVariance(4.0, 1.0), in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(1.0 / 3.0, 1.0 / 3.0)), ctx = RuleContext(matrix_correction = NoCorrection())) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(25 / 3, 13 / 3)),
            (m = (out = NormalMeanVariance(1.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanVariance(1.0, 2.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(2.5, 4.5)),
            (m = (out = NormalMeanVariance(3.0, 1.0), in1 = PointMass(2.0), in2 = NormalMeanPrecision(1.0, 0.5)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(6.5, 4.5)),
            (m = (out = NormalMeanVariance(4.0, 1.0), in1 = PointMass(2.0), in2 = NormalWeightedMeanPrecision(1.0 / 3.0, 1.0 / 3.0)), ctx = RuleContext(matrix_correction = ReplaceZeroDiagonalEntries(tiny))) => FactorizedCluster((:in1,) => PointMass(2.0), (:in2,) => NormalWeightedMeanPrecision(25 / 3, 13 / 3)),
        ],
    )
end

@testitem "rules:dot:precision is symmetric" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions, LinearAlgebra
    using MatrixCorrectionTools: NoCorrection
    using MessagePassingRulesBase: RuleContext

    # The precision towards `in2` is `a aᵀ w`: exactly symmetric, where `(a w) aᵀ` is not
    # always, which FastCholesky then warns about and symmetrises.
    a = [0.1, 0.7, 1.0 / 3.0, 2.0 / 7.0]
    m = getresult(call_message_update_rule(dot, :in2; m = (out = NormalMeanVariance(0.3, 0.9), in1 = PointMass(a)), ctx = RuleContext(matrix_correction = NoCorrection())))
    W = precision(m)
    @test W == a * a' * (1 / 0.9) && issymmetric(W)
    @test StandardMessagePassingRules.v_a_vT(a, 2.0) == a * a' * 2.0
end
