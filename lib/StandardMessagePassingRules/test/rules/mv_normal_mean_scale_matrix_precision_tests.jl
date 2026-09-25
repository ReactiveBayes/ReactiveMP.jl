# MvNormalMeanScaleMatrixPrecision, the multivariate normal with precision γG: tables of cases,
# and hand-derived cases for the marginal rules and the energy.

@testitem "rules:MvNormalMeanScaleMatrixPrecision:out-μ" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    G = Wishart(4, [13.0 14.0; 14.0 20.0])
    @test_message_update_rule(
        node = MvNormalMeanScaleMatrixPrecision, target = :out,
        cases = [
            (q = (μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = Gamma(1.0, 1.0), G = G),) => MvNormalMeanPrecision([2.0, 1.0], [52.0 56.0; 56.0 80.0]),
            (q = (μ = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = Gamma(3.0, 2.0), G = G),) => MvNormalMeanPrecision([2.0, 1.0], [312.0 336.0; 336.0 480.0]),
            (q = (μ = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = Gamma(4.0, 2.0), G = G),) =>
                MvNormalMeanPrecision([0.7500000000000003, -0.12500000000000017], [416.0 448.0; 448.0 640.0]),
            (m = (μ = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]),), q = (γ = Gamma(2.0, 1.0), G = G)) =>
                MvNormalMeanPrecision([2.0, 1.0], [0.4825426556235183 -0.23647165559425204; -0.23647165559425204 0.3643068278263923]),
            (m = (μ = MvNormalMeanCovariance([0.0, 1.0], [2.0 -1.0; -1.0 4.0]),), q = (γ = Gamma(3.0, 1.0), G = G)) =>
                MvNormalMeanPrecision([0.0, 1.0], [0.5656312548702044 0.14337881403841166; 0.14337881403841166 0.2852908371403689]),
            (m = (μ = MvNormalWeightedMeanPrecision([3.0, -1.0], [1.0 0.0; 0.0 1.0]),), q = (γ = Gamma(4.0, 2.0), G = G)) =>
                MvNormalMeanPrecision([3.0, -1.0], [0.9903743636718574 0.006727433814364876; 0.006727433814364877 0.9937380805790399]),
        ],
    )
    @test_message_update_rule(
        node = MvNormalMeanScaleMatrixPrecision, target = :μ,
        cases = [
            (q = (out = MvNormalMeanCovariance([2.0, 1.0], [3.0 2.0; 2.0 4.0]), γ = GammaShapeRate(1.0, 1.0), G = G),) => MvNormalMeanPrecision([2.0, 1.0], [52.0 56.0; 56.0 80.0]),
            (q = (out = MvNormalMeanPrecision([2.0, 3.0], [3.0 2.0; 2.0 4.0]), γ = Gamma(3.0, 1.0), G = G),) => MvNormalMeanPrecision([2.0, 3.0], [156.0 168.0; 168.0 240.0]),
            (m = (out = MvNormalMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]),), q = (γ = Gamma(1.0, 1.0), G = G)) =>
                MvNormalMeanPrecision([2.0, 1.0], [2.777070063694267 1.9872611464968155; 1.9872611464968155 3.770700636942675]),
            (m = (out = MvNormalMeanCovariance([0.0, 0.0], [7.0 -1.0; -1.0 9.0]),), q = (γ = GammaShapeRate(4.0, 2.0), G = G)) =>
                MvNormalMeanPrecision([0.0, 0.0], [0.1444643743381126 0.016444116187372262; 0.016444116187372262 0.1126703322039727]),
            (m = (out = MvNormalWeightedMeanPrecision([2.0, 1.0], [3.0 2.0; 2.0 4.0]),), q = (γ = GammaShapeRate(2.0, 1.0), G = G)) =>
                MvNormalMeanPrecision([0.7500000000000003, -0.12500000000000017], [2.8822495606326877 1.9964850615114234; 1.9964850615114234 3.8804920913884007]),
        ],
    )
end

@testitem "rules:MvNormalMeanScaleMatrixPrecision:γ-G" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, ExponentialFamily, BayesBase, Distributions

    G = Wishart(4, [13.0 14.0; 14.0 20.0])
    I4 = [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0]
    a = [1.0, 2.0, -1.0, -2.0]
    A = [3.5 -0.5 -0.25 0.0; -0.5 3.0 -0.25 0.0; -0.25 -0.25 6.0 0.25; 0.0 0.0 0.25 7.0]
    # A precision-parameterised input converted to Float32 moves the rate by about 1e-9.
    @test_message_update_rule(
        node = MvNormalMeanScaleMatrixPrecision, target = :γ, rtol = 1.0e-6,
        cases = [
            (q = (out = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanCovariance([3.0, 5.0], [3.0 2.0; 2.0 4.0]), G = G),) => GammaShapeRate(2.0, 1500.0),
            (q = (out = MvNormalMeanPrecision([1.0, 2.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanPrecision([3.0, 5.0], [3.0 2.0; 2.0 4.0]), G = G),) => GammaShapeRate(2.0, 828.0),
            (clusters = ((:out, :μ) => MvNormalMeanCovariance(ones(4), I4),), q = (G = G,)) => GammaShapeRate(2.0, 132.0),
            (clusters = ((:out, :μ) => MvNormalMeanPrecision(ones(4), I4),), q = (G = G,)) => GammaShapeRate(2.0, 132.0),
            (clusters = ((:out, :μ) => MvNormalWeightedMeanPrecision(ones(4), I4),), q = (G = G,)) => GammaShapeRate(2.0, 132.0),
            (clusters = ((:out, :μ) => MvNormalMeanCovariance(a, A),), q = (G = G,)) => GammaShapeRate(2.0, 1852.0),
            (clusters = ((:out, :μ) => MvNormalMeanPrecision(a, A),), q = (G = G,)) => GammaShapeRate(2.0, 1224.3440167026847),
            (clusters = ((:out, :μ) => MvNormalWeightedMeanPrecision(a, A),), q = (G = G,)) => GammaShapeRate(2.0, 106.333391526103),
        ],
    )
    @test_message_update_rule(
        node = MvNormalMeanScaleMatrixPrecision, target = :G,
        cases = [
            (q = (out = MvNormalMeanCovariance([1.0, 2.0], [1.0 2.0; 2.0 3.0]), μ = MvNormalMeanCovariance([1.0, 2.0], [3.0 2.0; 2.0 4.0]), γ = GammaShapeRate(1.0, 1.0)),) =>
                Wishart(4.0, [0.5833333333333333 -0.3333333333333333; -0.3333333333333333 0.3333333333333333]),
            (q = (out = MvNormalMeanCovariance([2.0, 5.0], [3.0 2.0; 2.0 4.0]), μ = MvNormalMeanCovariance([3.0, 5.0], [1.0 2.0; 2.0 3.0]), γ = GammaShapeRate(2.0, 4.0)),) =>
                Wishart(4.0, [0.736842105263158 -0.42105263157894735; -0.42105263157894735 0.5263157894736842]),
            (clusters = ((:out, :μ) => MvNormalWeightedMeanPrecision(ones(4), I4),), q = (γ = GammaShapeRate(1.0, 1.0),)) => Wishart(4.0, [0.5 0.0; 0.0 0.5]),
            (clusters = ((:out, :μ) => MvNormalWeightedMeanPrecision(2 * ones(4), I4),), q = (γ = GammaShapeRate(2.0, 4.0),)) => Wishart(4.0, [1.0 0.0; 0.0 1.0]),
            (clusters = ((:out, :μ) => MvNormalWeightedMeanPrecision(3 * ones(4), I4),), q = (γ = GammaShapeRate(22.0, 14.0),)) =>
                Wishart(4.0, [0.3181818181818182 0.0; 0.0 0.3181818181818182]),
        ],
    )
end

@testitem "rules:MvNormalMeanScaleMatrixPrecision:marginals-energy" tags = [:rules] begin
    using StandardMessagePassingRules, MessagePassingRulesTestUtils, MessagePassingRulesBase, ExponentialFamily, BayesBase, Distributions
    using StatsFuns: log2π
    using SpecialFunctions: digamma

    I2 = [1.0 0.0; 0.0 1.0]
    # Messages with precision I, and E[γ]·E[G] = (1/4)·2I = I/2 coupling them.
    joint = MvNormalWeightedMeanPrecision([1.0, 2.0, 3.0, 4.0], [1.5 0 -0.5 0; 0 1.5 0 -0.5; -0.5 0 1.5 0; 0 -0.5 0 1.5])
    γ, G = PointMass(0.25), PointMass(2 * I2)
    @test_marginal_update_rule(
        node = MvNormalMeanScaleMatrixPrecision, target = (:out, :μ),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (γ = γ, G = G)) => joint,
            (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2)), q = (γ = GammaShapeRate(1.0, 4.0), G = G)) =>
                FactorizedCluster((:out,) => PointMass([1.0, 1.0]), (:μ,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2)),
            (m = (out = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), μ = PointMass([1.0, 1.0])), q = (γ = γ, G = Wishart(4.0, I2 / 4))) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([3.25, 4.25], 1.25 * I2), (:μ,) => PointMass([1.0, 1.0])),
        ],
    )
    @test_marginal_update_rule(
        node = MvNormalMeanScaleMatrixPrecision, target = (:out, :μ, :γ, :G),
        cases = [
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), γ = γ, G = G),) =>
                FactorizedCluster((:out, :μ) => joint, (:γ,) => γ, (:G,) => G),
            (m = (out = MvNormalWeightedMeanPrecision([1.0, 2.0], I2), μ = PointMass([1.0, 1.0]), γ = γ, G = G),) =>
                FactorizedCluster((:out,) => MvNormalWeightedMeanPrecision([1.5, 2.5], 1.5 * I2), (:μ,) => PointMass([1.0, 1.0]), (:γ,) => γ, (:G,) => G),
            (m = (out = PointMass([1.0, 1.0]), μ = MvNormalWeightedMeanPrecision([3.0, 4.0], I2), γ = γ, G = G),) =>
                FactorizedCluster((:out,) => PointMass([1.0, 1.0]), (:μ,) => MvNormalWeightedMeanPrecision([3.5, 4.5], 1.5 * I2), (:γ,) => γ, (:G,) => G),
        ],
    )
    # (d log 2π - d E[log γ] - E[log |G|] + E[γ] tr(E[G] S)) / 2, with γ ~ GammaShapeRate(2, 4):
    # E[γ] = 1/2, E[log γ] = digamma(2) - log(4); G ~ Wishart(3, I): E[G] = 3I and
    # E[log |G|] = digamma(3/2) + digamma(1) + 2 log 2; S = I + I + ΔΔᵀ for Δ = (0.5, -0.5).
    q = (out = MvNormalMeanCovariance([1.5, 0.5], I2), μ = MvNormalMeanCovariance([1.0, 1.0], I2), γ = GammaShapeRate(2.0, 4.0), G = Wishart(3.0, I2))
    ElogG = digamma(1.5) + digamma(1.0) + 2 * log(2.0)
    @test getresult(call_average_energy(MvNormalMeanScaleMatrixPrecision; q)) ≈ (2 * log2π - 2 * (digamma(2.0) - log(4.0)) - ElogG + 0.5 * 3 * (4.0 + 0.5)) / 2
    # The joint with S = 2I, γ = 1/2 and G = 2I: E[γ] tr(E[G] S) = 4.
    whole = MvNormalMeanCovariance([1.0, 1.0, 1.0, 1.0], [1.0 0 0 0; 0 1.0 0 0; 0 0 1.0 0; 0 0 0 1.0])
    @test getresult(call_average_energy(MvNormalMeanScaleMatrixPrecision; q = (γ = PointMass(0.5), G = PointMass(2 * I2)), clusters = ((:out, :μ) => whole,))) ≈
        (2 * log2π - 2 * log(0.5) - log(4.0) + 4.0) / 2
end
