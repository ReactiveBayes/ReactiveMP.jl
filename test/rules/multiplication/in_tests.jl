
@testitem "rule:typeof(*):in" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions, StableRNGs, LinearAlgebra

    import ReactiveMP: @test_rules, make_inversedist_message

    @testset "Belief Propagation: (m_A::PointMass{<:Real}, m_out::MultivariateNormalDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] (*)(:in, Marginalisation) [
            (input = (m_A = PointMass(2), m_out = MvNormalMeanCovariance([2, 4], [12 8; 8 24])), output = MvNormalMeanCovariance([1, 2], [3 2; 2 6])),
            (input = (m_A = PointMass(0.5), m_out = MvNormalMeanPrecision([1, 2], [12 8; 8 24])), output = MvNormalMeanPrecision([2, 4], [3 2; 2 6])),
            (input = (m_A = PointMass(0.5), m_out = MvNormalWeightedMeanPrecision([2, 4], [12 8; 8 24])), output = MvNormalWeightedMeanPrecision([1, 2], [3 2; 2 6]))
        ]
    end
    @testset "Belief Propagation: (m_A::PointMass{<:UniformScaling}, m_out::MultivariateNormalDistributionsFamily)" begin
        @test_rules [check_type_promotion = true] (*)(:in, Marginalisation) [
            (input = (m_A = PointMass(2I), m_out = MvNormalMeanCovariance([2, 4], [12 8; 8 24])), output = MvNormalMeanCovariance([1, 2], [3 2; 2 6])),
            (input = (m_A = PointMass(0.5I), m_out = MvNormalMeanPrecision([1, 2], [12 8; 8 24])), output = MvNormalMeanPrecision([2, 4], [3 2; 2 6])),
            (input = (m_A = PointMass(0.5I), m_out = MvNormalWeightedMeanPrecision([2, 4], [12 8; 8 24])), output = MvNormalWeightedMeanPrecision([1, 2], [3 2; 2 6]))
        ]
    end
    @testset "Univariate Gaussian messages" begin
        rng = StableRNG(42)
        d1 = NormalMeanVariance(0.0, 1.0)
        d2 = NormalMeanVariance(0.5, 1.5)
        d3 = NormalMeanVariance(2.0, 0.5)
        OutMessage_1 = @call_rule typeof(*)(:in, Marginalisation) (m_out = d1, m_A = d2)
        OutMessage_2 = @call_rule typeof(*)(:in, Marginalisation) (m_out = d1, m_A = d3)
        OutMessage_3 = @call_rule typeof(*)(:in, Marginalisation) (m_out = d2, m_A = d3)
        groundtruthOutMessage_1 = (x) -> -log(abs(x)) - 0.5 * log(2π * (var(d2) + var(d1) / x^2)) - 1 / 2 * (mean(d1) - x * mean(d2))^2 / (var(d2) * x^2 + var(d1))
        groundtruthOutMessage_2 = (x) -> -log(abs(x)) - 0.5 * log(2π * (var(d3) + var(d1) / x^2)) - 1 / 2 * (mean(d1) - x * mean(d3))^2 / (var(d3) * x^2 + var(d1))
        groundtruthOutMessage_3 = (x) -> -log(abs(x)) - 0.5 * log(2π * (var(d3) + var(d2) / x^2)) - 1 / 2 * (mean(d2) - x * mean(d3))^2 / (var(d3) * x^2 + var(d2))

        @test typeof(OutMessage_1) <: ContinuousUnivariateLogPdf
        @test typeof(OutMessage_2) <: ContinuousUnivariateLogPdf
        @test typeof(OutMessage_3) <: ContinuousUnivariateLogPdf

        samples = rand(rng, Uniform(0.5, 4), 10)
        for i in samples
            @test OutMessage_1(i) ≈ groundtruthOutMessage_1(i)
            @test OutMessage_2(i) ≈ groundtruthOutMessage_2(i)
            @test OutMessage_3(i) ≈ groundtruthOutMessage_3(i)
        end
    end
    @testset "messages of type Any" begin
        rng         = StableRNG(42)
        d1          = LogNormal(1.5, 1)
        d2          = NormalMeanVariance(0.0, 1.0)
        num_samples = 3000
        samples_d2  = rand(rng, d2, num_samples)

        OutMessage = @call_rule typeof(*)(:in, Marginalisation) (m_out = d1, m_A = d2)

        @test typeof(OutMessage) <: ContinuousUnivariateLogPdf

        groundtruthOutMessage = make_inversedist_message(samples_d2, d1)
        @test isapprox(OutMessage(1.0), groundtruthOutMessage(1.0); atol = 0.1)
        @test isapprox(OutMessage(2.0), groundtruthOutMessage(2.0); atol = 0.1)
        @test isapprox(OutMessage(3.1), groundtruthOutMessage(3.1); atol = 0.1)
    end
end
