module RulesNormalMixtureMTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules

@testset "rules:NormalMixture:m" begin
    @testset "Variational : (m_out::UnivariateNormalDistributionsFamily..., m_p::GammaDistributionsFamily...) k=1" begin
        @test_rules [with_float_conversions = true] NormalMixture{2}((:m, k = 1), Marginalisation) [
            (input = (q_out = NormalMeanVariance(8.5, 0.5), q_switch = Bernoulli(0.2), q_p = GammaShapeRate(1.0, 2.0)), output = NormalMeanPrecision(8.5, 0.1)),
            (
                input = (q_out = NormalWeightedMeanPrecision(3 / 10, 6 / 10), q_switch = Categorical([0.5, 0.5]), q_p = GammaShapeRate(1.0, 1.0)),
                output = NormalMeanPrecision(0.5, 0.5)
            ),
            (
                input = (q_out = NormalWeightedMeanPrecision(5.0, 1 / 4), q_switch = Categorical([0.75, 0.25]), q_p = GammaShapeScale(1.0, 1.0)),
                output = NormalMeanPrecision(20.0, 0.75)
            ),
            (input = (q_out = NormalWeightedMeanPrecision(1, 1), q_switch = Categorical([1.0, 0.0]), q_p = GammaShapeRate(1.0, 2.0)), output = NormalMeanPrecision(1.0, 0.5))
        ]
    end

    @testset "Variational : (m_out::UnivariateNormalDistributionsFamily..., m_p::GammaDistributionsFamily...) k=2" begin
        @test_rules [with_float_conversions = true] NormalMixture{2}((:m, k = 2), Marginalisation) [
            (input = (q_out = NormalMeanVariance(8.5, 0.5), q_switch = Bernoulli(0.2), q_p = GammaShapeRate(1.0, 2.0)), output = NormalMeanPrecision(8.5, 0.4)),
            (
                input = (q_out = NormalWeightedMeanPrecision(3 / 10, 6 / 10), q_switch = Categorical([0.5, 0.5]), q_p = GammaShapeRate(1.0, 1.0)),
                output = NormalMeanPrecision(0.5, 0.5)
            ),
            (
                input = (q_out = NormalWeightedMeanPrecision(5.0, 1 / 4), q_switch = Categorical([0.75, 0.25]), q_p = GammaShapeScale(1.0, 1.0)),
                output = NormalMeanPrecision(20.0, 0.25)
            )
        ]
    end

    @testset "Variational : (m_out::MultivariateNormalDistributionsFamily..., m_p::Wishart...) k=1" begin
        @test_rules [with_float_conversions = true, atol = 1e-4] NormalMixture{2}((:m, k = 1), Marginalisation) [
            (
                input = (
                    q_out = MvNormalWeightedMeanPrecision([6.75, 12.0], [4.5 -0.75; -0.75 4.5]), q_switch = Categorical([0.5, 0.5]), q_p = Wishart(3.0, [2.0 -0.25; -0.25 1.0])
                ),
                output = MvNormalMeanPrecision([2.0, 3.0], [3.0 -0.375; -0.375 1.5])
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([3.75, 10.3125], [5.25 -0.75; -0.75 3.75]), q_switch = Categorical([0.75, 0.25]), q_p = Wishart(3.0, [2.0 -0.25; -0.25 1.0])
                ),
                output = MvNormalMeanPrecision([3.75, 10.3125], [4.5 -0.5625; -0.5625 2.25])
            ),
            (
                input = (q_out = MvNormalMeanPrecision([0.75, 17.25], [3.0 -0.75; -0.75 6.0]), q_switch = Categorical([1.0, 0.0]), q_p = Wishart(3.0, [2.0 -0.25; -0.25 1.0])),
                output = MvNormalMeanPrecision([0.75, 17.25], [6.0 -0.75; -0.75 3.0])
            )
        ]
    end

    @testset "Variational : (m_out::UnivariateNormalDistributionsFamily..., m_p::GammaDistributionsFamily...) k=1" begin
        @test_rules [with_float_conversions = true] NormalMixture{2}((:m, k = 1), Marginalisation) [
            (input = (q_out = PointMass(8.5), q_switch = Bernoulli(0.2), q_p = GammaShapeRate(1.0, 2.0)), output = NormalMeanPrecision(8.5, 0.1)),
            (input = (q_out = NormalWeightedMeanPrecision(3 / 10, 6 / 10), q_switch = Categorical([0.5, 0.5]), q_p = PointMass(1.0)), output = NormalMeanPrecision(0.5, 0.5))
        ]
    end
end

end
