module RulesNormalMixturePTest

using Test
using ReactiveMP
using Random
using Distributions

import ReactiveMP: @test_rules
import ReactiveMP: WishartMessage

@testset "rules:NormalMixture:p" begin
    @testset "Variational : (m_out::UnivariateNormalDistributionsFamily..., m_μ::GammaDistributionsFamily...) k=1" begin
        @test_rules [with_float_conversions = true] NormalMixture{2}((:p, k=1), Marginalisation) [
            (
                input = (
                    q_out = NormalMeanVariance(8.5, 0.5),
                    q_switch = Bernoulli(0.2),
                    q_m =  NormalMeanVariance(5.0, 2.0)
                ),
                output =  GammaShapeRate(1.1, 1.475)
            ),
            (
                input = (
                    q_out = NormalMeanVariance(-3, 2.0),
                    q_switch = Bernoulli(0.5),
                    q_m =  NormalMeanVariance(5.0, 2.0)
                ),
                output =  GammaShapeRate(1.25, 17.0)
            ),
        ]
    end
   
    @testset "Variational : (m_out::MultivariateNormalDistributionsFamily..., m_μ::MultivariateNormalDistributionsFamily...) k=1" begin
        @test_rules [with_float_conversions = true, atol=1e-4] NormalMixture{2}((:p, k=1), Marginalisation) [
            (
                input = (
                    q_out = MvNormalMeanPrecision([8.5], [0.5]),
                    q_switch = Bernoulli(0.2),
                    q_m =  MvNormalMeanPrecision([3.0], [0.1])
                ),
                output =  WishartMessage(2.2, [8.45;;])
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([8.5, 5.1], [0.5 0.1; 0.1 4]),
                    q_switch = Bernoulli(0.2),
                    q_m =  MvNormalMeanPrecision([3.0, 10], [0.1 0.2; 0.2 -0.3])
                ),
                output =  WishartMessage(3.2, [9.59487 -5.97148; -5.97148 5.13797])
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([5.0, 8.0], [3 0.5; 0.5 -6]),
                    q_switch = Categorical([0.25, 0.75]),
                    q_m =  MvNormalMeanPrecision([2.0, -3.0], [2.1 -1.0; -1.0 3.0])
                ),
                output =  WishartMessage(3.25, [2.47598 8.29032; 8.29032 30.3902])
            ),
            (
                input = (
                    q_out = MvNormalMeanCovariance([-3], [2.0]),
                    q_switch = Bernoulli(0.5),
                    q_m =  MvNormalMeanCovariance([5.0], [2.0])
                ),
                output =  WishartMessage(2.5, [34.0;;])
            ),
        ]
    end

end

end
