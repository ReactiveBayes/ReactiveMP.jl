module RulesNormalMixtureSwitchTest

using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

import ReactiveMP: @test_rules
import ExponentialFamily: WishartFast

@testitem "rules:NormalMixture:switch" begin
    @testset "Variational : (m_out::UnivariateNormalDistributionsFamily..., m_μ::UnivariateNormalDistributionsFamily...) k=1" begin
        @test_rules [check_type_promotion = true] NormalMixture{2}(:switch, Marginalisation) [(
            input = (
                q_out = NormalMeanVariance(8.5, 0.5),
                q_m = ManyOf(NormalMeanVariance(5.0, 2.0), NormalMeanVariance(10.0, 3.0)),
                q_p = ManyOf(GammaShapeRate(1.0, 2.0), GammaShapeRate(2.0, 1.0))
            ),
            output = Categorical([0.7713458788198754, 0.22865412118012463])
        )]
    end

    @testset "Variational : (m_out::MultivariateNormalDistributionsFamily..., m_μ::MultivariateNormalDistributionsFamily...) k=1" begin
        @test_rules [check_type_promotion = true, atol = 1e-4] NormalMixture{2}(:switch, Marginalisation) [(
            input = (
                q_out = MvNormalMeanCovariance([8.5], [0.5]),
                q_m = ManyOf(MvNormalMeanCovariance([5.0], [2.0]), MvNormalMeanCovariance([10.0], [3.0])),
                q_p = ManyOf(Wishart(2.0, fill(0.25, 1, 1)), Wishart(4.0, fill(0.5, 1, 1)))
            ),
            output = Categorical([0.7713458788198754, 0.22865412118012463])
        )]
    end
end

end
