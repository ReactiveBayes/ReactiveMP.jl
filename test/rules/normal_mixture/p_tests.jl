
@testitem "rules:NormalMixture:p" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions

    import ReactiveMP: @test_rules
    import ExponentialFamily: WishartFast

    @testset "Variational : (m_out::UnivariateNormalDistributionsFamily..., m_μ::UnivariateNormalDistributionsFamily...) k=1" begin
        @test_rules [check_type_promotion = true] NormalMixture{2}((:p, k = 1), Marginalisation) [
            (input = (q_out = NormalMeanVariance(8.5, 0.5), q_switch = Bernoulli(0.8), q_m = NormalMeanVariance(5.0, 2.0)), output = GammaShapeRate(1.1, 1.475)),
            (input = (q_out = NormalMeanVariance(-3, 2.0), q_switch = Bernoulli(0.5), q_m = NormalMeanVariance(5.0, 2.0)), output = GammaShapeRate(1.25, 17.0))
        ]
    end

    @testset "Variational : (m_out::MultivariateNormalDistributionsFamily..., m_μ::MultivariateNormalDistributionsFamily...) k=1" begin
        @test_rules [check_type_promotion = true, atol = 1e-4] NormalMixture{2}((:p, k = 1), Marginalisation) [
            (
                input = (q_out = MvNormalMeanPrecision([8.5], [0.5]), q_switch = Bernoulli(0.8), q_m = MvNormalMeanPrecision([3.0], [0.1])),
                output = WishartFast(2.2, fill(8.45, 1, 1))
            ),
            (
                input = (q_out = MvNormalMeanPrecision([8.5, 5.1], [0.5 0.1; 0.1 4]), q_switch = Bernoulli(0.8), q_m = MvNormalMeanPrecision([3.0, 10], [0.1 0.2; 0.2 -0.3])),
                output = WishartFast(3.2, [7.181260 -5.553096; -5.553096 5.094238])
            ),
            (
                input = (
                    q_out = MvNormalMeanPrecision([5.0, 8.0], [3 0.5; 0.5 -6]), q_switch = Categorical([0.25, 0.75]), q_m = MvNormalMeanPrecision([2.0, -3.0], [2.1 -1.0; -1.0 3.0])
                ),
                output = WishartFast(3.25, [2.53919 8.28558; 8.28558 30.38920])
            ),
            (
                input = (q_out = MvNormalMeanCovariance([-3], [2.0]), q_switch = Bernoulli(0.5), q_m = MvNormalMeanCovariance([5.0], [2.0])),
                output = WishartFast(2.5, fill(34.0, 1, 1))
            )
        ]
    end
end
