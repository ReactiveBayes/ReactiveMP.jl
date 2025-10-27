@testitem "rules:GammaMixture:a" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    import ReactiveMP: @test_rules
    import ReactiveMP: GammaShapeLikelihood
    # Tests for rule: GammaMixture((:a, k), Marginalisation)
    @testset "Variational : (:a, k=1)" begin
        p1 = 0.7
        γ1 = 0.7 * (mean(log, GammaShapeRate(2.0, 1.0)) + mean(log, GammaShapeRate(1.0, 2.0)))
        p2 = 0.5
        γ2 = 0.5 * (mean(log, GammaShapeRate(3.0, 2.0)) + mean(log, GammaShapeRate(1.0, 3.0)))
        println(typeof(p1), " ", typeof(γ1))
        println(typeof(p2), " ", typeof(γ2))
        @test_rules [check_type_promotion = false] GammaMixture{2}((:a, k = 1), Marginalisation) [
            (
                input = (
                    q_out = GammaShapeRate(2.0, 1.0),
                    q_switch = Categorical([0.7, 0.3]),
                    q_b = GammaShapeRate(1.0, 2.0)
                ),
                output = GammaShapeLikelihood(p1, γ1)
            ),
            (
                input = (
                    q_out = GammaShapeRate(3.0, 2.0),
                    q_switch = Bernoulli(0.5),
                    q_b = GammaShapeRate(1.0, 3.0)
                ),
                output = GammaShapeLikelihood(p2, γ2)
            )
        ]
    end
end
